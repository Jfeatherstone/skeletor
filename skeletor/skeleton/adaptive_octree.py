r"""
This performs skeletonization using an adaptive octree structure
that chooses the box partitioning automatically.
"""

# TODO: Update the adjacency matrix to be sparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

import os
import time, sys
import itertools
import colour

import open3d as o3d

from anytree import Node

from skeletor.utils import detectConnectedStructures
from skeletor.utils import plotBox, getBoxLines
from skeletor.spatial import pathIntegralAlongField, lineIntersection, partitionIntoBoxes
from skeletor.spatial import angularHistogramAroundPoint, findDominantHistogramDirections, discretizeDirectionVectors, courseGrainField

from .skeleton_base import SkeletonBase

class AdaptiveBox(Node):

    def __init__(self, boxCorner=None, boxSize=None, points=None, neighbors=None, parent=None, boxName=None):
        """
        A box that may contain points as a part of an octree.

        Parameters
        ---------
        boxCorner : numpy.ndarray[d]
            Coordinates of the "bottom left" corner of the
            box (however that is defined in your space).

        boxSize : numpy.ndarray[d]
            Extents of the box in each spatial direction.
            Does not necessarily have to be uniform.

        points : numpy.ndarray[N,d] or None
            Array of points that belong within this box.

            Will only add points that are properly contained
            within this box, meaning you can pass a full list
            of points and only the ones inside the box (as
            defined by boxCorner and boxSize) will be added.

        neighbors : list(Box) or None
            A list of boxes to be considered neighbors.
        """
        if boxName:
            self.boxName = boxName
        else:
            self.boxName = time.perf_counter()

        Node.__init__(self, self.boxName, parent)
        
        if hasattr(points, '__iter__'):
            self.points = np.array(points)
        else:
            self.points = np.array([], dtype=np.float64)

        if hasattr(neighbors, '__iter__'):
            self.neighbors = list(neighbors)
        else:
            self.neighbors = []
  
        # If we are given a specific box corner, set that
        if hasattr(boxCorner, '__iter__'):
            self.boxCorner = boxCorner
        else:
            # Otherwise, we can calculate it from the data
            # (add a bit of padding)
            self.boxCorner = np.min(points, axis=0)

        self.dim = np.shape(self.points)[-1] if len(self.points) > 0 else len(self.boxCorner)

        # If we are given a specific size, set that
        if hasattr(boxSize, '__iter__'):
            self.boxSize = boxSize
        else:
            # Otherwise, calculate it from data
            self.boxSize = np.array([np.max(points[:,i]) - np.min(points[:,i]) for i in range(self.dim)])

        # Will be set in _update()
        self.vertexDirection = None
        self.moments = np.array([])
        self.momentDirs = np.array([])
        self.discreteMoments = np.zeros(2*np.shape(self.points)[-1])
        self.basis = np.zeros(np.shape(self.points)[-1])


        self._update()


    def _update(self):
        """

        """
        if len(self.points) > 0:
            # Compute dominant directions
            hist, axes = angularHistogramAroundPoint(self.points, np.mean(self.points, axis=0))
            peakDirections = np.array(findDominantHistogramDirections(hist, axes))

            if len(peakDirections) > 0:
                # Compute magnitudes and directions of moments
                self.moments = np.sqrt(np.sum(peakDirections**2, axis=-1))
                self.momentDirs = (peakDirections.T / self.moments).T
                
                self.discreteMoments = discretizeDirectionVectors(peakDirections)

        neighborDirections = [self.getBoxCentroid() - n.getBoxCentroid() for n in self.neighbors]

        self.vertexDirection = discretizeDirectionVectors(neighborDirections)

    def __repr__(self):
        return f'Box: {self.boxName}\nCorner: {self.getBoxCorner()}\nPoints: {len(self.points)}'#\nDominant directions: {self.dominantDirections}'

    def getBoxCenter(self):
        """
        Return the geometric center of the box.
        """
        return self.boxCorner + self.boxSize/2
    
    def getBoxCorner(self):
        """
        Return the geometric corner of the box.
        """
        return self.boxCorner

    def getBoxSize(self):
        """
        Return the geometric size of the box.
        """
        return self.boxSize

    def getBoxDiagonal(self):
        """
        Return the diagonal length of the box.
        """
        return np.sqrt(np.sum((self.boxSize**2)))

    def getBoxCentroid(self):
        """
        Return the center of mass of all points included
        in the box.
        """
        if self.containsPoints():
            return np.mean(self.points, axis=0)

        return None

    def getBoxFaceCenters(self):
        r"""
        Return the center points for all
        \( 2 d \) faces of the box.
        """
        # Vectors normal to each face (with magnitude
        # equal to half of the box extent in that direction)
        faceDirections = np.concatenate([(self.boxSize[i]*np.eye(self.dim)[i]/2, -self.boxSize[i]*np.eye(self.dim)[i]/2) for i in range(self.dim)], axis=0)
        # Now add directions to the box center
        faceCenters = self.getBoxCenter() + faceDirections

        return faceCenters

    def containsPoints(self):
        """
        Whether or not the box contains points.
        """
        return len(self.points) > 0

    def getPoints(self):
        """
        """
        return self.points

    def getNumPoints(self):
        """
        """
        return len(self.points)

    def addPoints(self, points):
        """
        """
        self.points = np.concatenate((self.points, points), axis=0)
        self._update()

    def addNeighbors(self, neighbors):
        """

        """
        if hasattr(neighbors, '__iter__'):
            self.neighbors += [n for n in neighbors if n not in self.neighbors]
        else:
            self.neighbors += [neighbors]

        self._update()


    def shrink(self):
        """
        """
        newBounds = np.array([(np.min(self.points[:,i]), np.max(self.points[:,i])) for i in range(self.dim)])

        # Small padding
        self.boxSize = (newBounds[:,1] - newBounds[:,0])*(1 + 1e-3)
        self.boxCorner = newBounds[:,0] - self.boxSize*1e-3

        self._update()



    def divide(self, numBoxes):
        """

        """
        # Calculate which axes the points are generally
        # oriented along
        # See spatial.py for information on why these are added like this
        # (briefly, it's because the order of the discretized directions
        # is [x, -x, y, -y, z, -z, ...])
        axisStrengths = self.discreteMoments[::2] + self.discreteMoments[1::2]

        axisReductionFactors = np.ones(len(axisStrengths))
        # The strongest axis shouldn't be reduced unless we have to
        # Cut axes in half starting with the weakest ones and working
        # our way up (and repeating if necessary)
        while np.product(axisReductionFactors) < numBoxes:
            for i in np.argsort(axisStrengths):
                axisReductionFactors[i] *= 2

                if np.product(axisReductionFactors) == numBoxes:
                    break

        # Tiny amount of padding
        newBoxSize = (self.boxSize / axisReductionFactors)*(1+1e-3)
        
        # Compute the new corners of the boxes
        newCornerPoints = [self.getBoxCorner()[i] + np.arange(axisReductionFactors[i])*newBoxSize[i] for i in range(len(axisReductionFactors))]
        newCorners = np.array(list(itertools.product(*newCornerPoints)))
        
        # Partition the points into each new box
        # First compute tuples like (x,y,z)
        newBoxMembership = np.int64(np.floor((self.points - self.getBoxCorner()) / newBoxSize))
        # Then flatten to integer indexing like i
        conversionDict = dict(zip([tuple(np.int64(t)) for t in np.round((newCorners - self.getBoxCorner())/newBoxSize)], np.arange(len(newCorners))))

        #try:
        newBoxMembership = np.array([conversionDict[tuple(m)] for m in newBoxMembership])
        #except Exception as e:
        #    print(conversionDict)
        #    print(newBoxMembership)
        #    raise Exception()

        # Now create the new boxes
        newBoxes = []
        for i in range(numBoxes):
            if len(newBoxMembership[newBoxMembership == i]) > 0:
                newBoxes.append(AdaptiveBox(newCorners[i], newBoxSize, self.points[newBoxMembership == i],
                                            neighbors=self.neighbors, parent=self.parent, boxName=f'{self.boxName}_{i}'))

        return newBoxes


    def plot(self, ax=None, drawBounds=True, drawPoints=True, drawCentroid=True, drawMoments=False, drawFaceCenters=False, **kwargs):
        """
        """
        if not 'c' in kwargs:
            kwargs["c"] = 'tab:blue'

        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d' if self.dim == 3 else None)

        if drawBounds:
            fig = plotBox(self.getBoxCorner(), self.getBoxSize(), ax=ax, alpha=.3, **kwargs)

        if drawPoints:
            plt.gca().scatter(*self.points.T, label=self.boxName, alpha=.15, **kwargs)

        if drawCentroid:
            plt.gca().scatter(*self.getBoxCentroid().T, c='tab:red', s=20)

        if drawMoments and len(self.moments) > 0:
            momentScale = (np.max(self.points, axis=0) - np.min(self.points, axis=0))/4
            for m in self.momentDirs:
                plt.gca().plot(*list(zip(self.getBoxCentroid(), self.getBoxCentroid() + m*momentScale)))

        if drawFaceCenters:
            plt.gca().scatter(*self.getBoxFaceCenters().T, s=30)

        return plt.gcf()



def _medianDistanceConnectionCriteria(boxPoints, neighborPoints, threshold):
    """
    Compute the median distances for each cell to the
    plane defined by the cells's centroids, and the
    normal vector between the two centroids.

    Calculate the same quantity for the set of
    points from merging the two cells.

    If the latter quantity (times the threshold) is
    less than the minimum of the two former quantities,
    the two cells are neighbors.
    """
    # Calculate centroids
    boxCentroid = np.mean(boxPoints, axis=0)
    neighborCentroid = np.mean(neighborPoints, axis=0)
    separation = neighborCentroid - boxCentroid 
    combinedCentroid = boxCentroid + separation/2

    # Find the plane in between the two centroids
    planeNormal = combinedCentroid / np.sqrt(np.sum(combinedCentroid**2))

    # Calculate d1, d2, and d12 (from the paper)
    d1 = _calculateSquaredMedianDistance(boxPoints, boxCentroid, planeNormal)
    d2 = _calculateSquaredMedianDistance(neighborPoints, neighborCentroid, planeNormal)
    d12 = _calculateSquaredMedianDistance(np.concatenate((boxPoints, neighborPoints), axis=0), combinedCentroid, planeNormal)
    
    return d12*threshold <= min(d1, d2)
    

    
def _calculateSquaredMedianDistance(points, planePoint, planeNormal):
    """
    Compute the squared median distance for points to a plane.
    """

    d = np.dot(planeNormal, planePoint)
    planeDistance = (np.dot(planeNormal, np.transpose(points)) - d).T**2

    return np.median(planeDistance)



class AdaptiveOctree(SkeletonBase):

    def __init__(self, points, nBoxes=1, minPointsPerBox=1, maxPointsPerBox=.05, verbose=False, debug=False):
        """
        """
        
        # Super class will take care of the basic stuff
        super().__init__(points, verbose, debug)

        if maxPointsPerBox < 1:
            maxPointsPerBox = int(len(points)*maxPointsPerBox)

        ################################
        # Partition Points Into Boxes
        ################################
        boxSize, boxMembership, boxCorners = partitionIntoBoxes(self.points, nBoxes, returnIndices=True) 

        numFilledBoxes = int(np.max(boxMembership)) + 1

        self.boxes = []
        #rootBox = Box(np.zeros(np.shape(points)[-1]), )

        # Create the boxes
        for i in range(numFilledBoxes):
            self.boxes.append(AdaptiveBox(boxCorners[i], boxSize, points[boxMembership == i], boxName=i))

        # Split any boxes with a large amount of points
        while True:
            # Check if any box can be reduced without splitting
            # By setting the divide to 2^(d-1), this forces anisotropy along
            # the axis of the strongest moment of the set of points.
            # eg. split into 4 in 3D, or 2 in 2D
            #for i in range(len(self.boxes)):
            #    while True:
            #        dividedBoxes = self.boxes[i].divide(2**(self.dim-1))
            #        # If, after dividing the box, we only get one new box (with enough
            #        # points in it to be relevant), we just directly replace the old box.
            #        if len([b for b in dividedBoxes if b.getNumPoints() > minPointsPerBox]) == 1:
            #            self.boxes[i] = dividedBoxes[np.argmax([b.getNumPoints() for b in dividedBoxes])]
            #        else:
            #            break
            # Shrink every box
            for i in range(len(self.boxes)):
                self.boxes[i].shrink()

            # Find which boxes have too many points in them
            boxesToSplit = [i for i in range(len(self.boxes)) if self.boxes[i].getNumPoints() > maxPointsPerBox]
            if len(boxesToSplit) == 0:
                break
            
            # Add all of the boxes that don't need to be split to the new list
            newBoxes = [self.boxes[i] for i in range(len(self.boxes)) if i not in boxesToSplit]

            for i in boxesToSplit:
                # How many boxes to split into; 2^d is the most basic choice,
                # since it will result in 8 equal subvolumes in 3D or 4 equal
                # subareas in 2D
                # (otherwise, any choice here should be a power of 2)
                numSplitBoxes = 2**self.dim
                newBoxes += self.boxes[i].divide(numSplitBoxes)

            self.boxes = newBoxes
      

        # Get rid of empty boxes (or boxes with too few points)
        self.boxes = [b for b in self.boxes if b.getNumPoints() > minPointsPerBox]
        
        if self.debug:
            print(f'Partioned space into {len(self.boxes)} boxes, all of which contain points.')

        ################################
        # Find Neighbors of Boxes
        ################################
        # Parameters of neighbor finding

        # The upper limit of difference in angle of the
        # principle moments of two boxes for them to be considered
        # proper neighbors.
        angleThreshold = .2 # Radians

        # Threshold for whether two boxes could potentially be neighbors
        # **Relative to the diagonal length of a given box**
        initialDistanceThreshold = 1/2
        
        # Factor for whether two boxes are actually neighbors,
        # when looking if the moment of one box passes very close
        # to the centroid of another
        # **Relative to the average nearest neighbor distance**
        asymmetricDistanceThreshold = 2

        # Factor for whether two boxes are neighbors based simply
        # on having centroids that are very close to each other.
        # **Relative to the average nearest neighbor distance**
        mergeDistanceThreshold = 5

        # Threshold for determining neighbors via point density
        # along an edge cut. For exact details, see Condition 4
        # calculation below, but should be in the region [0,1]
        # with a value of 1 being the most strict, and 0 being
        # perfectly permissive.
        pointDensityThreshold = 0.7

        # Compute the course-grained point density field, as this will
        # be used in determining if boxes are proper neighbors
        # To do this, we need to know the scale over which we
        # should course grain; we'll take this as the average interparticle
        # distance.
        kdTree = KDTree(self.points)
        nnDistances, nnIndices = kdTree.query(points, 2)
        self.avgNNDistance = np.mean(nnDistances[:,1])

        if self.verbose:
            print(f'Found average nearest neighbor distance: {self.avgNNDistance}')

        ################################
        # Compute face center distances
        ################################

        # Compute which boxes could be neighbors based on how close their faces are
        # A flattened list of every face of every box; total length should be 2*dim*N
        allBoxFaces = np.array([fc for b in self.boxes for fc in b.getBoxFaceCenters()])
        # Our boxes will have 2*dim faces (4 in 2D, 6 in 3D)
        # This array looks like (in 2D) [0,0,0,0,1,1,1,1,2,2,2,2,...]
        faceIdentities = np.array([i for i in range(len(self.boxes)) for j in range(2*self.dim)])

        # We will use a kd tree to check distances
        kdTree = KDTree(allBoxFaces)
      
        # It is hard to define a single distance threshold because we may
        # have very differently sized boxes, so we change the threshold for
        # each box. cf. basic_octree 
        potentialNeighbors = []
        for i in range(len(self.boxes)):

            # Diagonal length over two seems reasonable, but we can control it
            # with a kwarg
            distanceThreshold = np.sqrt(np.sum(self.boxes[i].getBoxSize()**2)) * initialDistanceThreshold

            # This is a list of lists, [[i,j], [k,l,m], ...]
            # where i and j are indices of the faces that are neighbors to the 0th face,
            # k, l and m are neighbors to the 1st face, etc.
            faceNeighbors = kdTree.query_ball_point(allBoxFaces[faceIdentities == i], distanceThreshold)

            # Collapse all into a flat list, and switch indexing to box
            boxNeighbors = [faceIdentities[n] for f in faceNeighbors for n in f]
            boxNeighbors = np.unique(boxNeighbors)

            # Add to the running list
            potentialNeighbors.append(boxNeighbors)

        # Histogram of the number of neighbors
        if self.debug:
            # -1 to account for self counting
            plt.hist([len(p)-1 for p in potentialNeighbors], bins=np.arange(2*self.dim+1) - .5)
            plt.xlabel('Potential Neighbors')
            plt.title('Histogram of Potential Neighbors')
            plt.show()

        # DEBUG: add potential neighbors as actual neighbors
        #for i in range(len(self.boxes)):
        #    self.boxes[i].addNeighbors([self.boxes[n] for n in potentialNeighbors[i]])
        

        #######################################################
        # Check if potential neighbors are actual neighbors
        #######################################################

        for i in range(len(self.boxes)):

            for j in potentialNeighbors[i]:
                
                # Skip self connections
                if j == i:
                    continue

                # The neighbor doesn't necessisarily have to have
                # moments, since the second condition below can be
                # satisifed with 0 moments

                # Skip if already neighbors
                if self.boxes[j] in self.boxes[i].neighbors:
                    continue

                # Separation between centroids
                separation = self.boxes[i].getBoxCentroid() - self.boxes[j].getBoxCentroid()
                # Normalize (norm stands for normalized, not the norm, which would be the
                # opposite meaning... maybe not a great convention)
                separationNorm = separation / np.sqrt(np.sum(separation**2))

                # To be considered a real neighbor, one of the following must be true:
                #    1 The centroids of the two boxes are very close to each other
                #    OR
                #    2 Both boxes have moments that align (up to a threshold) with the
                #      separation vector between the centroids
                #    OR
                #    3 One box has a moment that aligns (up to a threshold) with the
                #      separation vector between the centroids, and this moment passes
                #      very close to the centroid of the other box
                #    OR
                #    4 The path integral of the point density field
                #      along the separation vector is greater than a threshold.

                #############################
                #       Condition 1
                #############################
                if np.sqrt(np.sum(separation**2)) < mergeDistanceThreshold*self.avgNNDistance:
                    if self.verbose:
                        print(f'Registered neighbor by proximity of centroids ({i}, {j})')

                    # Add as true neighbors
                    self.boxes[i].addNeighbors(self.boxes[j])
                    self.boxes[j].addNeighbors(self.boxes[i])
                    continue
                
                #############################
                #       Condition 2
                #############################
                if len(self.boxes[i].moments) > 0:
                    # Compute projections and then angles of moments compared
                    # to separation vector
                    currBoxMomentProjections = np.dot(self.boxes[i].momentDirs, separationNorm)
                    # Correct for periodicity
                    currBoxMomentAngles = [a if a < np.pi/2 else np.pi - a for a in np.abs(np.arccos(currBoxMomentProjections))]
                else:
                    currBoxMomentAngles = np.array([])
                
                if len(self.boxes[j].moments) > 0:
                    neighborMomentProjections = np.dot(self.boxes[j].momentDirs, separationNorm)
                    neighborMomentAngles = [a if a < np.pi/2 else np.pi - a for a in np.abs(np.arccos(neighborMomentProjections))]
                else:
                    neighborMomentAngles = np.array([])

                # (Have to make sure we actually have some moments for the neighbor)
                if len(self.boxes[i].moments) > 0 and np.min(currBoxMomentAngles) < angleThreshold and len(self.boxes[j].moments) > 0 and np.min(neighborMomentAngles) < angleThreshold:
                    if self.verbose:
                        print(f'Registered neighbor by alignment of principle moments ({i}, {j})')

                    # Add as true neighbors
                    self.boxes[i].addNeighbors(self.boxes[j])
                    self.boxes[j].addNeighbors(self.boxes[i])
                    continue


                #############################
                #       Condition 3
                #############################

                # Much smaller distance threshold 
                distanceThreshold = self.avgNNDistance*asymmetricDistanceThreshold

                if len(self.boxes[i].moments) > 0:
                    # Compute point on the plane defined by the direction of the moment and the other box's centroid
                    currBoxClosestPass = np.dot(self.boxes[i].momentDirs[np.argmin(currBoxMomentAngles)], self.boxes[j].getBoxCentroid())
                    # Compute distance betewen this point and the other box's centroid
                    currBoxClosestPassDistance = np.sqrt(np.sum((currBoxClosestPass - self.boxes[j].getBoxCentroid())**2))
                else:
                    # Arbitrarily high number
                    currBoxClosestPassDistance = np.inf
               
                if len(self.boxes[j].moments) > 0:
                    # Same for switching the roles of the current and neighbor box
                    neighborClosestPass = np.dot(self.boxes[j].momentDirs[np.argmin(neighborMomentAngles)], self.boxes[i].getBoxCentroid())
                    neighborClosestPassDistance = np.sqrt(np.sum((neighborClosestPass - self.boxes[i].getBoxCentroid())**2))
                else:
                    neighborClosestPassDistance = np.inf

                if (len(self.boxes[i].moments) > 0 and np.min(currBoxMomentAngles) < angleThreshold and currBoxClosestPassDistance < distanceThreshold) or (len(self.boxes[j].moments) > 0 and np.min(neighborMomentAngles) < angleThreshold and neighborClosestPassDistance < distanceThreshold):
                    if self.verbose:
                        print(f'Registered neighbor by alignment of principle moment with centroid ({i}, {j})')

                    # Add as true neighbors
                    self.boxes[i].addNeighbors(self.boxes[j])
                    self.boxes[j].addNeighbors(self.boxes[i])
                    continue

                #############################
                #       Condition 4
                #############################
                # While we say we want to compute the path integral here, course
                # graining the entire field is very expensive (both in time and
                # memory).
                # Instead, we get a parametric representation of the separation
                # line segment between the boxes, and then compute the parameter
                # values of each point on that line. By taking the histogram
                # of these parameter values, we essentially have a line cut
                # through the density field.
                nearbyPoints = np.concatenate((self.boxes[i].points, self.boxes[j].points))
                tArr = np.dot(separationNorm, (nearbyPoints - self.boxes[j].getBoxCentroid()).T)

                # Find the parameter value for the other centroid
                # ie. points with a parameter value larger than
                # this value are no longer between the two centroids,
                # so we don't care about them.
                # The minimum is simply 0, so we care about the
                # range [0, maxTValue]
                # Since we use the normalized separation vector above, this is
                # just the length of the line segment.
                maxTValue = np.sqrt(np.sum(separation**2))

                # Remove values outside of that range
                keepIndices = np.where((tArr >= 0) & (tArr <= maxTValue))[0]
                tArr = tArr[keepIndices]

                # If there are no points between the two centroids, we
                # can't continue
                if len(tArr) == 0:
                    continue

                # Now we can calulate the distance from each of those points
                # to the line
                pointsProjectedOnLine = self.boxes[j].getBoxCentroid() + np.multiply.outer(tArr, separationNorm)
                pointDistanceToLine = np.sqrt(np.sum((nearbyPoints[keepIndices] - pointsProjectedOnLine)**2, axis=-1))

                # We don't want a step size lower than our nearest neighbor distance
                pathPoints = int(maxTValue / self.avgNNDistance)
                # Just for more clear code below
                pathStepSize = self.avgNNDistance

                # We weight our point density by the perpendicular distance of each
                # point to the line, since nearby points should count more.
                weights = pointDistanceToLine / (np.max(pointDistanceToLine) + 1e-4)
                weights = 1 / (1 + weights)
                weights /= np.max(weights)

                pointDensityAlongLine, binEdges = np.histogram(tArr, pathPoints, weights=weights)

                #plt.plot(pointDensityAlongLine)
                #plt.show()
                    
                # Next we smooth this point density a little bit, and
                # then binarize
                # Smoothing is the simplest possible approach, just a kernel of 3
                smoothingKernel = 3 # Should be odd
                #print(list(range(-(smoothingKernel-1)//2, (smoothingKernel+1)//2)))

                pointDensityAlongLine = np.sum([np.roll(pointDensityAlongLine, i) for i in range(-(smoothingKernel-1)//2, (smoothingKernel+1)//2)], axis=0)
                pointDensityAlongLine /= smoothingKernel

                # Though we have to through away our edges as the roll operation expects periodic
                # boundary conditions
                pointDensityAlongLine = pointDensityAlongLine[1:-1]

                #plt.plot(pointDensityAlongLine)
                #plt.show()

                # Clip the upper values; since we will sum in a second,
                # this means that very high density regions can only
                # counteract low density regions by a little bit.
                #pointDensityAlongLine[pointDensityAlongLine > 1] = 1

                # If we have any points that are less than a small value along
                # the line, that means we have gaps.
                # 0.50 is quite arbitrary, but it should be a number near 1.
                # Exactly 1 would mean we allow 0 points that have a value less
                # than 1.
                #print(np.sum(pointDensityAlongLine) / pathPoints)
                if np.sum(pointDensityAlongLine) / pathPoints >= pointDensityThreshold:
                    if self.debug:
                        print(f'Registered neighbor by path integral threshold ({i}, {j})')

                    self.boxes[i].addNeighbors(self.boxes[j])
                    self.boxes[j].addNeighbors(self.boxes[i])
                    continue

                if self.verbose:
                    print(f'Couldn\'t establish neighbors {i} and {j}')

                if self.debug:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d' if self.dim == 3 else None)
                    self.boxes[i].plot(ax=ax, drawMoments=True)
                    self.boxes[j].plot(ax=ax, drawMoments=True)
                    #ax.set_box_aspect(aspect=(1, 1, 1))
                    plt.axis('square')
                    fig.tight_layout()
                    plt.show()


    def generateSkeleton(self):
        """
        Generate the skeleton representation.
        """
        ##########################################
        #       Compute the rough skeleton
        ##########################################
        self.skeletonPoints = np.array([b.getBoxCentroid() for b in self.boxes])

        self.skeletonAdjMat = np.zeros((len(self.skeletonPoints), len(self.skeletonPoints)))

        # TODO: Switch to sparse representation
        for i in range(len(self.skeletonPoints)):
            self.skeletonAdjMat[i] = np.array([self.boxes[ind] in self.boxes[i].neighbors for ind in range(len(self.skeletonPoints))], dtype=np.int64)

        if self.verbose:
            print(f'Found skeleton with {len(self.skeletonPoints)} points.')
        
        return self.skeletonPoints, self.skeletonAdjMat


    def _o3d_extra_plot_geometries(self, plotPoints, plotSkeleton, plotBoxes=False, plotMoments=False):
        """
        """
        boxPoints = []

        if plotPoints:
            for i in range(len(self.boxes)):
                boxPointCloud = o3d.geometry.PointCloud()
                boxPointCloud.points = o3d.utility.Vector3dVector(self.boxes[i].points)
                boxPointCloud.paint_uniform_color(colour.Color(pick_for=i).rgb)

                boxPoints.append(boxPointCloud)

        boxOutlines = []

        if plotBoxes:

            for i in range(len(self.boxes)):
                # 12 edges to a box in 3d, so 24 points needed to define them
                # This isn't efficient in the sense it doesn't reuse corners as you
                # could, but its such a small number of edges it shouldn't matter
                allCornerPoints = getBoxLines(self.boxes[i].getBoxCorner(), self.boxes[i].getBoxSize()).reshape(24,3)

                boxEdges = o3d.geometry.LineSet()
                boxEdges.points = o3d.utility.Vector3dVector(allCornerPoints)
                boxEdges.lines = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(allCornerPoints))[::2]])
                boxEdges.paint_uniform_color(colour.Color(pick_for=i).rgb)

                boxOutlines.append(boxEdges)

        boxMoments = []

        if plotMoments:
            
            for i in range(len(self.boxes)):
                if len(self.boxes[i].momentDirs) == 0:
                    continue

                momentScale = (np.max(self.boxes[i].points, axis=0) - np.min(self.boxes[i].points, axis=0))
                
                momentPoints = [self.boxes[i].getBoxCentroid()]

                for m in self.boxes[i].momentDirs:
                    momentPoints.append(self.boxes[i].getBoxCentroid() + m*momentScale)

                momentIndices = [[0, i] for i in range(1, len(momentPoints))]

                boxMomentLines = o3d.geometry.LineSet()
                boxMomentLines.points = o3d.utility.Vector3dVector(momentPoints)
                boxMomentLines.lines = o3d.utility.Vector2iVector(momentIndices)
                boxMomentLines.paint_uniform_color(colour.Color(pick_for=i).rgb)

                boxMoments.append(boxMomentLines)

        return boxPoints + boxOutlines + boxMoments


    def _mpl_extra_plot_geometries(self, plotPoints, plotSkeleton, ax, plotBoxes=False, plotMoments=False):
        """
        WIP
        """
        # TODO

        return None

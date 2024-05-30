r"""
This performs skeletonization using an adaptive octree structure
that chooses the box partitioning automatically.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

import os
import time, sys
import itertools
import colour

from anytree import Node

from skeletor.utils import partitionIntoBoxes, angularHistogramAroundPoint, findDominantHistogramDirections, discretizeDirectionVectors, courseGrainField, pathIntegralAlongField
from skeletor.utils import detectConnectedStructures
from skeletor.utils import plotBox

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
            ax = fig.add_subplot(projection='3d')

        if drawBounds:
            fig = plotBox(self.getBoxCorner(), self.getBoxSize(), ax=ax, alpha=.3, **kwargs)

        if drawPoints:
            plt.gca().scatter(*self.points.T, label=self.boxName, alpha=.15, **kwargs)

        if drawCentroid:
            plt.gca().scatter(*self.getBoxCentroid().T, c='tab:red', s=20)

        if drawMoments and len(self.moments) > 0:
            momentScale = np.max(self.points, axis=0) - np.min(self.points, axis=0)/4
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



class AdaptiveOctree():

    def __init__(self, points, nBoxes=1, minPointsPerBox=1, maxPointsPerBox=.05, debug=False):
        """
        """
        
        self.points = points
        self.dim = np.shape(points)[-1]
        self.debug = debug
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
        angleThreshold = .5 # Radians

        # Factor to multiply the average box size by to determine 
        # whether two boxes could potentially be neighbors
        # Much higher than in non-adaptive case
        firstDistanceThresholdFactor = 10
        
        # Factor to multiply the average box size by to determine 
        # whether two boxes are actually neighbors
        secondDistanceThresholdFactor = 3

        pathIntegralThresholdFactor = 2

        # Compute the course-grained point density field, as this will
        # be used in determining if boxes are proper neighbors
        # To do this, we need to know the scale over which we
        # should course grain; we'll take this as the average interparticle
        # distance.
        kdTree = KDTree(self.points)
        nnDistances, nnIndices = kdTree.query(points, 2)
        avgNNDistance = np.mean(nnDistances[:,1])

        if debug:
            print(f'Found average nearest neighbor distance: {avgNNDistance}')

        cgDensityField, cgCorner = courseGrainField(self.points, latticeSpacing=avgNNDistance, returnCorner=True)
        avgDensity = np.mean(cgDensityField) # * nBoxes/numFilledBoxes
        # TODO: Maybe scale this average by the ratio of occupied boxes,
        # to try and account for any large regions with no points

        ################################
        # Compute face center distances
        ################################

        # Compute which boxes could be neighbors based on how close their faces are
        # A flattened list of every face of every box; total length should be 2*dim*N
        allBoxFaces = np.array([fc for b in self.boxes for fc in b.getBoxFaceCenters()])
        # Our boxes will have 2*dim faces (4 in 2D, 6 in 3D)
        # This array looks like (in 2D) [0,0,0,0,1,1,1,1,2,2,2,2,...]
        faceIdentities = np.array([i for i in range(len(self.boxes)) for j in range(2*self.dim)])
        
        distanceThreshold = avgNNDistance*firstDistanceThresholdFactor

        # Find all pairs of faces that are within the distance threshold to each other
        kdTree = KDTree(allBoxFaces)
        # This is a list of lists, [[i,j], [k,l,m], ...]
        # where i and j are indices of the faces that are neighbors to the 0th face,
        # k, l and m are neighbors to the 1st face, etc.
        potentialNeighbors = kdTree.query_ball_point(allBoxFaces, distanceThreshold)

        # Remove self neighbors
        # This is dealt with later
        #potentialNeighbors = np.array([[ind for ind in potentialNeighbors[n] if ind != n] for n in range(len(potentialNeighbors))], dtype=object)

        # Convert from indexing the faces to indexing the boxes
        #potentialNeighbors = [[np.int64(np.floor(ind/(self.dim*2))) for n in potentialNeighbors[faceIdentities == i] for ind in n] for i in range(len(self.boxes))]
        potentialNeighbors = np.array([[faceIdentities[ind] for ind in potentialNeighbors[i]] for i in range(len(faceIdentities))], dtype='object')

        # And combine all 2*d faces into the same box
        potentialNeighbors = [np.concatenate(potentialNeighbors[np.where(faceIdentities == i)]) for i in range(len(self.boxes))]

        # Remove duplicates, in case, eg. the face of one box is a neighbor
        # to two faces of another box.
        potentialNeighbors = [np.unique(n) for n in potentialNeighbors]

        # Histogram of the number of neighbors
        if debug:
            # -1 to account for self counting
            plt.hist([len(p)-1 for p in potentialNeighbors], bins=np.arange(2*self.dim+1) - .5)
            plt.xlabel('Potential Neighbors')
            plt.title('Histogram of Potential Neighbors')
            plt.show()

        # Check if any of the potential neighbors are actually neighbors
        # based on alignment of moments and other criteria
        # Much smaller distance threshold 
        distanceThreshold = avgNNDistance*secondDistanceThresholdFactor

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
                # Normalize
                separationNorm = separation / np.sqrt(np.sum(separation**2))

                # To be considered a real neighbor, one of the following must be true:
                #    1 Both boxes have moments that align (up to a threshold) with the
                #      separation vector between the centroids
                #    OR
                #    2 One box has a moment that aligns (up to a threshold) with the
                #      separation vector between the centroids, and this moment passes
                #      very close to the centroid of the other box
                #    OR
                #    3 The path integral of the course grained point density field
                #      along the separation vector is greater than a threshold.
                #
                #

                # Check condition 1 
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
                    if self.debug:
                        print('Registered neighbor by alignment of principle moments')
                    # Add as true neighbors
                    self.boxes[i].addNeighbors(self.boxes[j])
                    self.boxes[j].addNeighbors(self.boxes[i])
                    continue

                # Check condition 2
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
                    if self.debug:
                        print('Registered neighbor by alignment of principle moment with centroid')
                    # Add as true neighbors
                    self.boxes[i].addNeighbors(self.boxes[j])
                    self.boxes[j].addNeighbors(self.boxes[i])
                    continue

                # Check condition 3
                # Compute path integral
                pathSteps = 30 # More than enough
                path = self.boxes[i].getBoxCentroid() + np.array([l*separation for l in np.linspace(0, 1, pathSteps)])
                pathIntegral = pathIntegralAlongField(cgDensityField, path, latticeSpacing=avgNNDistance, fieldOffset=cgCorner)
                if pathIntegral/np.sqrt(np.sum(separation**2)) > avgDensity*pathIntegralThresholdFactor:
                    if self.debug:
                        print('Registered neighbor by path integral threshold')
                    self.boxes[i].addNeighbors(self.boxes[j])
                    self.boxes[j].addNeighbors(self.boxes[i])
                    continue


                if self.debug:
                    print(f'Couldn\'t establish neighbors {i} and {j}')
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    self.boxes[i].plot(ax=ax, drawMoments=True)
                    self.boxes[j].plot(ax=ax, drawMoments=True)
                    #ax.set_box_aspect(aspect=(1, 1, 1))
                    plt.axis('square')
                    fig.tight_layout()
                    plt.show()

    
        ########################################
        #   Reduce Connections and Merge Boxes
        ########################################
        
        # Step 1: Try and merge boxes whose centers are
        # very close to one another.
        distanceThreshold = avgNNDistance*5
        pathIntegralThresholdFactor = 2

        for box in self.boxes:
            for neighbor in box.neighbors:
                separation = neighbor.getBoxCentroid() - box.getBoxCentroid()
                if np.sqrt(np.sum(separation**2)) < distanceThreshold:
                    # Check that there is a path of points between the centroids 
                    pathSteps = 30 # More than enough
                    path = box.getBoxCentroid() + np.array([l*separation for l in np.linspace(0, 1, pathSteps)])
                    pathIntegral = pathIntegralAlongField(cgDensityField, path, latticeSpacing=avgNNDistance, fieldOffset=cgCorner)
                    if pathIntegral/np.sqrt(np.sum(separation**2)) > avgDensity*pathIntegralThresholdFactor:
                        print(f'Merge {box} and {neighbor}')
                        break


    def skeleton(self):
        """
        """
        points = np.array([b.getBoxCentroid() for b in self.boxes])

        indices = np.arange(len(points))

        adjMat = np.zeros((len(points), len(points)))
        for i in range(len(indices)):
            adjMat[i] = np.array([self.boxes[ind] in self.boxes[indices[i]].neighbors for ind in indices], dtype=np.int64)

        return points, adjMat


    def plotSkeleton(self, ax=None, plotBoxes=True, centroidKwargs={}, lineKwargs={}):
        """
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d' if self.dim == 3 else None)

        # Draw boxes
        if plotBoxes:
            for b in self.boxes:
                b.plot(ax=ax, drawPoints=False, drawCentroid=False, c=str(colour.Color(pick_for=b)))
       
        points, adjMat = self.skeleton()

        # Draw centroids
        ax.scatter(*points.T, **centroidKwargs)
       
        # Draw lines
        for i in range(len(adjMat)):
            edgeIndices = np.where(adjMat[i] > 0)[0]
            for j in range(len(edgeIndices)):
                ax.plot(*list(zip(points[i], points[edgeIndices[j]])), **lineKwargs)

        return plt.gcf()


    def plot(self, ax=None, plotBoxKwargs={}):
        """
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d' if self.dim == 3 else None)

        for i in range(len(self.boxes)):
            self.boxes[i].plot(ax=ax, c=str(colour.Color(pick_for=i)), **plotBoxKwargs)

        return plt.gcf()


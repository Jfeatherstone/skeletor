r"""
Pretty sure `ask` stands for Adaptive SKeltre... but not totally sure.

Also pretty sure this is my attempt to improve skeltre, but it has been
a while...
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

import os
import time, sys
import itertools
import colour

from anytree import Node

from skeletor.utils import partitionIntoBoxes, angularHistogramAroundPoint, findDominantHistogramDirections, discretizeDirectionVectors
from skeletor.utils import detectConnectedStructures
from skeletor.utils import plotBox

class Box(Node):

    def __init__(self, boxCorner, boxSize, points=None, neighbors=None, parent=None, boxName=None):
        """

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

        self.boxSize = boxSize
        self.boxCorner = boxCorner

        if hasattr(points, '__iter__'):
            self.points = np.array(points)
        else:
            self.points = np.array([], dtype=np.float64)

        if hasattr(neighbors, '__iter__'):
            self.neighbors = list(neighbors)
        else:
            self.neighbors = []
    

        self.dim = np.shape(points)[-1]

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
            hist, thetaBins, phiBins = angularHistogramAroundPoint(self.points)
            peakDirections = np.array(findDominantHistogramDirections(hist, thetaBins, phiBins, normalizeMoments=False))

            if len(peakDirections) > 0:
                # Compute magnitudes and directions of moments
                self.moments = np.sqrt(np.sum(peakDirections**2, axis=-1))
                self.momentDirs = (peakDirections.T / self.moments).T
                
                self.discreteMoments = discretizeDirectionVectors(peakDirections)

        neighborDirections = [self.getBoxCentroid() - n.getBoxCentroid() for n in self.neighbors]

        self.vertexDirection = discretizeDirectionVectors(neighborDirections)

#    def __str__(self):
#        """
#        """
#        return f'{self.getBoxCenter()}'

    def __repr__(self):
        return f'Box: {self.boxName}\nCorner: {self.getBoxCorner()}\nPoints: {len(self.points)}'#\nDominant directions: {self.dominantDirections}'

    def getBoxCenter(self):
        """

        """
        return self.boxCorner + self.boxSize/2
    
    def getBoxCorner(self):
        """

        """
        return self.boxCorner

    def getBoxSize(self):
        """

        """
        return self.boxSize

    def getBoxCentroid(self):
        """

        """
        if self.containsPoints():
            return np.mean(self.points, axis=0)

        return None

    def getBoxFaceCenters(self):
        """
        """
        # Vectors normal to each face (with magnitude
        # equal to half of the box extent in that direction)
        faceDirections = np.concatenate([(self.boxSize[i]*np.eye(self.dim)[i]/2, -self.boxSize[i]*np.eye(self.dim)[i]/2) for i in range(self.dim)], axis=0)
        # Now add directions to the box center
        faceCenters = self.getBoxCenter() + faceDirections

        return faceCenters

    def containsPoints(self):
        """

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

    def mergeWith(self, box):
        """

        """

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

        newBoxSize = self.boxSize / axisReductionFactors
        
        # Compute the new corners of the boxes
        newCornerPoints = [self.getBoxCorner()[i] + np.arange(axisReductionFactors[i])*newBoxSize[i] for i in range(len(axisReductionFactors))]
        newCorners = np.array(list(itertools.product(*newCornerPoints)))
        
        # Partition the points into each new box
        # First compute tuples like (x,y,z)
        newBoxMembership = np.int64(np.floor((self.points - self.getBoxCorner()) / newBoxSize))
        # Then flatten to integer indexing like i
        conversionDict = dict(zip([tuple(np.int64(t)) for t in np.round((newCorners - self.getBoxCorner())/newBoxSize)], np.arange(len(newCorners))))

        try:
            newBoxMembership = np.array([conversionDict[tuple(m)] for m in newBoxMembership])
        except:
            print(newBoxMembership)
            raise Exception()

        # Now create the new boxes
        newBoxes = []
        for i in range(numBoxes):
            if len(newBoxMembership[newBoxMembership == i]) > 0:
                newBoxes.append(Box(newCorners[i], newBoxSize, self.points[newBoxMembership == i],
                                    neighbors=self.neighbors, parent=self.parent, boxName=f'{self.boxName}_{i}'))

        return newBoxes


    def plot(self, ax=None, drawPoints=True, drawStrongestMoment=False, drawFaceCenters=False, **kwargs):
        """
        """
        fig = plotBox(self.getBoxCorner(), self.getBoxSize(), ax=ax, **kwargs)
        
        if drawPoints:
            fig.gca().scatter(self.points[:,0], self.points[:,1], self.points[:,2], 'tab:blue', **kwargs)

        if drawStrongestMoment:
            strongestMoment = self.momentDirs[np.argmax(self.moments)]
            fig.gca().plot(*list(zip(self.getBoxCentroid(), self.getBoxCentroid() + strongestMoment*100)))

        if drawFaceCenters:
            faceCenters = self.getBoxFaceCenters()
            fig.gca().scatter(faceCenters[:,0], faceCenters[:,1], faceCenters[:,2])

        return fig



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



class Octree():

    def __init__(self, points, nBoxes=1, neighborThreshold=1/16, debug=False, adaptiveBoxes=True, minimizeBoxes=True, allowBoxRebase=False, maxPointsPerBox=50, minPointsPerBox=4):
        """
        """
        
        self.points = points
        self.neighborThreshold = neighborThreshold
        self.debug = debug
        self.adaptiveBoxes = adaptiveBoxes
        self.minimizeBoxes = minimizeBoxes
        self.allowBoxRebase = allowBoxRebase
        self.maxPointsPerBox = maxPointsPerBox

        ################################
        # Partition Points Into Boxes
        ################################
        boxSize, boxMembership, boxCorners = partitionIntoBoxes(self.points, nBoxes, returnIndices=True) 
      
        numFilledBoxes = int(np.max(boxMembership)) + 1

        self.boxes = []
        #rootBox = Box(np.zeros(np.shape(points)[-1]), )

        # Create the boxes
        for i in range(numFilledBoxes):
            self.boxes.append(Box(boxCorners[i], boxSize, points[boxMembership == i], boxName=i))

        # If desired, split any boxes with a large amount of points
        if self.adaptiveBoxes:

            while True:
                # Check if any box can be reduced without splitting
                # By setting the divide to 4, this forces anisotropy along
                # the axis of the strongest moment of the set of points.
                for i in range(len(self.boxes)):
                    while True:
                        dividedBoxes = self.boxes[i].divide(4)
                        # If, after dividing the box, we only get one new box (with enough
                        # points in it to be relevant), we just directly replace the old box.
                        if len([b for b in dividedBoxes if b.getNumPoints() > minPointsPerBox]) == 1:
                            self.boxes[i] = dividedBoxes[np.argmax([b.getNumPoints() for b in dividedBoxes])]
                        else:
                            break

                # Find which boxes have too many points in them
                boxesToSplit = [i for i in range(len(self.boxes)) if self.boxes[i].getNumPoints() > self.maxPointsPerBox]
                if len(boxesToSplit) == 0:
                    break
                
                # Add all of the boxes that don't need to be split to the new list
                newBoxes = [self.boxes[i] for i in range(len(self.boxes)) if i not in boxesToSplit]

                for i in boxesToSplit:
                    # How many boxes to split into; 8 is the most basic choice,
                    # since it will result in 8 equal subvolumes in 3D
                    # (otherwise, any choice here should be a power of 2)
                    numSplitBoxes = 8
                    newBoxes += self.boxes[i].divide(numSplitBoxes)

                self.boxes = newBoxes

        if self.minimizeBoxes:
            self.boxes = [b for b in self.boxes if b.getNumPoints() > minPointsPerBox]

        if self.neighborThreshold > 0:
            # Compute which boxes could be neighbors based on how close their faces are
            allBoxFaces = np.array([fc for b in self.boxes for fc in b.getBoxFaceCenters()])
            # 6 refers to the 6 faces of a rectangular solid
            faceIdentities = np.array([i for i in range(len(self.boxes)) for j in range(6)])
           
            # Chosen empirically
            distanceThreshold = np.mean([b.getBoxSize() for b in self.boxes])/3
    
            kdTree = KDTree(allBoxFaces)
            potentialNeighbors = kdTree.query_ball_point(allBoxFaces, distanceThreshold)
    
            # Remove self neighbors
            potentialNeighbors = np.array([[ind for ind in potentialNeighbors[n] if ind != n] for n in range(len(potentialNeighbors))], dtype=object)
    
            potentialNeighbors = [[np.int64(np.floor(ind/6)) for n in potentialNeighbors[faceIdentities == i] for ind in n] for i in range(len(self.boxes))]
            #potentialNeighbors = np.unique(potentialNeighbors, axis=-1)
            if debug:
                plt.hist([len(p) for p in potentialNeighbors])
                plt.show()
            # Check if any of the potential neighbors are actually neighbors
            # based on alignment of moments
            angleThreshold = .5
            distanceThreshold = np.mean([b.getBoxSize() for b in self.boxes])/10
    
            for i in range(len(self.boxes)):
                # We have to have some moments
                if len(self.boxes[i].moments) == 0:
                    continue
    
                for j in potentialNeighbors[i]:
                    # The neighbor doesn't necessisarily have to have
                    # moments, since the second condition below can be
                    # satisifed with 0 moments
    
                    # Skip if already neighbors
                    if self.boxes[j] in self.boxes[i].neighbors:
                        continue
    
                    # Separation between centroids
                    separation = self.boxes[i].getBoxCentroid() - self.boxes[j].getBoxCentroid()
                    # Normalize
                    separation /= np.sqrt(np.sum(separation**2))
    
                    # To be considered a real neighbor, one of the following must be true:
                    #    1 Both boxes have a moment that aligns (up to a threshold) with the
                    #      separation vector between the centroids
                    #    OR
                    #    2 One box has a moment that aligns (up to a threshold) with the
                    #      separation vector between the centroids, and this moment passes
                    #      very close to the centroid of the other box
    
                    # Compute projections and then angles of moments compared
                    # to separation vector
                    currBoxMomentProjections = np.dot(self.boxes[i].momentDirs, separation)
                    currBoxMomentAngles = [a if a < np.pi/2 else np.pi - a for a in np.abs(np.arccos(currBoxMomentProjections))]
                    
                    if len(self.boxes[j].moments) > 0:
                        neighborMomentProjections = np.dot(self.boxes[j].momentDirs, separation)
                        neighborMomentAngles = [a if a < np.pi/2 else np.pi - a for a in np.abs(np.arccos(neighborMomentProjections))]
                    else:
                        neighborMomentAngles = np.array([])
    
                    # Check condition 1
                    # (Have to make sure we actually have some moments for the neighbor)
                    if np.min(currBoxMomentAngles) < angleThreshold and len(self.boxes[j].moments) > 0 and np.min(neighborMomentAngles) < angleThreshold:
                        if self.debug:
                            print('Condition 1')
                        # Add as true neighbors
                        self.boxes[i].addNeighbors(self.boxes[j])
                        self.boxes[j].addNeighbors(self.boxes[i])
                        continue
    
                    # Compute point on the plane defined by the direction of the moment and the other box's centroid
                    currBoxClosestPass = np.dot(self.boxes[i].momentDirs[np.argmin(currBoxMomentAngles)], self.boxes[j].getBoxCentroid())
                    # Compute distance betewen this point and the other box's centroid
                    currBoxClosestPassDistance = np.sqrt(np.sum((currBoxClosestPass - self.boxes[j].getBoxCentroid())**2))
                   
                    if len(self.boxes[j].moments) > 0:
                        # Same for switching the roles of the current and neighbor box
                        neighborClosestPass = np.dot(self.boxes[j].momentDirs[np.argmin(neighborMomentAngles)], self.boxes[i].getBoxCentroid())
                        neighborClosestPassDistance = np.sqrt(np.sum((neighborClosestPass - self.boxes[i].getBoxCentroid())**2))
                    else:
                        neighborClosestPassDistance = 1e10
    
                    # Check condition 2
                    if (np.min(currBoxMomentAngles) < angleThreshold and currBoxClosestPassDistance < distanceThreshold) or (len(self.boxes[j].moments) > 0 and np.min(neighborMomentAngles) < angleThreshold and neighborClosestPassDistance < distanceThreshold):
                        if self.debug:
                            print('Condition 2')
                        # Add as true neighbors
                        self.boxes[i].addNeighbors(self.boxes[j])
                        self.boxes[j].addNeighbors(self.boxes[i])
                        continue
                



        # TODO
        if self.allowBoxRebase:
            pass

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
            ax = fig.add_subplot(projection='3d')

        # Draw boxes
        if plotBoxes:
            for b in self.boxes:
                b.plot(ax=ax, drawPoints=False, c=str(colour.Color(pick_for=b)), alpha=.1)
       
        points, adjMat = self.skeleton()

        # Draw centroids
        ax.scatter(*points.T, **centroidKwargs)
       
        # Draw lines
        for i in range(len(adjMat)):
            edgeIndices = np.where(adjMat[i] > 0)[0]
            for j in range(len(edgeIndices)):
                ax.plot(*list(zip(points[i], points[edgeIndices[j]])), **lineKwargs)

        return plt.gcf()


    def plot(self, ax=None):
        """
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        for i in range(len(self.boxes)):
            self.boxes[i].plot(ax=ax, c=str(colour.Color(pick_for=i)), alpha=.4)

        return plt.gcf()


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
        self.vertexDim = 0
        self.vertexDir = np.zeros(np.shape(self.points)[-1])
        self.vertexNorm = 0

        self.moments = np.array([])
        self.momentDirs = np.array([])
        self.discreteMoments = np.zeros(2*np.shape(self.points)[-1])
        self.basis = np.zeros(np.shape(self.points)[-1])


        self._update()


    def _update(self):
        """

        """
        self.vertexDim = len(self.neighbors)

        neighborDirections = [self.getBoxCenter() - n.getBoxCenter() for n in self.neighbors]
        self.vertexDir = (discretizeDirectionVectors(neighborDirections) > 0).astype(np.int64)
        # Not technically the vertex direction described in the paper, since I separate
        # out the positive and negative x,y,z. To get exactly what is described
        # in the paper:
        # self.vertexDir[::2] - self.vertexDir[1::2]

        # But we will use this to calculate the norm
        self.vertexNorm = np.sum(self.vertexDir[::2] - self.vertexDir[1::2])

    def __repr__(self):
        return f'Box: {self.boxName}\nCorner: {self.getBoxCorner()}\nPoints: {len(self.points)}'#\nDominant directions: {self.dominantDirections}'

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.boxName == other.boxName and len(self.points) == len(other.points)


    def computeDominateDirections():
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

        return self.momentDirs


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
            if neighbors not in self.neighbors:
                self.neighbors += [neighbors]

        self._update()


    def mergeWith(self, box):
        """

        """


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

class Vertex():

    def __init__(self, position, index, neighbors, weight=1):
        self.position = position
        self.index = index
        self.neighbors = neighbors
        self.weight = weight

        self.edgeLabels = [np.array(self.index) - np.array(n) for n in neighbors]
        self.vertexDim = len(np.unique(self.edgeLabels, axis=0))

        self.vertexDir = (discretizeDirectionVectors(self.edgeLabels) > 0).astype(np.int64)
        self.vertexDir = self.vertexDir[::2] - self.vertexDir[1::2]


    def merge(self, other):
        """
        """
        # Weighted average of old positions
        newPosition = (self.weight * self.position + other.weight * other.position) / (self.weight + other.weight)
        # Just take the current index (doesn't really matter)
        newIndex = self.index
        newNeighbors = self.neighbors + other.neighbors
        # Remove duplicates, and the new vertex itself
        newNeighbors = [tuple(t) for t in np.unique(newNeighbors, axis=0) if tuple(t) != self.index and tuple(t) != other.index]
        newWeight = self.weight + other.weight

        return Vertex(newPosition, newIndex, newNeighbors, newWeight)

def _isVPair(vertex1: Vertex, vertex2: Vertex):
    """
    """
    merged = vertex1.merge(vertex2)

    if merged.vertexDim > max(vertex1.vertexDim, vertex2.vertexDim):
        #print('No VPair by dim')
        return False

    # Check that they have a common neighbor
    if not (True in [v in vertex2.neighbors for v in vertex1.neighbors]):
        #print('No VPair by neighbor')
        #print(vertex1.index, vertex1.neighbors)
        #print(vertex2.index, vertex2.neighbors)
        return False
    
    #print('VPair')
    return True

def _isEPair(vertex1: Vertex, vertex2: Vertex):
    """
    """
    merged = vertex1.merge(vertex2)

    # Check if vdim(v1+v2) <= max(vdim(v1), vdim(v2))
    # Note that the statements are all opposite, since we will only
    # return true at the end
    if merged.vertexDim > max(vertex1.vertexDim, vertex2.vertexDim):
        print('No EPair by dim')
        return False

    # Check that we don't have all zeros for vertex1
    if len(np.where(vertex1.vertexDir == 0)[0]) == 3:
        print('No EPair by count')
        return False

    # Check that a nonzero entry in vdir(vertex1) is the same as
    # the connection direction
    connectionDirIndex = np.where(np.abs(np.array(vertex1.index) - np.array(vertex2.index)) > 0)[0]
    if vertex1.vertexDir[connectionDirIndex] == 0:
        print('No EPair by connection dir')
        return False

    # Check that the two don't just form a line
    if vertex1.vertexDim < 3 or vertex2.vertexDim < 3:
        print('No EPair by G(1,1,2) subgraph')
        return False
    
    print('EPair')
    return True


def _createVPair(vertex: Vertex):
    """
    """
    if vertex.vertexDim == 0:
        return []

    vPairList = [(vertex.index, neighbor.index) for neighbor in vertex.neighbors if _isEPair(vertex, neighbor)]

    return vPairList

class Octree():

    def __init__(self, points, nBoxes=500, neighborThreshold=1/32, debug=False, minPointsPerBox=2):
        """
        """
        print('Octree generation begun...')
        
        self.points = points
        self.neighborThreshold = neighborThreshold
        self.debug = debug

        ################################
        # Partition Points Into Boxes
        ################################
        boxSize, boxMembership, boxCorners = partitionIntoBoxes(self.points, nBoxes, returnIndices=True) 
        self.boxSize = boxSize

        numFilledBoxes = int(np.max(boxMembership)) + 1

        # Convert box corners to indices
        # Subtract off top left corner
        boxIndices = [tuple(t) for t in np.floor((boxCorners - np.min(boxCorners, axis=0))/boxSize).astype(np.int64)]

        # Store the boxes in a dictionary
        self.boxes = {}

        # Create the boxes
        for i in range(numFilledBoxes):
            if len(points[boxMembership == i]) >= minPointsPerBox:
                self.boxes[boxIndices[i]] = Box(boxCorners[i], boxSize, points[boxMembership == i], boxName=boxIndices[i])

        
        # Determine neighbors
        dirVecArr = np.array([(1,0,0), (-1,0,0),
                              (0,1,0), (0,-1,0),
                              (0,0,1), (0,0,-1)])
   
        # If a box has a certain neighbor, we add the value
        # of that direction vector in the above array to the
        # list of neighbors in boxNeighbors

        # Only have to loop over the boxes that actually do have
        # points in them
        for k,b in self.boxes.items():
            # Look at each neighbor
            for j in range(len(dirVecArr)):
              
                neighborIndex = tuple(np.array(k, dtype=np.int64) + dirVecArr[j])

                # Check if the neighbor is in the dictionary
                if not neighborIndex in self.boxes:
                    continue

                # No need to check if the box is already a neighbor
                if neighborIndex in b.neighbors:
                    continue

                isNeighbor = _medianDistanceConnectionCriteria(b.points, self.boxes[neighborIndex].points, neighborThreshold)

                if isNeighbor:
                    b.addNeighbors(self.boxes[neighborIndex])
                    self.boxes[neighborIndex].addNeighbors(b)

        # Vertex dimension is calculated automatically as points are added
        # so no need to do that explicitly
        
        # Now we are going to switch to Vertex objects, since the Box objects are
        # a little unwieldy
        self.vertices = [Vertex(b.getBoxCentroid(), b.boxName, [n.boxName for n in b.neighbors]) for k,b in self.boxes.items()]
        # Back to a dictionary
        self.vertices = dict(zip([v.index for v in self.vertices], self.vertices))
        
        return
        # Now, we do the reduction process for E pairs and V pairs
        # as described in the paper
        # We start with vertices with high vertex dimension and work
        # our way down to 2
        vPairList = []
        # So we can keep pointers to the merged versions of vertices
        indexConversion = dict(zip([v for v in self.vertices], [v for v in self.vertices]))

        for dim in range(5,1,-1):
            # Fetch every vertex with at least the given dimension
            indexList = [v.index for k,v in self.vertices.items() if v.vertexDim >= dim]

            # Find VPairs within the list of vertices
            for index in indexList:
                for neighborIndex in [ind for ind in self.vertices[indexConversion[index]].neighbors]:
                    if _isVPair(self.vertices[indexConversion[index]], self.vertices[indexConversion[neighborIndex]]):
                        vPairList.append((indexConversion[index], indexConversion[neighborIndex]))
                        # Only want one from each vertex maximum, so break
                        # afterwards
                        break
            
            foundEPair = True
            while len(vPairList) > 0 or foundEPair:

                # Merge all vPairs
                for index, neighborIndex in vPairList:
                    v1 = self.vertices[indexConversion[index]]
                    v2 = self.vertices[indexConversion[neighborIndex]]

                    # Merge
                    newPosition = (v1.weight * v1.position + v2.weight * v2.position) / (v1.weight + v2.weight)
                    # Just take the first index (doesn't really matter)
                    newIndex = indexConversion[index]
                    newNeighbors = v1.neighbors + v2.neighbors
                    # Remove duplicates, and the new vertex itself
                    newNeighbors = [indexConversion[t] for t in np.unique(newNeighbors, axis=0) if indexConversion[t] != newIndex]  
                    print(newNeighbors)
                    newWeight = v1.weight + v2.weight

                    self.vertices[newIndex] = Vertex(newPosition, newIndex, newNeighbors, newWeight)
                    # Point the old vertex to the merged one
                    indexConversion[indexConversion[neighborIndex]] = newIndex

                # Delete all of the merged vPairs
                vPairList = []

                # Look for vPairs again (same code as above)
                # Fetch every vertex with at least the given dimension
                indexList = [v.index for k,v in self.vertices.items() if v.vertexDim >= dim]

                for index in indexList:
                    for neighborIndex in [ind for ind in self.vertices[indexConversion[index]].neighbors]:
                        if _isVPair(self.vertices[indexConversion[index]], self.vertices[indexConversion[neighborIndex]]):
                            vPairList.append((indexConversion[index], indexConversion[neighborIndex]))
                            # Only want one from each vertex maximum, so break
                            # afterwards
                            break

                print(vPairList) 
                if len(vPairList) == 0:
                    foundEPair = False
                    # Try to find a single ePair, which we can then merge into a vpair
                    for index in indexList:
                        if foundEPair:
                            break

                        for neighborIndex in [ind for ind in self.vertices[index].neighbors]:
                            if _isEPair(self.vertices[index], self.vertices[neighborIndex]):

                                ePair = (index, neighborIndex)
                                foundEPair = True
                                break

                    if foundEPair:
                        # Merge
                        self.vertices[indexConversion[ePair[0]]] = self.vertices[indexConversion[ePair[0]]].merge(self.vertices[indexConversion[ePair[1]]])
                        indexConversion[indexConversion[ePair[1]]] = indexConversion[ePair[0]]


    def vertexPoints(self):
        """
        Generate an array of points comprising the vertices of
        the octree, and the adjacency matrix describing connectivity.
        """
        #points = np.array([b.getBoxCentroid() for k,b in self.boxes.items()])

        #indices = list(self.boxes.keys())

        #adjMat = np.zeros((len(points), len(points)))
        #kdTree = KDTree(points)
        #for i in range(len(indices)):
            #neighbors = kdTree.query_ball_point(points[i], 1.5*np.mean(self.boxSize))
            #adjMat[i][neighbors] = 1
            #adjMat[i] = np.array([ind in [b.name for b in self.boxes[indices[i]].neighbors] for ind in indices], dtype=np.int64)

        points = np.array([v.position for k,v in self.vertices.items()])

        indices = list(self.vertices.keys())
        indexToLinear = dict(zip(indices, np.arange(len(points))))

        adjMat = np.zeros((len(points), len(points)))
        for i in range(len(indices)):
            adjMat[i][np.array([indexToLinear[ind] for ind in self.vertices[indices[i]].neighbors], dtype=np.int64)] = 1

        return points, adjMat


    def plot(self, ax=None):
        """
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        for k,b in self.boxes.items():
            b.plot(ax=ax, c=str(colour.Color(pick_for=b)), alpha=.4)

        return plt.gcf()


    def plotVertices(self, ax=None, plotBoxes=True, plotEdges=True):
        """
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        # Draw boxes
        if plotBoxes:
            for k,b in self.boxes.items():
                b.plot(ax=ax, drawPoints=False, c=str(colour.Color(pick_for=b)), alpha=.1)
       
        points, adjMat = self.vertexPoints()

        # Draw centroids
        ax.scatter(*points.T, s=3)
       
        # Draw lines
        if plotEdges:
            for i in range(len(adjMat)):
                edgeIndices = np.where(adjMat[i] > 0)[0]
                for j in range(len(edgeIndices)):
                    ax.plot(*list(zip(points[i], points[edgeIndices[j]])))

        return plt.gcf()

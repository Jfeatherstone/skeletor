r"""
This performs laplacian contraction but breaks up the point cloud
using a basic octree structure to decrease computation time.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

import os
import time, sys
import itertools
import colour

import open3d as o3d

from anytree import Node

# DEBUG
from tqdm import tqdm

import robust_laplacian
import scipy.sparse.linalg as sla
from scipy import sparse

from skeletor.utils import detectConnectedStructures
from skeletor.utils import getBoxLines, plotBox

from skeletor.spatial import pathIntegralAlongField, partitionIntoBoxes
from skeletor.spatial import angularHistogramAroundPoint, findDominantHistogramDirections, discretizeDirectionVectors, courseGrainField

from .skeleton_base import SkeletonBase

class Box(Node):

    def __init__(self, boxCorner, boxSize, points=None, neighbors=None, parent=None, boxName=None):
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
            # TODO: Find a better way to adaptively choose these parameters
            hist, axes = angularHistogramAroundPoint(self.points, np.mean(self.points, axis=0))
            peakDirections = np.array(findDominantHistogramDirections(hist, axes, peakFindPrevalence=0.5))

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
            plt.gca().scatter(*self.points.T, label=self.boxName, alpha=.05, **kwargs)

        if drawCentroid:
            plt.gca().scatter(*self.getBoxCentroid().T, c='tab:red', s=20)

        if drawMoments and len(self.moments) > 0:
            momentScale = (np.max(self.points, axis=0) - np.min(self.points, axis=0))
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



class OctreeContractionSkeleton(SkeletonBase):

    def __init__(self, points, nBoxes=1000, minPointsPerBox=1, debug=False, verbose=False):
        """
        """ 
        # Super class will take care of the basic stuff
        super().__init__(points, verbose, debug)

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

        # Get rid of empty boxes (or boxes with too few points)
        self.boxes = [b for b in self.boxes if b.getNumPoints() > minPointsPerBox]
        
        if self.verbose:
            print(f'Partioned space into {nBoxes} boxes, of which {numFilledBoxes} contain points.')

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
        asymmetricDistanceThreshold = 5

        # Factor for whether two boxes are neighbors based simply
        # on having centroids that are very close to each other.
        # **Relative to the average nearest neighbor distance**
        mergeDistanceThreshold = 5

        # Threshold for determining neighbors via point density
        # along an edge cut. For exact details, see Condition 4
        # calculation below, but should be in the region [0,1]
        # with a value of 1 being the most strict, and 0 being
        # perfectly permissive.
        pointDensityThreshold = 0.6

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

        distanceThreshold = np.sqrt(np.sum(self.boxes[0].getBoxSize()**2)) * initialDistanceThreshold

        # Find all pairs of faces that are within the distance threshold to each other
        kdTree = KDTree(allBoxFaces)
        # This is a list of lists, [[i,j], [k,l,m], ...]
        # where i and j are indices of the faces that are neighbors to the 0th face,
        # k, l and m are neighbors to the 1st face, etc.
        potentialNeighbors = kdTree.query_ball_point(allBoxFaces, distanceThreshold)

        # Convert from indexing the faces to indexing the boxes
        #potentialNeighbors = [[np.int64(np.floor(ind/(self.dim*2))) for n in potentialNeighbors[faceIdentities == i] for ind in n] for i in range(len(self.boxes))]
        potentialNeighbors = np.array([[faceIdentities[ind] for ind in potentialNeighbors[i]] for i in range(len(faceIdentities))], dtype='object')

        # And combine all 2*d faces into the same box
        potentialNeighbors = [np.concatenate(potentialNeighbors[np.where(faceIdentities == i)]) for i in range(len(self.boxes))]

        # Remove duplicates, in case, eg. the face of one box is a neighbor
        # to two faces of another box.
        potentialNeighbors = [np.unique(n) for n in potentialNeighbors]

        # In a more dedicated octree implementation, we would then do a
        # more in depth check to see if indeed these neighbors are
        # actually neighbors, but we can ignore that here.
        for i in range(len(self.boxes)):
            self.boxes[i].addNeighbors([self.boxes[n] for n in potentialNeighbors[i]])
        
        # Histogram of the number of neighbors
        if self.debug:
            # -1 to account for self counting
            plt.hist([len(p)-1 for p in potentialNeighbors], bins=np.arange(2*self.dim+1) - .5)
            plt.xlabel('Potential Neighbors')
            plt.title('Histogram of Potential Neighbors')
            plt.show()


    def contractPointCloud(self, iterations=5, attraction=10, contraction=1):
        """
        """
        for k in range(iterations):
            for i in tqdm(range(len(self.boxes))):
                # Find points from all nearby boxes
                nearbyPoints = np.concatenate([n.points for n in self.boxes[i].neighbors])

                contractedBoxPoints = contractPoints(self.boxes[i].points, nearbyPoints,
                                                     attraction=attraction, contraction=contraction)

                self.boxes[i].points = contractedBoxPoints

        return np.concatenate([b.points for b in self.boxes])


    def generateSkeleton(self):
        """
        Generate the skeleton representation.
        """
        ##########################################
        #       Compute the rough skeleton
        ##########################################
        self.skeletonPoints = np.array([b.getBoxCentroid() for b in self.boxes])

        self.skeletonAdjMat = np.zeros((len(self.skeletonPoints), len(self.skeletonPoints)))

        for i in range(len(self.skeletonPoints)):
            self.skeletonAdjMat[i] = np.array([self.boxes[ind] in self.boxes[i].neighbors for ind in range(len(self.skeletonPoints))], dtype=np.int64)

        if self.verbose:
            print(f'Found skeleton with {len(self.skeletonPoints)} points.')
       
        ##########################################
        #       Merge vertices by proximity
        ##########################################

        # Factor for whether two boxes are neighbors based simply
        # on having centroids that are very close to each other.
        # **Relative to the average nearest neighbor distance**
        mergeDistanceThreshold = 5

        # First, we look at if there are vertices
        # that are very close to each other that we can merge
        skeletonKDTree = KDTree(self.skeletonPoints)
        
        nearestNeighborDistances, nearestNeighborVertices = skeletonKDTree.query(self.skeletonPoints, 2)

        # Remove 0th element because it will be self
        vertexDistances = np.array([n[1] for n in nearestNeighborDistances])
        # Create pairs
        vertexPairs = np.array([(i,nearestNeighborVertices[i][1]) for i in range(len(nearestNeighborVertices))], dtype=np.int64)

        # We want to merge vertices that are closer than the average nearest neighbor distance
        # for the *original* point cloud.
        mergePairIndices = np.where(vertexDistances <= self.avgNNDistance*mergeDistanceThreshold)
        mergeDistances = vertexDistances[mergePairIndices]

        mergePairs = vertexPairs[mergePairIndices][np.argsort(mergeDistances)]

        merged = np.zeros(len(self.skeletonPoints), dtype=bool)
        redundantIndices = np.zeros(len(self.skeletonPoints), dtype=bool)

        for i in range(len(mergePairs)):
            # If we've already merged one of the two points, we move on
            if merged[mergePairs[i][0]] or merged[mergePairs[i][1]]:
                continue

            # Otherwise, we merge the two
            self.skeletonPoints[mergePairs[i][0]] = (self.skeletonPoints[mergePairs[i][0]] + self.skeletonPoints[mergePairs[i][1]]) / 2

            self.skeletonAdjMat[mergePairs[i][0]] += self.skeletonAdjMat[mergePairs[i][1]]
            self.skeletonAdjMat[self.skeletonAdjMat > 0] = 1

            # Remove self adjacency
            self.skeletonAdjMat[mergePairs[i][0],mergePairs[i][0]] = 0

            # Mark the second point as redundant
            redundantIndices[mergePairs[i][1]] = True

            # And mark both as merged
            merged[mergePairs[i][0]] = True
            merged[mergePairs[i][1]] = True
        
        allMergeIndices = np.where(redundantIndices)[0]

        # Indices that weren't remove in merging
        keepIndices = [i for i in range(len(self.skeletonPoints)) if i not in allMergeIndices]

        # Remove the indices
        self.skeletonPoints = self.skeletonPoints[keepIndices]
        # TODO: Switch to sparse representation
        self.skeletonAdjMat = self.skeletonAdjMat[keepIndices][:,keepIndices]

        if self.verbose:
            print(f'Removed {len(merged) - len(keepIndices)} points from skeleton by merging.')

        return self.skeletonPoints, self.skeletonAdjMat

        # TODO: everything below this is still WIP

        ##########################################
        #       Remove redundant vertices
        ##########################################

        pointIdentities = np.arange(len(self.skeletonPoints))

        for i in range(len(self.skeletonPoints)):
            neighbors = np.where(self.skeletonAdjMat[pointIdentities[i]])[0]
            for j in neighbors:
                # Find the neighborhood of all nearby points
                allNeighborIndices = np.concatenate((np.where(self.skeletonAdjMat[pointIdentities[i]] > 0)[0], np.where(self.skeletonAdjMat[pointIdentities[j]] > 0)[0]))
                allNeighborIndices = np.unique(allNeighborIndices)

                nearbyPoints = np.array([p for k in allNeighborIndices for p in self.boxes[pointIdentities[k]].points])
                
                # Now we want to compute the average points (per unit length)
                # along each edge that are incident to either vertex,
                # and then compare that value to the same value but if
                # we were to merge the two vertices.

                linesPointI = [(i, k) for k in np.where(self.skeletonAdjMat[pointIdentities[i]] > 0)[0]]
                linesPointJ = [(j, k) for k in np.where(self.skeletonAdjMat[pointIdentities[j]] > 0)[0]]

                allLines = linesPointJ + linesPointI
                
                # Remove the connection between i and j (since we will be
                # removing that one in a second so we cant really compare
                # it to anything afterwards).
                # TODO

                averagePointsPerLengthOld = 0
                for k,l in allLines:
                    point1 = self.skeletonPoints[pointIdentities[k]]
                    point2 = self.skeletonPoints[pointIdentities[l]]

                    # This is pretty much the same calculation as for the
                    # 'Condition 4' neighbor criterion above, so see there
                    # for more info.
                    separation = point1 - point2
                    separationNorm =  np.sqrt(np.sum(separation**2))

                    if separationNorm == 0:
                        continue

                    separationUnit = separation / separationNorm

                    tArr = np.dot(separationUnit, (nearbyPoints - point2).T)

                    # Find the parameter value for the other centroid
                    # ie. points with a parameter value larger than
                    # this value are no longer between the two centroids,
                    # so we don't care about them.
                    # The minimum is simply 0, so we care about the
                    # range [0, maxTValue]
                    # Since we use the normalized separation vector above, this is
                    # just the length of the line segment.
                    maxTValue = separationNorm

                    # Remove values outside of that range
                    keepIndices = np.where((tArr >= 0) & (tArr <= maxTValue))[0]
                    tArr = tArr[keepIndices]
                    
                    if len(tArr) == 0:
                        continue

                    # Now we can calulate the distance from each of those points
                    # to the line
                    pointsProjectedOnLine = point2 + np.multiply.outer(tArr, separationUnit)
                    pointDistanceToLine = np.sqrt(np.sum((nearbyPoints[keepIndices] - pointsProjectedOnLine)**2, axis=-1))

                    # We don't want a step size lower than our nearest neighbor distance
                    pathPoints = int(np.ceil(maxTValue / self.avgNNDistance))
                    # Just for more clear code below
                    pathStepSize = self.avgNNDistance

                    # We weight our point density by the perpendicular distance of each
                    # point to the line, since nearby points should count more.
                    weights = pointDistanceToLine / (np.max(pointDistanceToLine) + 1e-4)
                    weights = 1 / (1 + weights)
                    weights /= np.max(weights)

                    #print(separation)
                    #print(pathPoints)
                    pointDensityAlongLine, binEdges = np.histogram(tArr, pathPoints, weights=weights)

                    averagePointsPerLengthOld += np.sum(pointDensityAlongLine) / maxTValue / len(allLines)

                # Now to compare, we see what would happen if we were to merge
                # the two current points
                averagePointsPerLengthNew = 0
                for k in allNeighborIndices:
                    point1 = (self.skeletonPoints[pointIdentities[i]] + self.skeletonPoints[pointIdentities[j]]) / 2
                    point2  = self.skeletonPoints[pointIdentities[k]]

                    # Same as above
                    separation = point1 - point2
                    separationUnit = separation / np.sqrt(np.sum(separation**2))

                    tArr = np.dot(separationUnit, (nearbyPoints - point2).T)
                    maxTValue = np.sqrt(np.sum(separation**2))

                    # Remove values outside of that range
                    keepIndices = np.where((tArr >= 0) & (tArr <= maxTValue))[0]
                    tArr = tArr[keepIndices]
                    
                    if len(tArr) == 0:
                        continue

                    # Now we can calulate the distance from each of those points
                    # to the line
                    pointsProjectedOnLine = point2 + np.multiply.outer(tArr, separationUnit)
                    pointDistanceToLine = np.sqrt(np.sum((nearbyPoints[keepIndices] - pointsProjectedOnLine)**2, axis=-1))

                    # We don't want a step size lower than our nearest neighbor distance
                    pathPoints = int(np.ceil(maxTValue / self.avgNNDistance))
                    #print(separationNorm, self.avgNNDistance)

                    # We weight our point density by the perpendicular distance of each
                    # point to the line, since nearby points should count more.
                    weights = pointDistanceToLine / (np.max(pointDistanceToLine) + 1e-4)
                    weights = 1 / (1 + weights)
                    weights /= np.max(weights)

                    pointDensityAlongLine, binEdges = np.histogram(tArr, pathPoints, weights=weights)

                    averagePointsPerLengthNew += np.sum(pointDensityAlongLine) / maxTValue / len(allNeighborIndices)

                print(averagePointsPerLengthOld, averagePointsPerLengthNew)

                # If the new merged configuration is about as good
                # as the old one, we merge
                if averagePointsPerLengthNew >= averagePointsPerLengthOld*0.95:
                    
                    if self.verbose:
                        print(f'Merged points {pointIdentities[i]} and {pointIdentities[j]} (working indices {i} and {j})')
                        plt.plot(pointIdentities)

                    # Move the merged point
                    self.skeletonPoints[pointIdentities[i]] = (self.skeletonPoints[pointIdentities[i]] + self.skeletonPoints[pointIdentities[j]]) / 2

                    # Add all neighbors to point i from j
                    self.boxes[pointIdentities[i]].addNeighbors(self.boxes[pointIdentities[j]].neighbors)

                    # Update the adjacency matrix
                    self.skeletonAdjMat[pointIdentities[i]] += self.skeletonAdjMat[pointIdentities[j]]
                    self.skeletonAdjMat[self.skeletonAdjMat > 0] = 1
                    # Remove self adjacency
                    self.skeletonAdjMat[pointIdentities[i],pointIdentities[i]] = 0

                    # Replace all mentions of j with i
                    pointIdentities[np.where(pointIdentities == pointIdentities[j])] = pointIdentities[i]

                    if self.verbose:
                        plt.plot(pointIdentities)
                        plt.show()

        keepPoints = np.unique(pointIdentities)
        self.skeletonPoints = self.skeletonPoints[keepPoints]

        self.skeletonAdjMat = self.skeletonAdjMat[keepPoints]
        self.skeletonAdjMat = self.skeletonAdjMat[:,keepPoints]

        # We first want to test pairs of vertices to see
        # if we were to merge them, would it affect how
        # well the edges follow the density field of the
        # point cloud. If there is little to no change (or
        # even an increase) then that means we can safely
        # merge the two vertices.

        # First, find all pairs of vertices that are close by
        # We can define "close by" using the average length
        # of a line.
        skeletonEdges = np.array(np.where(self.skeletonAdjMat > 0), dtype=np.int64).T
        # Remove duplicate edges (eg. [0, 1] and [1, 0])
        skeletonEdges = [np.sort(e) for e in skeletonEdges]
        skeletonEdges = np.unique(skeletonEdges, axis=0)

        averageEdgeLength = np.mean([np.sqrt(np.sum((self.skeletonPoints[i] - self.skeletonPoints[j])**2)) for i,j in skeletonEdges])

        # Use a kd tree
        skeletonPointTree = KDTree(self.skeletonPoints)
        # Use half the edge length, because that seems reasonable
        pointNeighborIndices = skeletonPointTree.query_ball_tree(skeletonPointTree, averageEdgeLength/2)

        # We can easily remove points since that will mess up
        # our indexing, so when we merge we just identify
        # points together, and then cull duplicates at the
        # end
        pointIdentities = np.arange(len(self.skeletonPoints))

        # Start with points with the most neighbors
        pointOrder = np.argsort([np.sum(self.skeletonAdjMat[i]) for i in range(len(self.skeletonAdjMat))])
        for i in pointOrder:
            for j in pointNeighborIndices[pointIdentities[i]]:
                print(i,j)

                # Since we will be doing line cuts through the density field (effectively)
                # we need a set of points to compute that density field.
                # The most general way is to include all boxes that are a neighbor with
                # either box j or box i. Might be slightly overskill, but...
                allNeighborIndices = np.concatenate((np.where(self.skeletonAdjMat[pointIdentities[i]] > 0)[0], np.where(self.skeletonAdjMat[pointIdentities[j]] > 0)[0]))
                allNeighborIndices = np.unique(allNeighborIndices)
                print(allNeighborIndices)

                nearbyPoints = np.array([p for k in allNeighborIndices for p in self.boxes[pointIdentities[k]].points])
                print(nearbyPoints.shape)

                break
                # Sum over all the edges for these two points
                averagePointsPerLength = 0
                    
                # Iterate over edges for point i
                for k in np.where(self.skeletonAdjMat[pointIdentities[i]] > 0)[0]:
                    pass

                # Iterate over edges for point j
            break

        ##########################################
        #       Find neighborhood of each line
        ##########################################
        # First, we locate which lines could be related
        # to each other; we have far fewer lines than we
        # had points originally, but we still don't
        # want to do an n^2 calculation for the lines.
        skeletonEdges = np.array(np.where(self.skeletonAdjMat > 0), dtype=np.int64).T
        # Remove duplicate edges (eg. [0, 1] and [1, 0])
        skeletonEdges = [np.sort(e) for e in skeletonEdges]
        skeletonEdges = np.unique(skeletonEdges, axis=0)

        # As a rough estimate, we just say that any line that has
        # a point within a certain distance of either endpoint or the
        # mid point of the current line could be considered a neighbor.
        # TODO: This could miss if two lines intersect... but I think that
        # would be a rare case
        
        # This is similar to what we did for box faces above
        # This array looks like: [0,0,0,1,1,1,2,2,2,...]
        edgeIdentities = np.array([i for i in range(len(skeletonEdges)) for j in range(3)])
        allEdgePoints = np.array([[self.skeletonPoints[i],
                                   self.skeletonPoints[j],
                                   (self.skeletonPoints[i] + self.skeletonPoints[j])/2] for i,j in skeletonEdges])

        allEdgePoints = allEdgePoints.reshape((3*len(allEdgePoints), self.dim))

        # Now we use a KD Tree to find which lines could be interacting
        # with which other lines
        edgePointsTree = KDTree(allEdgePoints)

        # Since we have uniform boxes technically we could just
        # use a static threshold for how far away another point could
        # be, but I want this to be adaptable for a non-uniform box
        # case, so we do it box by box
        nearbyEdges = []
        for i in range(len(skeletonEdges)):

            # Length of the line / 2
            halfEdgeLength = np.sqrt(np.sum((self.boxes[skeletonEdges[i][0]].getBoxCentroid() - self.boxes[skeletonEdges[i][1]].getBoxCentroid())**2)) / 2

            # The two endpoints and midpoint of this line
            currentEdgePoints = allEdgePoints[[3*i, 3*i+1, 3*i+2]]

            # Find nearby points
            nearbyPointIndices = edgePointsTree.query_ball_point(currentEdgePoints, halfEdgeLength)

            # We don't care which of the endpoints/midpoint the neighbors are
            # close to, so we just flatten the indices (but we can't just use
            # flatten() because the index lists could be triangular)
            nearbyPointIndices = [j for k in nearbyPointIndices for j in k] 

            # Now find the identities of the lines that these points belong to
            nearbyLineIndices = np.unique(edgeIdentities[nearbyPointIndices])

            nearbyEdges.append(nearbyLineIndices)

        if self.verbose:
            print(f'Found preliminary neighborhoods of edges.')

        ##########################################
        #       Begin merging redundant vertices
        ##########################################

        # We will have to keep track if we merge points, and
        # we can't directly remove them from an array because
        # then the indexing will be all messed up.
        pointIdentities = np.arange(len(self.skeletonPoints))
        
        # Start with points with the fewest nearby edges
        # Not really sure why, but seems like a good idea...
        edgeOrder = np.argsort([len(n) for n in nearbyEdges])

        for i in edgeOrder:
            nearbyEdgeIndices = nearbyEdges[i]

            # If two lines are mostly parallel 
            # TODO



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


def contractPoints(points, referencePoints=None, pointMasses=None, attraction=1.0, contraction=0.5):
    """
    Perform Laplacian contraction on a set of points, potentially in relation to
    a reference set of points (points that should attract other points but not
    move themselves).
    """
    dim = np.shape(points)[-1]

    if hasattr(referencePoints, '__iter__'):
        allPoints = np.concatenate((points, referencePoints))

        # We need a certain amount of points for the process to actually
        # do anything; 5 is arbitrary
        if len(allPoints < 5):
            return points

        # robust_laplacian's point cloud laplacian calculation uses the
        # nearby nearby neighbors, and requires that you have at least
        # n_neighbors points, which is equal to 30 by default.

        # Compute the laplacian and mass matrix
        L, M = robust_laplacian.point_cloud_laplacian(allPoints, n_neighbors=min(30, len(allPoints)-1))

        # We should only have positive attraction towards reference
        # points, and we should have slight negative attraction towards
        # regular points (to avoid clumping)
        if hasattr(pointMasses, '__iter__'):
            pointRepulsion = attraction * pointMasses
        else:
            pointRepulsion = attraction * np.ones(len(points))

        # Multiply the attraction of the reference points by a very large number so
        # they don't move from their original positions much
        referencePointAttraction = attraction * np.ones(len(referencePoints))*1e6
        pointContraction = contraction * 1e3 * np.sqrt(np.mean(M.diagonal())) * np.ones(len(points))
        referencePointContraction = contraction * 1e3 * np.sqrt(np.mean(M.diagonal())) * np.ones(len(referencePoints))

        # Define weight matrices
        WH = sparse.diags(np.concatenate((pointRepulsion, referencePointAttraction)))
        WL = sparse.diags(np.concatenate((pointContraction, referencePointContraction)))  # I * laplacian_weighting

    else:
        allPoints = points
        
        # We need a certain amount of points for the process to actually
        # do anything; 5 is arbitrary
        if len(allPoints < 5):
            return points

        # Compute the laplacian and mass matrix
        L, M = robust_laplacian.point_cloud_laplacian(allPoints, n_neighbors=min(30, len(points)-1))

        attractionWeights = attraction * np.ones(M.shape[0])
        # This is weighted by the sqrt of the mean of the mass matrix, not really sure why, but :/
        contractionWeights = contraction * 1e3 * np.sqrt(np.mean(M.diagonal())) * np.ones(M.shape[0])

        # Define weight matrices
        WH = sparse.diags(attractionWeights)
        WL = sparse.diags(contractionWeights)  # I * laplacian_weighting

    A = sparse.vstack([L.dot(WL), WH]).tocsc()
    b = np.vstack([np.zeros((allPoints.shape[0], 3)), WH.dot(allPoints)])

    A_new = A.T @ A

    # Solve each dimension separately
    solvedAxes = [sla.spsolve(A_new, A.T @ b[:,i], permc_spec='COLAMD') for i in range(dim)]
    # If we are in 2D, just add back in the previous z dimension (no need to solve it since
    # we will throw it away eventually)
    if dim == 2:
        solvedAxes += [list(points[:,2])]
    ret = np.vstack(solvedAxes).T

    if (np.isnan(ret)).all():
        #logging.warn('Matrix is exactly singular. Stopping Contraction.')
        ret = points

    return ret[:len(points)]

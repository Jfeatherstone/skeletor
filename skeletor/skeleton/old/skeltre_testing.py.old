import numpy as np
from skeletor.utils import partitionIntoBoxes, angularHistogramAroundPoint, findDominantHistogramDirections
from skeletor.utils import detectConnectedStructures

from scipy.spatial import KDTree

import matplotlib.pyplot as plt

def connectionCriteria(boxPoints, neighborPoints, threshold):
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
    d1 = calculateSquaredMedianDistance(boxPoints, boxCentroid, planeNormal)
    d2 = calculateSquaredMedianDistance(neighborPoints, neighborCentroid, planeNormal)
    d12 = calculateSquaredMedianDistance(np.concatenate((boxPoints, neighborPoints), axis=0), combinedCentroid, planeNormal)
    
    return d12*threshold <= min(d1, d2)
    

    
def calculateSquaredMedianDistance(points, planePoint, planeNormal):
    """
    Compute the squared median distance for points to a plane.
    """

    d = np.dot(planeNormal, planePoint)
    planeDistance = (np.dot(planeNormal, np.transpose(points)) - d).T**2

    return np.median(planeDistance)


class Octree(object):

    def __init__(self, points, nBoxes=5000, neighborThreshold=1/16, neighborMethod='arbitrary', debug=False):
        """

        """

        self.nBoxes = nBoxes
        self.points = points
 
        ################################
        # Partition Points Into Boxes
        ################################
        self.boxSize, self.boxMembership, self.boxIndices = partitionIntoBoxes(self.points, self.nBoxes, returnIndices=True) 
      
        self.numFilledBoxes = int(np.max(self.boxMembership)) + 1

        # Calculate how large the space is, so we can determine the number
        # of boxes that spans the whole volume (but most might be empty)
        self.occupiedVolumeBounds = np.array(list(zip(np.min(points, axis=0), np.max(points, axis=0))))

        volumeSize = self.occupiedVolumeBounds[:,1] - self.occupiedVolumeBounds[:,0]
        volumeSizeInBoxes = np.ceil(volumeSize / self.boxSize).astype(np.int64)
        
        if debug:
            print(f'Partitioned space into {volumeSizeInBoxes[0]}x{volumeSizeInBoxes[1]}x{volumeSizeInBoxes[2]} cells, of which {self.numFilledBoxes} contain points.')

        ################################
        # Generate Dictionaries
        ################################
        # Generate a 3D array of lists for the points in each box
        self.boxPoints = {}
        self.boxNeighbors = {}
        self.dominantDirections = {}
        self.boxContainsPoints = np.zeros(volumeSizeInBoxes, dtype=bool)

        for i in range(volumeSizeInBoxes[0]):
            for j in range(volumeSizeInBoxes[1]):
                for k in range(volumeSizeInBoxes[2]):
                    # Empty list
                    self.boxPoints[(i,j,k)] = []
                    # Assign neigbors dict now as well bc why not
                    self.boxNeighbors[(i,j,k)] = []
                    # Assign directions as well
                    self.dominantDirections[(i,j,k)] = []

        # Now add the boxes with actual points
        for i in range(len(self.boxIndices)):
            self.boxPoints[tuple(self.boxIndices[i])] = points[self.boxMembership == i]

        # It is helpful to have an array of whether each box
        # has any points in it
        for i in range(volumeSizeInBoxes[0]):
            for j in range(volumeSizeInBoxes[1]):
                for k in range(volumeSizeInBoxes[2]):
                    self.boxContainsPoints[(i,j,k)] = len(self.boxPoints[(i,j,k)]) > 0

        # boxPoints now contains the list of points for each key (i,j,k)
        # (most of them are empty lists [])
     
        ################################
        # Determine Dominant Directions
        ################################
        # We want to determine the orientation of
        # points in each box, but in general there could
        # be more than one dominant direction (eg. a
        # 3-point vertex)

        for i in range(len(self.boxIndices)):
            index = tuple(self.boxIndices[i])
            # Compute the centroid
            centroid = np.median(self.boxPoints[index], axis=0)
            
            # Calculate dominant directions (see other files
            # for more information on this)
            pointsWithCentroid = np.concatenate((self.boxPoints[index], [centroid]), axis=0)
            hist, thetaBins, phiBins = angularHistogramAroundPoint(pointsWithCentroid, -1, smoothing=21)
            peakDirections = findDominantHistogramDirections(hist, thetaBins, phiBins)

            self.dominantDirections[index] = peakDirections

        #################################################
        # Determine Connections via Dominant Directions
        #################################################
        # Extension length (kind of arbitrarily chosen)
        extensionLength = np.max(self.boxSize) * 1
        angleThreshold = 1
        
        if neighborMethod.lower() == 'arbitrary':
            for i in range(len(self.boxIndices)):
                selfIndex = tuple(self.boxIndices[i])

                # Look at each dominant direction
                for j in range(len(self.dominantDirections[selfIndex])):
                    centroid = np.median(self.boxPoints[selfIndex], axis=0)

                    # Compute a bunch of points along the extended line
                    # We only need them in interval on a length scale 
                    # a bit smaller than the box size

                    # This generates points along the line at intervals of 2*extensionLength/np.min(self.boxSize)
                    linePoints = centroid + np.array([t*self.dominantDirections[selfIndex][j] for t in np.linspace(0, extensionLength, int(2*extensionLength/np.min(self.boxSize)))])
                    intersectingBoxes = [tuple(ind) for ind in np.floor((linePoints - self.occupiedVolumeBounds[:,0]) / self.boxSize).astype(np.int64)]
                    # Have to convert back to tuples, since unique turns things into arrays
                    intersectingBoxes = [tuple(ind) for ind in np.unique(intersectingBoxes, axis=0)]

                    # Remove the current box index
                    intersectingBoxes = [ind for ind in intersectingBoxes if ind != selfIndex]

                    # We can disregard any boxes that are out of bounds
                    intersectingBoxes = [ind for ind in intersectingBoxes if (np.array(ind) >= 0).all()]
                    intersectingBoxes = [ind for ind in intersectingBoxes if not True in [ind[k] >= volumeSizeInBoxes[k] for k in range(3)]]
                    
                    # If this direction doesn't intersect any other cells, we move
                    # to the next one
                    if len(intersectingBoxes) == 0:
                        continue

                    # Now order the boxes based on their distance from the current
                    # one
                    indexDistances = np.sum((np.array(selfIndex) - np.array(intersectingBoxes))**2, axis=-1)
                    intersectingBoxes = [intersectingBoxes[k] for k in np.argsort(indexDistances)]

                    #print(selfIndex, intersectingBoxes)
                    # Now check through each box to see if it is properly a neighbor
                    foundNeighbor = False
                    for k in range(len(intersectingBoxes)):
                        # Compute the separation vector between the current cell and the
                        # potential neighbor cell (sign doesn't matter)
                        separation = centroid - np.median(self.boxPoints[intersectingBoxes[k]])

                        # Loop over the dominant directions of the potential neighbor
                        # (I know there's a lot of for loops here)
                        for l in range(len(self.dominantDirections[intersectingBoxes[k]])):
                            # Project the dominant direction onto the separation vector
                            projection = np.dot(self.dominantDirections[intersectingBoxes[k]][l], self.dominantDirections[selfIndex][j])

                            # Dominant directions are already unit vectors, so no need to divide
                            # out the magnitudes
                            angle = np.arccos(projection)
                            
                            # If less than the threshold, then this is a proper neighbor
                            if abs(angle) <= angleThreshold:
                                # Probably needs to change later
                                # TODO since convention is different than in other
                                # neighbor checking algorithm
                                self.boxNeighbors[selfIndex] += [intersectingBoxes[k]]
                                self.boxNeighbors[intersectingBoxes[k]] += [selfIndex]
                                foundNeighbor = True
                                break

                        # If we've already found a neighbor for this strand
                        # we shouldn't keep looking
                        if foundNeighbor:
                            break


        #################################################
        # Determine Connections via Dispersion
        #################################################
        # Now, we want to determine the connections between neighbors
        # We defined the dict boxNeighbors above to have empty
        # list for each entry
        if neighborMethod.lower() == 'discrete':
            dirVecArr = np.array([(1,0,0), (-1,0,0),
                                  (0,1,0), (0,-1,0),
                                  (0,0,1), (0,0,-1)])
       
            # If a box has a certain neighbor, we add the value
            # of that direction vector in the above array to the
            # list of neighbors in boxNeighbors

            # Only have to loop over the boxes that actually do have
            # points in them
            for i in range(len(self.boxIndices)):
                selfIndex = tuple(self.boxIndices[i])
                # Look at each neighbor
                for j in range(len(dirVecArr)):
                  
                    neighborIndex = tuple(self.boxIndices[i] + dirVecArr[j])

                    # Make sure the neighbor is still within the occupied volume
                    if True in [neighborIndex[k] >= volumeSizeInBoxes[k] or neighborIndex[k] < 0 for k in range(3)]:
                        continue

                    # If the neighbor has no points, it can't have a connection
                    if not self.boxContainsPoints[neighborIndex]:
                        continue

                    # No need to check if the box is already a neighbor
                    if neighborIndex in self.boxNeighbors[selfIndex]:
                        continue

                    isNeighbor = connectionCriteria(self.boxPoints[selfIndex], self.boxPoints[neighborIndex], neighborThreshold)

                    if isNeighbor:
                        self.boxNeighbors[selfIndex] += [neighborIndex]
                        self.boxNeighbors[neighborIndex] += [selfIndex]

        ################################
        # Count Up Connections
        ################################
        # Now that we have all of our connections, let's count them up
        # This is referred to as Vdim in the paper
        self.vDimArr = np.zeros_like(self.boxContainsPoints, dtype=np.int64)

        for i in range(volumeSizeInBoxes[0]):
            for j in range(volumeSizeInBoxes[1]):
                for k in range(volumeSizeInBoxes[2]):
                    self.vDimArr[(i,j,k)] = len(self.boxNeighbors[(i,j,k)])

        
        if debug:
            print(f'Found {np.sum(self.vDimArr)/2} connections.')


    def getCentroids(self):
        """
        Return the centroid position for each box that contains points.

        Ordering of points generally is arbitrary, but will always be the
        same ordering as in getAdjMat().
        """
        # Find the indices of boxes that have points
        filledBoxes = np.array(np.where(self.boxContainsPoints), dtype=np.int64).T
        # Compute the centroid of each of these
        centroids = np.array([np.mean(self.boxPoints[tuple(filledBoxes[i])], axis=0) for i in range(len(filledBoxes))])

        return centroids

    def getAdjMat(self):
        """

        """
        # Has to be in the same order as the return from getCentroids()
        # Find the indices of boxes that have points
        filledBoxes = np.array(np.where(self.boxContainsPoints), dtype=np.int64).T

        adjMat = np.zeros((len(filledBoxes), len(filledBoxes)))
        conversionDict = dict(zip([tuple(ind) for ind in filledBoxes], np.arange(self.numFilledBoxes)))

        for i in range(adjMat.shape[0]):
            neighbors = self.boxNeighbors[tuple(filledBoxes[i])]
                #neighborIndices = np.array(filledBoxes[i]) + np.array(neighborDirections)
            for j in range(len(neighbors)):
                adjMat[i,conversionDict[tuple(neighbors[j])]] = 1

        return adjMat
        
                    
    def _arbitrary_skeletonize(self, centroids, adjMat, threshold=-.8, secondaryAdjMat=None):
        """
        """

        # Randomly sample points, and reduce connections if they are
        # redundant

        # First, remove points that aren't connected to others
        connectedPoints = np.where(np.sum(adjMat, axis=0) > 0)[0]
        centroids = centroids[connectedPoints]
        adjMat = adjMat[connectedPoints][:,connectedPoints]

        # Randomly order
        order = np.arange(len(centroids))
        np.random.shuffle(order)

        # Compute direction vectors for all centroids
        dominantDirections = []

        for i in range(len(centroids)):
            hist, thetaBins, phiBins = angularHistogramAroundPoint(centroids, i, adjMat, smoothing=21)
            peakDirections = findDominantHistogramDirections(hist, thetaBins, phiBins, normalize=True)

            dominantDirections.append(peakDirections)

        ##############################
        # Line Reduction
        #
        # .---.---. => .------.
        #
        ##############################

        removedPoints = np.zeros(len(centroids), dtype=bool)
        for i in order:

            if removedPoints[i]:
                continue

            # First, check if this centroid is part of a line.
            # Criterion for this is having only two dominant
            # directions, which are almost diametrically opposed.
            if len(dominantDirections[i]) == 2 and np.dot(dominantDirections[i][0], dominantDirections[i][1]) <= threshold:
                # In this case, the current point isn't necessary, and we can just directly join the
                # two endpoints

                # Find the neighbors who comprise the endpoints of this line
                neighbors = [ind for ind in np.where(adjMat[i] > 0)[0] if not removedPoints[ind]]
                # Add secondary neighbors
                if hasattr(secondaryAdjMat, '__iter__'):
                    neighbors += [ind for ind in np.where(secondaryAdjMat[i] > 0)[0] if not removedPoints[ind] and not ind in neighbors]

                if len(neighbors) == 0:
                    continue

                separationVectors = centroids[i] - centroids[neighbors]

                # First endpoint
                firstNeighbor = neighbors[np.argmax(np.dot(dominantDirections[i][0], separationVectors.T))]
                # Second endpoint
                secondNeighbor = neighbors[np.argmax(np.dot(dominantDirections[i][1], separationVectors.T))]

                # Make them neighbors
                adjMat[firstNeighbor,secondNeighbor] = 1
                adjMat[secondNeighbor,firstNeighbor] = 1

                # Mark this point as obsolete
                removedPoints[i] = True


        # Adjust for the points we removed
        includePoints = np.where(removedPoints == False)[0]
        skeletonizedPoints = centroids[includePoints]
        skeletonizedAdjMat = adjMat[includePoints][:,includePoints]

        order = np.arange(len(skeletonizedPoints))
        np.random.shuffle(order)

        removedPoints = np.zeros(len(order), dtype=bool)

        return skeletonizedPoints, skeletonizedAdjMat

        ##############################
        # Parallel Line Reduction
        #
        # .---.
        # |   | => .---.
        # .---.
        #
        ##############################
        distanceThresholdSquared = (np.mean(self.boxSize) * .5)**2
        parallelProjectionThreshold = .8
        for i in order:
            # Here, we want to reduce square cycles into a single line
            # since otherwise we will get unphysical cycles.

            neighbors = [ind for ind in np.where(skeletonizedAdjMat[i] > 0)[0] if not removedPoints[ind]]

            # We only want to do this with if the two points are very close
            # to each other (since we could actually have some square cycles
            # in our data, and we woulnd't want to get rid of real features)

            neighborDirections = skeletonizedPoints[i] - skeletonizedPoints[neighbors]
           
            # Remove neighbors that are far away
            neighbors = [neighbors[j] for j in range(len(neighbors)) if np.sum(neighborDirections[j]**2) <= distanceThresholdSquared]
            # Iterate over neighbor indices that are close enough
            for n in range(len(neighbors)):
                # The neighbor should have an edge that is almost exactly parallel
                # to the current one, and the separation vector between the two
                # points should be orthogonal to these two edges
                secondOrderNeighbors = [ind for ind in np.where(skeletonizedAdjMat[neighbors[n]] > 0)[0] if not removedPoints[ind]]
                secondOrderNeighborDirections = skeletonizedPoints[neighbors[n]] - skeletonizedPoints[secondOrderNeighbors]

                parallelEdgePairs = [(j,k) for j in range(len(neighborDirections)) for k in range(len(secondOrderNeighborDirections)) if np.dot(neighborDirections[j], secondOrderNeighborDirections[k]) / (np.sqrt(np.sum(neighborDirections[j]**2)*np.sum(secondOrderNeighborDirections[k]**2))) >= parallelProjectionThreshold]

                for j in range(len(parallelEdgePairs)):
                    # Now check that the separation vector is orthogonal
                    if abs(np.dot(neighborDirections[n], neighborDirections[parallelEdgePairs[j][0]])) <= 1 - parallelProjectionThreshold and abs(np.dot(neighborDirections[n], neighborDirections[parallelEdgePairs[j][1]])) <= 1 - parallelProjectionThreshold:
                        # If this is the case, we should replace the four involved vertices with just two
                        # (seem ascii image at top of this section)
                        newInitialPoint = np.mean((skeletonizedPoints[i], skeletonizedPoints[neighbors[n]]), axis=0)
                        #newFinalPoints
                    


        return skeletonizedPoints, skeletonizedAdjMat



    def skeletonize(self, method='arbitrary', threshold=-.8, closed=False, secondRoundNeighborDetection=True, allowMergePoints=True, mergeThreshold=1, graphSizeThreshold=5):
        """
        """

        centroids = self.getCentroids()
        adjMat = self.getAdjMat()

        if method.lower() == 'arbitrary':
            skelPoints, skelAdjMat = self._arbitrary_skeletonize(centroids, adjMat, threshold)
                
            if allowMergePoints:
                mergeDistanceThreshold = np.mean(self.boxSize)/4 * mergeThreshold
                distanceMatrix = np.zeros((len(skelPoints), len(skelPoints)))
                for i in range(len(skelPoints)):
                    distanceMatrix[i] = np.sqrt(np.sum((skelPoints[i] - skelPoints)**2, axis=-1))
                
                # Now find points that are close together
                pointPairs = [[i,j] if i < j else [j,i] for i in range(len(skelPoints)) for j in np.where(distanceMatrix[i] < mergeDistanceThreshold)[0] if i != j]

                pointPairs = np.unique(pointPairs, axis=0)
                #print(pointPairs)

                # Now merge the pairs together
                # Have to keep track of new identities in case there is a triangle of points
                skelPointIdentities = np.arange(len(skelPoints))
                for i in range(len(pointPairs)):
                    # Take the mean of the old points
                    skelPoints[skelPointIdentities[pointPairs[i][0]]] = np.mean(skelPoints[skelPointIdentities[pointPairs[i]]], axis=0)

                    # Merge the adjacency matrix elements
                    skelAdjMat[skelPointIdentities[pointPairs[i][0]]] += skelAdjMat[skelPointIdentities[pointPairs[i][1]]]

                    # Renormalize to 1
                    #skelAdjMat[skelPointIdentities[pointPairs[i][0]]] = (skelAdjMat[skelPointIdentities[pointPairs[i][0]]] > 0).astype(np.int64)

                    # Identify the two points as the same
                    skelPointIdentities[pointPairs[i][1]] = pointPairs[i][0]

                # Remove the duplicate points
                pointsToKeep = [i for i in range(len(skelPoints)) if i in skelPointIdentities]
                skelPoints = skelPoints[pointsToKeep]
                skelAdjMat = skelAdjMat[pointsToKeep][:,pointsToKeep]
                # Rerun the skeletonization
                #skelPoints, skelAdjMat = self._arbitrary_skeletonize(skelPoints, skelAdjMat, threshold)
                                        

            if secondRoundNeighborDetection:
                # If we want to go through a second round of neighbor identification,
                # we can try to repair missing links in an otherwise continuous structure.
                # This just involves recalculating the adjacency matrix based on proximity
                # of other points (ignoring dominant directions). This only works because
                # we have already removed most points from the skeleton.

                kdTree = KDTree(skelPoints)
                # Use twice the box size for the radius (kind of arbitrary, but seems reasonable)
                distanceThreshold = np.mean(self.boxSize)
                neighborsByDistance = kdTree.query_ball_point(skelPoints, distanceThreshold)

                temporaryNeighbors = np.zeros_like(skelAdjMat)

                for i in range(len(skelPoints)):
                    temporaryNeighbors[i,neighborsByDistance[i]] = 1

                # Recalculate skeleton with new neighbors
                skelPoints, skelAdjMat = self._arbitrary_skeletonize(skelPoints, skelAdjMat, threshold, secondaryAdjMat=temporaryNeighbors)

            if closed:
                # Keep repeating until there aren't any points left with only a single neighbor
                while len(np.where(np.sum(skelAdjMat, axis=0) == 1)[0]) > 0:
                    # Remove points with only 1 neighbor
                    includeIndices = np.where(np.sum(skelAdjMat, axis=0) > 1)[0]
                    skelPoints = skelPoints[includeIndices]
                    skelAdjMat = skelAdjMat[includeIndices][:,includeIndices]
                    
                    # Rerun skeletonization
                    skelPoints, skelAdjMat = self._arbitrary_skeletonize(skelPoints, skelAdjMat, threshold)

            if graphSizeThreshold is not None:
                # Detect which structures are connected, and remove small side graphs that aren't connected
                # to the main one if they contain less than a certain number of nodes
                structures = detectConnectedStructures(skelAdjMat)
                #print(structures)
                #print(np.unique(structures, return_counts=True))

                counts = np.unique(structures, return_counts=True)
                nodesPerStructure = dict(zip(counts[0], counts[1]))
                pointsToKeep = [i for i in range(len(skelPoints)) if nodesPerStructure.get(structures[i], 0) > graphSizeThreshold]

                skelPoints = skelPoints[pointsToKeep]
                skelAdjMat = skelAdjMat[pointsToKeep][:,pointsToKeep]

            return skelPoints, skelAdjMat

        ################################
        # Merge V-Pairs and E-Pairs
        ################################

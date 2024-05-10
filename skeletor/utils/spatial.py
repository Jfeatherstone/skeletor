import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.signal import convolve

from .peak_finding import findPeaks2D

def partitionIntoBoxes(points, nBoxes, cubes=False, returnIndices=False):
    """
    Partition a set of points into boxes of equal size.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Array of N points in d dimensions
        
    nBoxes : int
        Number of boxes to partition the space into; not necessarily the
        number of unique box identities returned, since only non-empty
        boxes will be returned.
        
    cubes : bool
        Whether to partition the space into isotropic volumes (True), or to
        allow the dimensions of the boxes to vary between dimensions (False).

    returnIndices : bool
        Whether to also return the indices (i,j,k...) of each box (True)
        or not (False).
        
    Returns
    -------
    boxSize : numpy.ndarray[d]
        Dimensions of the subdivided spaces.
        
    boxIdentities : numpy.ndarray[N]
        Identities of the box to which each point belongs to.
        Guaranteed to be continuous interger labels, ie. the
        total number of occupied boxes, M, is:
        
            `np.max(np.unique(boxIdentities)) + 1`
            
        and the existence of box `i` implies the existence
        of box `i-1` for i > 0.
        
    boxIndices : numpy.ndarray[M,d]
        Indices of each box; note that M != N in most
        cases.

        Only returned if `returnIndices=True`.
        
    """

    occupiedVolumeBounds = np.array(list(zip(np.min(points, axis=0), np.max(points, axis=0))))

    # Pad very slightly
    volumeSize = (occupiedVolumeBounds[:,1] - occupiedVolumeBounds[:,0])*1.01
    #print(points.shape[-1])
    boxSize = volumeSize / nBoxes**(1/points.shape[-1]) # [x, y, z, ...]

    if cubes:
        # If we are partitioning into cubes, then we have to choose the dimension of the
        # side; we choose the finest dimension, because that seems reasonable.
        boxSize = np.repeat(np.min(boxSize), points.shape[-1])
    
    boxIdentities = np.floor((points - occupiedVolumeBounds[:,0]) / boxSize).astype(np.int64)

    # Now change box identities from (i,j,k) to just i
    boxLabels = [tuple(t) for t in np.unique(boxIdentities, axis=0)] # (i,j,k)
    # dictionary: {(i,j,k) : l}
    boxLabelConversion = dict(zip(boxLabels, np.arange(len(boxLabels))))
    linearBoxIdentities = np.array([boxLabelConversion[tuple(l)] for l in boxIdentities]) # l
    
    if returnIndices:
        # Upper left corner of the boxes
        #boxCorners = [tuple(occupiedVolumeBounds[:,0] + t*boxSize) for t in np.unique(boxIdentities, axis=0)]
        boxIndices = [tuple(t) for t in np.unique(boxIdentities, axis=0)]
        # Note that this conversion is slightly different than before since we
        # don't want the corner for each point, but for each box; see docstring
        boxCornersConversion = dict(zip(boxLabels, boxIndices))
        inverseLabelConversion = {v : k for k,v in boxLabelConversion.items()}
        linearBoxCorners = np.array([boxCornersConversion[inverseLabelConversion[l]] for l in np.unique(linearBoxIdentities)])*boxSize + occupiedVolumeBounds[:,0]

        return boxSize, linearBoxIdentities, linearBoxCorners
    
    return boxSize, linearBoxIdentities


def courseGrainField(points, values=None, defaultValue=0, latticeSpacing=None, kernel='gaussian', kernelSize=5, subsample=None, returnSpacing=False, returnCorner=False):
    """
    Course grains a collection of values at arbitrary points,
    into a discrete field.

    If `values=None`, course-grained field is the point density.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Spatial positions of N points in d-dimensional space.

    values : numpy.ndarray[N,[k]] or func(points)->numpy.ndarray[N,[k]] or None
        Field values at each point. Can be k-dimensional vector,
        resulting in k course-grained fields.

        Can also be a (vectorized) function that returns a value given
        a collection of points. eg. neighbor counting function. This
        functionality is provided such that if the function is computationally
        expensive, eg. neighbor counting, the points can be subdivided into
        batches and the course grained fields can be summed at the end. This
        is a way to approximate the course grained field for a huge (>1e6)
        number of points, while still remaining computationally feasible.
        See `subsample`.

        If `None`, returned field will be the point density.

    defaultValue : float or numpy.ndarray[k]
        The default value of the course-grained field;
        probably `0` for most applications.

    latticeSpacing : float or None
        The spacing of lattice points for the course-grained field.

        If `None`, will be chosen such that the largest-spanning axis
        has 100 lattice points, with other axes using the same spacing.

    kernel : str or numpy.ndarray[A,A]
        The kernel to course-grain the field with. 'gaussian'
        option is implemented as default, but a custom matrix
        can be provided. If using default gaussian option,
        kernel size can be set with `kernelSize`.

    kernelSize : int
        The kernel size to use if `kernel='gaussian'`.
        If a custom kernel is provided, this has no effect.

    returnSpacing : bool

    returnCorner : bool
    """

    # Calculate the bounds of the volume enclosing all of the data
    occupiedVolumeBounds = np.array(list(zip(np.min(points, axis=0), np.max(points, axis=0))))

    # Create a lattice with the selected scale for that cube
    if latticeSpacing is not None:
        spacing = latticeSpacing
    else:
        # Choose such that the largest spanning axis has 100 points
        spacing = (occupiedVolumeBounds[:,1] - occupiedVolumeBounds[:,0]) / 100

    fieldDims = np.ceil(1 + (occupiedVolumeBounds[:,1] - occupiedVolumeBounds[:,0])/(spacing)).astype(np.int64)

    # Calculate which lattice cell each scatter point falls into
    latticePositions = np.floor((points - occupiedVolumeBounds[:,0])/spacing).astype(np.int64)

    # Check if an array of values was passed for each point
    # Otherwise we just have a scalar field (and we'll collapse
    # the last dimension later on).
    if hasattr(values, '__iter__'):
        k = np.shape(values)[-1]
        valArr = values
    else:
        k = 1
        valArr = np.ones((np.shape(points)[0], 1))

    fieldArr = np.zeros((*fieldDims, k))
    # Instead of actually applying a gaussian kernel now, which would be
    # very inefficient since we'd need to sum a potentially very large number
    # of k*d dimensional matrices (more or less), we instead just assign each
    # lattice point, then smooth over it after with the specified kernel.
    # Where this might cause issues:
    # - If the lattice spacing is too large, you will get some weird artifacts
    #   from this process. Though in that case, you'll get a ton of artifacts from
    #   elsewhere too, so just don't use too large a lattice spacing :)
    #print(tuple(latticePositions[0]))
    for i in range(np.shape(points)[0]):
        fieldArr[tuple(latticePositions[i])] += valArr[i]

    # Now smooth over the field
    if kernel == 'gaussian':
        gaussianBlurKernel = np.zeros(np.repeat(kernelSize, np.shape(points)[-1]))
        singleAxis = np.arange(kernelSize)
        kernelGrid = np.meshgrid(*np.repeat([singleAxis], np.shape(points)[-1], axis=0))
        #kernelGrid = np.meshgrid(singleAxis, singleAxis, singleAxis)
        # No 2 prefactor in the gaussian denominator because I want the kernel to
        # decay nearly to 0 at the corners
        kernelArr = np.exp(-np.sum([(kernelGrid[i] - (kernelSize-1)/2.)**2 for i in range(np.shape(points)[-1])], axis=0) / (kernelSize))
        # Now account for however many dimensions k we have
        #kernelArr = np.repeat([kernelArr] if k > 1 else kernelArr, k, axis=0)

    # Otherwise, we expect that kernel should already be passed as a
    # proper square d-dimensional matrix
    else:
        kernelArr = kernel

    # Perform a convolution of the field with our kernel
    # 'same' keeps the same bounds on the field, but might cause
    # some weird effects near the boundaries
    # Divide out the sum of the kernel to normalize
    transConvolution = np.zeros_like(fieldArr.T)

    for i in range(k):
        # Note that convolve(x, y) == convolve(x.T, y.T).T
        # We need this so we can go over our k axis
        transConvolution[i] = convolve(fieldArr.T[i], kernelArr.T, mode='same') / np.sum(kernelArr)

    convolution = transConvolution.T

    # If k == 1, collapse the extra dimension
    if k == 1:
        convolution = convolution[..., 0]
    
    returnResult = [convolution]

    if returnSpacing:
        returnResult += [spacing]

    if returnCorner:
        returnResult += [occupiedVolumeBounds[:,0]]

    return returnResult if len(returnResult) > 1 else convolution


def pathIntegralAlongField(field, path, latticeSpacing=1, fieldOffset=None, debug=False):
    """
    Computes the path integral along a scalar discretized field in any
    dimension.

    Uses linear interpolation along the path, so works the best for smoothly-
    varying fields.

    Does not interpolate the steps along the path, so the input path steps
    should be appropriately broken up.

    Make sure that all of the quantities passed to this method use the same
    ordering of dimensions! For example, if you are integrating across an
    image, these often use y-x convention, whereas you may be tempted to
    put your path information in x-y format.
    
    Parameters
    ----------
    field : numpy.ndarray[N,M,...]
        Field over which to compute the path integral.

    path : numpy.ndarray[L,d]
        L ordered points representing a path
        through the field.

    latticeSpacing : float, or numpy.ndarray[d]
        The lattice spacing for the discretized field;
        can be a single value for all dimensions, or different
        values for each dimension.

    fieldOffset : numpy.ndarray[d] or None
        The position of the bottom left corner
        of the discrete lattice on which the field exists.

    debug : bool
        Whether to plot diagnostic information about the field
        and path. Only works if field is two dimensional.

    Returns
    -------
    result : float
    """
    # Scale the path to have no units
    scaledPath = path.astype(np.float64)
    
    if fieldOffset is not None:
        scaledPath -= fieldOffset
    
    scaledPath /= latticeSpacing

    # Generate the lattice positions that our field is defined on
    axes = [np.arange(d) for d in np.shape(field)]
    points = np.array(np.meshgrid(*axes, indexing='ij')).T
    points = points.reshape(np.product(points.shape[:-1]), points.shape[-1])
    
    # Generate a kdtree of our lattice points to detect the closest ones for
    # interpolation
    kdTree = KDTree(points)
    # Search for points within the distance of 1 lattice spacing
    # This guarantees that you won't interpolate from points that
    # conflict with each other.
    interpolationPoints = kdTree.query_ball_point(scaledPath, 1+1e-5)
    fieldValuesAlongPath = np.zeros(len(scaledPath))
    
    for i in range(len(scaledPath)):
        localPoints = interpolationPoints[i]
        # Compute distances to each nearby point
        localDistances = [np.sqrt(np.sum((scaledPath[i] - points[j])**2)) for j in localPoints]
        interpolationContributions = localDistances / np.sum(localDistances)
        fieldValuesAlongPath[i] = np.sum(interpolationContributions * np.array([field[tuple(l)] for l in points[localPoints]]))

    # We need to weigh our numerical integration by the step size
    # We just to do a centered step scheme. ie. the interpolated value computed
    # above for point i "counts" for half of the path approaching point i, and
    # half of the path leaving point i. Thus, the first and last point are weighed
    # only half, since they don't have an incoming or outgoing path each.
    # We also have to scale back to the original lattice spacing.
    pathSpacings = np.sqrt(np.sum(((scaledPath[1:] - scaledPath[:-1])/latticeSpacing)**2, axis=-1))
    symSpacing = (pathSpacings[:-1] + pathSpacings[1:])/2
    symSpacing = np.concatenate(([pathSpacings[0]/2], symSpacing, [pathSpacings[-1]/2]))

    pathIntegral = np.sum(symSpacing*fieldValuesAlongPath)
    
    if debug and points.shape[-1] == 2:
        plt.imshow(field) # imshow reverses reads y,x, while the other two do x,y
        plt.scatter(*points.T[::-1], s=1, c='white')
        plt.plot(*scaledPath.T[::-1], '-o', c='red', markersize=2)
        plt.show()
    
    return pathIntegral

def calculateAdjacencyMatrix(points, neighborDistance):
    """
    Calculate the number of neighbors (points within a certain distance) each
    point has.

    This can also be achieved by summing the rows of the adjacency matrix:

    ```
    adjMat = generateAdjMat(points, distance)
    numNeighbors = np.sum(adjMat, axis=0)
    ```

    but this requires few enough points that a full adjacency matrix can
    feasibly be computed.

    Parameters
    ----------
    points : numpy.ndarray[N, d]
        Positions of N points in d-dimensional space.

    neighborDistance : float
        The maximum distance between two points for which they will
        not be considered to be neighbors.

    """

    kdTree = KDTree(points)

    adjMat = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        neighbors = kdTree.query_ball_point(points[i], neighborDistance)
        adjMat[i,neighbors] = 1
   
    # Remove self connections
    adjMat = adjMat - np.eye(len(points))

    return adjMat


def angularHistogramAroundPoint(points, index=None, adjMat=None, smoothing=21, histBins=50):
    """
    Compute an angular histogram (axes are theta and phi angles) of directions to neighbors
    from the given point.

    Parameters
    ----------
    points : numpy.ndarray[N,3]
        All points in the point cloud (not just those
        that are neighbors). Neighbor points will be selected
        using the provided adjacency matrix.

    index : int or None
        The index of the point to compute the histogram for.

        If None, the histogram will be computed around the median of
        the provided points.

    adjMat : numpy.ndarray[N,N] or None
        The unweighted adjacency matrix for the points, ie.
        adjMat[i,j] == 1 iff the two i and j are neighbors, and
        0 otherwise.

        If `None`, all points in `points` will be considered to be
        in contact with the given point.

    smoothing : int (odd) or None
        Size of the gaussian smoothing kernel to use on the histogram.

    histBins : int
        Number of bins to use in generating the histogram.

    Returns
    -------
    hist : numpy.ndarray[N,N]
        2D histogram data, with each axis representing a spherical angle.

    thetaBins : numpy.ndarray[N]
        Values of theta angle for histogram axis.

    phiBins : numpy.ndarray[N]
        Values of phi angle for histogram axis.

    """
    # TODO: Make this work in arbitrary dimension
    if not hasattr(adjMat, '__iter__'):
        # Ones matrix (minus diagonals) such that every points except
        # the point itself is considered a neighbor.
        adjMat = np.ones((len(points), len(points))) - np.eye(len(points))

    if index is None:
        centralPoint = np.median(points, axis=0)
        centralAdj = np.ones(len(points))
    else:
        centralPoint = points[index]
        centralAdj = adjMat[index]

    # Compute the average edge orientation for each node
    neighbors = np.where(centralAdj > 0)
    # Compute the direction of the displacement between each point
    displacements = centralPoint - points[neighbors]

    # Compute slopes of displacement lines
    # z = y m_y + x m_x + b
    myArr = displacements[:,2] / (displacements[:,1] + 1e-8)
    mxArr = displacements[:,2] / (displacements[:,0] + 1e-8)
    interceptArr = [centralPoint.dot([-mxArr[j], -myArr[j], 1]) for j in range(len(displacements))]

    # Change the slopes into spherical coordinates
    magnitudes = np.sqrt(np.sum(displacements**2, axis=-1))
    theta = np.arccos(displacements[:,2]/(magnitudes + 1e-8))
    phi = np.sign(displacements[:,1]) * np.arccos(displacements[:,0] / (np.sqrt(np.sum(displacements[:,:2]**2, axis=-1)) + 1e-8))

    # Make sure there aren't any nan values
    goodIndices = np.array(np.array(np.isnan(theta), dtype=int) + np.array(np.isnan(phi), dtype=int) == 0, dtype=bool)
    thetaArr = theta[goodIndices]
    phiArr = phi[goodIndices]

    # Now turn into a 2D histogram
    # This is equivalent to course graining, and I already happen to have a
    # method for that for arbitrary dimension, so yay :D
    # Also note that I used to have variable bins for theta and phi just around
    # this data, but that means that the amount of smoothing applied is dependent
    # on the spread of the data, making choosing a single parameter value difficult.
    # Instead, now I use the full range for all angles, which guarantees that
    # a smoothing kernel of eg. 5 means the same thing for any set of points.
    #hist, thetaBins, phiBins = np.histogram2d(thetaArr, phiArr, bins=histBins)

    #if not smoothing is None and smoothing > 0:
    #    # `smoothing` is the kernel size
    #    singleAxis = np.arange(smoothing)
    #    kernelGrid = np.meshgrid(singleAxis, singleAxis)
    #    kernel = np.exp(-np.sum([(kernelGrid[i] - (smoothing-1)/2.)**2 for i in range(2)], axis=0) / (2*smoothing))
#
#        hist = convolve(hist, kernel, mode='same') / np.sum(kernel)

    latticeSpacing = 
    hist = courseGrainField()

    return hist, thetaBins, phiBins


def findDominantHistogramDirections(hist, thetaBins, phiBins, prevalence=.05, debug=False, normalizeHistogram=False, normalizeMoments=True):
    """
    Based on the angular histogram (see angularHistogramAroundPoint()),
    compute the dominant directions (moments), or the directions pointing towards
    neighbors.

    Parameters
    ----------
    hist : numpy.ndarray[N,N]
        2D histogram data, with each axis representing a spherical angle.

    thetaBins : numpy.ndarray[N]
        Values of theta angle for histogram axis.

    phiBins : numpy.ndarray[N]
        Values of phi angle for histogram axis.

    debug : bool
        Whether to plot the peak finding data (True) or not (False).

    Returns
    -------
    peakDirections : numpy.ndarray[P,3]
        Unit vectors in the dominant directions.
    """
    # Find peaks in the histogram
    # Peak prevalence is the range the peak spans; eg. a peak
    # that spans from the minimum to the maximum of the data has
    # a prevalence of 1
    peaks, prevalences = findPeaks2D(hist, minPeakPrevalence=prevalence, normalizePrevalence=normalizeHistogram)

    if len(peaks) == 0:
        return []

    # Convert from indices to angles
    peakThetaArr = thetaBins[np.array(peaks)[:,0]]
    peakPhiArr = phiBins[np.array(peaks)[:,1]]

    # Convert from (theta, phi) to (x, y, z)
    moments = np.array([[np.sin(peakThetaArr[i])*np.cos(peakPhiArr[i]),
                       np.sin(peakThetaArr[i])*np.sin(peakPhiArr[i]),
                       np.cos(peakThetaArr[i])] for i in range(len(peaks))])

    # Normalize
    if normalizeMoments:
        moments = np.array([dir / np.sqrt(np.sum(dir**2)) for dir in moments])

    if debug:
        plt.pcolor(thetaBins, phiBins, hist)
        plt.colorbar()
        for i in range(len(peaks)):
            plt.scatter(thetaBins[peaks[i][1]], phiBins[peaks[i][0]], c='red', s=50, alpha=.75)
        plt.show()

    return moments


# Directions for the faces of a cube
DISCRETE_DIR_VECTORS = np.array([[1,0,0], [-1,0,0],
                                 [0,1,0], [0,-1,0],
                                 [0,0,1], [0,0,-1]])

CARTESIAN_BASIS = np.array([[1,0,0],
                            [0,1,0],
                            [0,0,1]])

def discretizeDirectionVectors(dirVectors, basisVectors=CARTESIAN_BASIS):
    """
    Turn an arbitrary set of direction vectors (x,y,z)
    into a 6 dim vector representing directions along 3
    basis vectors, (x,y,z) by default.

    We need 6 values since if we were to just project the vectors
    along each basis vector, symmetric structures would not
    show up.

    eg. If we have a line, we could have the following direction
    vectors:
        [ 1,0,0]
        [-1,0,0]

    Just projecting these into the basis vectors and summing would give
    the discretized result [0,0,0], while using only positive contributions
    to the positive and negative bases gives [1,1,0,0,0,0], properly identifying
    that the direction vectors point in x and -x.

    Parameters
    ----------
    dirVectors : numpy.ndarray[N,3]
        Array of N direction vectors.

    basisVectors : numpy.ndarray[3,3]
        The basis vectors to which the arbitrary vectors
        should be projected into.

    Returns
    -------
    discreteDirVector : numpy.ndarray[6]
        Sum of all direction vectors dotted into
        direction vectors towards 6 closest neighbors
        (faces of a cube).
    """
    
    # The basis vectors should be given as 3 vectors, so we should
    # add in the negatives of the vectors as well.
    allBasisVectors = np.concatenate([(b, -b) for b in basisVectors], axis=0)

    discreteVector = np.zeros(6)

    for i in range(len(dirVectors)):
        currentVec = np.dot(dirVectors[i], allBasisVectors.T).T
        currentVec[currentVec < 0] = 0

        discreteVector += currentVec

    return discreteVector


def rotationMatrix(theta, phi, psi):
    """
    Generate the rotation matrix corresponding to rotating
    a point in 3D space.
    """
    return np.array([[np.cos(theta)*np.cos(psi), np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi), np.sin(phi)*np.sin(psi) - np.cos(psi)*np.cos(phi)*np.sin(theta)],
                     [-np.cos(theta)*np.sin(psi), np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi), np.sin(phi)*np.cos(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)],
                     [np.sin(theta), -np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]])



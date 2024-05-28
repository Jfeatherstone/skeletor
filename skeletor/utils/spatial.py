import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.signal import convolve

from pepe.topology import findPeaksMulti

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


def courseGrainField(points, values=None, defaultValue=0, latticeSpacing=None, fixedBounds=None, kernel='gaussian', kernelSize=5, subsample=None, returnSpacing=False, returnCorner=False):
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

    fixedBounds : numpy.ndarray[d] or None
        The bounds of the field to define the discretized
        grid over. If None, will be calculated based on the
        extrema of the provided points.

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
    # TODO: Make sure this works for 1D data
    dim = np.shape(points)[-1] if len(np.shape(points)) > 1 else 1

    if dim == 1:
        points = np.array(points)[:,None]
    
    if not hasattr(fixedBounds, '__iter__'):
        occupiedVolumeBounds = np.array(list(zip(np.min(points, axis=0), np.max(points, axis=0))))
    else:
        occupiedVolumeBounds = np.array(fixedBounds)
    
    # Create a lattice with the selected scale for that cube
    if latticeSpacing is not None:
        spacing = latticeSpacing
        # We also have to correct the occupied volume bounds if we were provided with
        # a fixed set of bounds. Otherwise, we will end up with an extra bin at the
        # end
        if hasattr(fixedBounds, '__iter__'):
            occupiedVolumeBounds[:,1] -= spacing
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


def angularHistogramAroundPoint(points, center, adjArr=None, neighborDistance=None, smoothing=5, histBins=50):
    """
    Compute an angular histogram (axes are theta and phi angles) of directions to neighbors
    from the given point.
    
    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Points to be used in determining computing
        the angular histogram, including the point for which
        the histogram is computed around. The central point
        is chosen using the `index` parameter.
        
        If not all points are to be used in creating the histogram,
        whether a given point is to be used or not used can
        be specified by the `adjMat` parameter.
        
    center : numpy.ndarray[d] or int
        The point to use as the center in calculating the angular
        histogram.
        
        If an integer, will be assumed to be the index of the point
        to compute the histogram for in the array `points`.
        
    adjArr : numpy.ndarray[N] or None
        The unweighted adjacency matrix row for the center point, ie.
        adjMat[i,j] == 1 iff the two i and j are neighbors, and 
        0 otherwise. Only points considered to be neighbors will
        be used in calculating the histogram.
        
        If `None`, all points will be considered neighbors.

        There should be very little computational cost to passing
        a larger array of points but then using a adjacency matrix
        to select a subset of those points (essentially just a call
        of `numpy.where`).

    neighborDistance : float or None
        The distance within which two points are considered to
        be neighbors. Only relevant if adjMat is not provided,
        and therefore needs to be calculated.
        
    smoothing : int (odd) or None
        Size of the gaussian smoothing kernel to use on the histogram. 
        
    histBins : int
        Number of bins to use for each axis in generating the histogram.
        
    Returns
    -------
    hist : numpy.ndarray[N,N]
        2D histogram data, with each axis representing a spherical angle.
        
    thetaBins : numpy.ndarray[N]
        Values of theta angle for histogram axis.
    
    phiBins : numpy.ndarray[N]
        Values of phi angle for histogram axis.

    """
    dim = np.shape(points)[-1]

    if not hasattr(center, '__iter__'):
        centerPoint = points[center]
    else:
        centerPoint = center

    # Calculate the adjacency matrix if given a neighbor distance
    if not hasattr(adjArr, '__iter__') and neighborDistance is not None:
        kdTree = KDTree(points)
        indices = kdTree.query_ball_point(centerPoint, r=neighborDistance)
        confirmedAdjArr = np.zeros(len(points))
        confirmedAdjArr[indices] = 1
        
    elif hasattr(adjArr, '__iter__'):
        confirmedAdjArr = adjArr
        
    else:
        confirmedAdjArr = np.ones(len(points))

    # If we have an adjacency array (either directly passed or
    # calculated) choosen a subset of the points as neighbors
    neighbors = np.where(confirmedAdjArr > 0)
    displacements = points[neighbors] - centerPoint
    
    # Compute the average edge orientation for each node
    # Compute the direction of the displacement between each point

    # This is done by calculating the generalized spherical coordinates
    # within which we care only about the angles

    # Convert to spherical coordinates
    sphericalCoords = cartesianToSpherical(displacements)
    radii = sphericalCoords[:,0]
    angleCoords = sphericalCoords[:,1:]

    # For d dimensions, we will have d-1 angles
    # d-2 of them will be bounded between [0, pi], and one
    # will be bounded between [0, 2pi]
    # Here, we will always put the unique one last.
    angleBounds = np.array((([np.pi]*(dim-2)) if dim > 2 else []) + [2*np.pi])
    angleBounds = np.array(list(zip(np.repeat(0, dim-1), angleBounds)))

    # Now generate the d-1 dimensional histogram
    latticeSpacing = angleBounds[:,1]/histBins
    hist = courseGrainField(angleCoords, latticeSpacing=latticeSpacing, fixedBounds=angleBounds, kernelSize=smoothing)
    
    angleAxes = np.array([angleBounds[:,1]*l for l in np.linspace(0, 1, histBins)]).T
    
    return hist, angleAxes


def findDominantHistogramDirections(hist, angleAxes, peakFindPrevalence=0.5, debug=False):
    r"""
    Based on the angular histogram (see `angularHistogramAroundPoint()`),
    compute the dominant directions, or the directions pointing towards
    neighbors.
    
    Parameters
    ----------
    hist : numpy.ndarray[N,N]
        d-1 dimensional histogram data, with each axis representing
        a spherical angle.
        
    angleAxes : list of numpy.ndarray[N]
        Values of the spherical angles for histogram axes. The
        unique axis (that has a range [0, 2pi]) should be last.
   
    debug : bool
        Whether to plot the peak finding data (`True`) or not (`False`).
        
    Returns
    -------
    peakDirections : numpy.ndarray[P,d]
        Unit vectors in the dominant directions in cartesian
        coordinates.
    """
    # Find peaks in the histogram
    # Peak prevalence is the range the peak spans; eg. a peak
    # that spans from the minimum to the maximum of the data has
    # a prevalence of 1
    peaks, prevalences = findPeaksMulti(hist, minPeakPrevalence=peakFindPrevalence, periodic=True)
    
    if len(peaks) == 0:
        return np.array([])
    
    # If we are peak finding in 1D, we need to add a dummy index
    # such that we can index as [peak number, dimension]
    if len(np.shape(peaks)) == 1:
        peaks = peaks[:,None]

    # Convert from indices to angles
    peakAngles = np.zeros((len(peaks), len(angleAxes)))
    for i in range(len(angleAxes)):
        # Just in case we are in the last pixel and round up, we should mod the length
        # of the axis
        peakAngles[:,i] = angleAxes[i][np.round(peaks[:,i]).astype(np.int64) % len(angleAxes[i])]

    # Convert from spherical to cartesian
    # Use 1 as the radius since we want unit vectors
    peakSphericalCoords = np.array([[1, *p] for p in peakAngles])    
    
    return sphericalToCartesian(peakSphericalCoords)


# Directions for the faces of a cube
DISCRETE_DIR_VECTORS = np.array([[1,0,0], [-1,0,0],
                                 [0,1,0], [0,-1,0],
                                 [0,0,1], [0,0,-1]])


def discretizeDirectionVectors(dirVectors, basisVectors=None):
    """
    Turn an arbitrary set of direction vectors in d-dimensions
    (x,y,z...) into a 2*d dimensional vector representing the
    contributions to (+x,-x,+y,-y,+z,-z,...)

    We need 2*d values since if we were to just project the vectors
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
    dirVectors : numpy.ndarray[N,d]
        Array of N direction vectors.

    basisVectors : None or numpy.ndarray[d,d]
        The basis vectors to which the arbitrary vectors
        should be projected into.

        If `None`, cartesian basis will be used.

    Returns
    -------
    discreteDirVector : numpy.ndarray[2*d]
        Sum of all direction vectors dotted into
        direction vectors towards 2*d closest neighbors
        (faces of a hypercube in d dimensions).
    """ 
    if len(dirVectors) == 0:
        return []

    dim = np.shape(dirVectors)[-1]

    if not hasattr(basisVectors, '__iter__') and basisVectors is None:
        # Cartesian basis (rows of the identity)
        basis = np.eye(dim)
    else:
        assert np.shape(basisVectors)[0] == np.shape(basisVectors)[1], f'Invalid basis provided (shape={np.shape(basisVectors)}); should have shape ({dim},{dim})'
        basis = basisVectors

    # The basis vectors should be given as d vectors, so we should
    # add in the negatives of the vectors as well.
    allBasisVectors = np.concatenate([(b, -b) for b in basis], axis=0)

    discreteVector = np.zeros(2*dim)

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


def cartesianToSpherical(points):
    """
    Convert a point from cartesian coordinates to
    generalized spherical coordinates.

    The input cartesian coordinates should follow the order:
        (x, y, ...)

    The output spherical coordinates will follow the order:
        (r, t_1, t_2, t_3, ..., p)

    where `r` is the radius, the angles `t_i` are bounded between
    `[0, pi]`, and the angle p is bounded between `[0, 2pi]`.

    Parameters
    ----------
    points : numpy.ndarray[d] or numpy.ndarray[N,d]
        One or multiple sets of points in cartesian coordinates.

    Returns
    -------
    cartesianPoints : numpy.ndarray[d] or numpy.ndarray[N,d]
        Output always matches the shape of the input `points`.

    """
    nPoints = np.shape(points)[0] if len(np.shape(points)) > 1 else 1
    dim = np.shape(points)[-1]

    if nPoints == 1 and len(np.shape(points)) == 1:
        arrPoints = np.array([points])
    else:
        arrPoints = np.array(points)

    # See note about this roll in sphericalToCartesian function.
    # It is to make sure the points can come in as (x,y,z...)
    #if dim >= 3:
    #    arrPoints = np.roll(arrPoints, 1, axis=-1)

    sphericalPoints = np.zeros((nPoints, dim))
    # See the page on n-spheres for these equations:
    # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    # Radius
    sphericalPoints[:,0] = np.sqrt(np.sum(arrPoints**2, axis=-1))
    for i in range(1, dim-1):
        sphericalPoints[:,i] = np.arctan2(np.sqrt(np.sum(arrPoints[:,i:]**2, axis=-1)), arrPoints[:,i-1])
    # The unique angle
    sphericalPoints[:,-1] = np.arctan2(arrPoints[:,-1], arrPoints[:,-2])

    # Remove extra dimensions if you only have a single point
    return sphericalPoints[0] if (nPoints == 1 and len(np.shape(points)) == 1) else sphericalPoints
   

def sphericalToCartesian(points):
    """
    Convert a point from generalized spherical coordinates to
    cartesian coordinates.

    The input spherical coordinates should follow the order:
        (r, t_1, t_2, t_3, ..., p)

    where `r` is the radius, the angles `t_i` are bounded between
    `[0, pi]`, and the angle p is bounded between `[0, 2pi]`.

    This transformation is consistent in 2D with:
        (r cos(p), r sin(p))
    and in 3D with:
        (r sin(t) cos(p), r sin(t) sin(p), r cos(t))
        
    Parameters
    ----------
    points : numpy.ndarray[d] or numpy.ndarray[N,d]
        One or multiple sets of points in spherical coordinates.

    Returns
    -------
    cartesianPoints : numpy.ndarray[d] or numpy.ndarray[N,d]
        Output always matches the shape of the input `points`.
    """
    nPoints = np.shape(points)[0] if len(np.shape(points)) > 1 else 1
    dim = np.shape(points)[-1]

    if nPoints == 1 and len(np.shape(points)) == 1:
        arrPoints = np.array([points])
    else:
        arrPoints = np.array(points)

    cartesianPoints = np.zeros((nPoints, dim))
    for i in range(dim-1):
        cartesianPoints[:,i] = arrPoints[:,0] * np.product(np.sin(arrPoints[:,1:i+1]), axis=-1) * np.cos(arrPoints[:,i+1])
    # The last one is different
    cartesianPoints[:,-1] = arrPoints[:,0] * np.product(np.sin(arrPoints[:,1:]), axis=-1)

    # For some reason, the calculation above shifts the order of
    # the cartesian components by one IF the dimension is greater than 3...
    # I really don't know why, but adding this line (and a complementary one
    # in sphericalToCartesian) makes sure that we always have the order
    # (x,y,z...)
    #if dim >= 3:
    #    cartesianPoints = np.roll(cartesianPoints, -1, axis=-1)
    
    # Remove extra dimensions if you only have a single point
    return cartesianPoints[0] if (nPoints == 1 and len(np.shape(points)) == 1) else cartesianPoints


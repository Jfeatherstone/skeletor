import numpy as np

from scipy.spatial import KDTree

from pepe.topology import findPeaksMulti

from .misc import cartesianToSpherical, sphericalToCartesian
from .course_grain import courseGrainField


def calculateAdjacencyMatrix(points, neighborDistance):
    """
    Calculate the neighbors (points within a certain distance) each
    point has.

    This is a symmetric matrix where the sum of each row is the degree
    (number of neighbors) of the corresponding node.

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
        adjMat[i, neighbors] = 1
   
    # Remove self connections
    adjMat = adjMat - np.eye(len(points))

    return adjMat


def angularHistogramAroundPoint(points, center, adjArr=None, neighborDistance=None, smoothing=1, histBins=50):
    """
    Compute an angular histogram (axes are theta and phi angles) of directions to neighbors
    from the given point.
    
    Parameters
    ----------
    points : numpy.ndarray[N, d]
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
        
    adjArr : numpy.ndarray[N], optional
        The unweighted adjacency matrix row for the center point, ie.
        adjMat[i,j] == 1 iff the two i and j are neighbors, and
        0 otherwise. Only points considered to be neighbors will
        be used in calculating the histogram.
        
        If not provided, all points will be considered neighbors.

        There should be very little computational cost to passing
        a larger array of points but then using a adjacency matrix
        to select a subset of those points (essentially just a call
        of `numpy.where`).

    neighborDistance : float, optional
        The distance within which two points are considered to
        be neighbors. Only relevant if adjMat is not provided,
        and therefore needs to be calculated.
        
    smoothing : int (odd), optional
        Size of the gaussian smoothing kernel to use on the histogram. A value
        of `1` means no smoothing will be performed.
        
    histBins : int
        Number of bins to use for each axis in generating the histogram.
        
    Returns
    -------
    hist : numpy.ndarray[N, N]
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
    # We don't actually need the radius for anything
    # radii = sphericalCoords[:,0]
    angleCoords = sphericalCoords[:, 1:]

    # For d dimensions, we will have d-1 angles
    # d-2 of them will be bounded between [0, pi], and one
    # will be bounded between [0, 2pi]
    # Here, we will always put the unique one last.
    angleBounds = np.array((([np.pi] * (dim - 2)) if dim > 2 else []) + [2 * np.pi])
    angleBounds = np.array(list(zip(np.repeat(0, dim - 1), angleBounds)))

    # Now generate the d-1 dimensional histogram
    latticeSpacing = angleBounds[:, 1] / histBins
    hist = courseGrainField(angleCoords, latticeSpacing=latticeSpacing, fixedBounds=angleBounds, kernelSize=smoothing)
    
    angleAxes = np.array([angleBounds[:, 1] * frac for frac in np.linspace(0, 1, histBins)]).T
    
    return hist, angleAxes


def findDominantHistogramDirections(hist, angleAxes, peakFindPrevalence=0.5, debug=False):
    r"""
    Based on the angular histogram (see `angularHistogramAroundPoint()`),
    compute the dominant directions, or the directions pointing towards
    neighbors.
    
    Parameters
    ----------
    hist : numpy.ndarray[N, N]
        d-1 dimensional histogram data, with each axis representing
        a spherical angle.
        
    angleAxes : list of numpy.ndarray[N]
        Values of the spherical angles for histogram axes. The
        unique axis (that has a range `[0, 2 pi]`) should be last.
   
    debug : bool, optional
        Whether to plot the peak finding data.
        
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
        peaks = peaks[:, None]

    # Convert from indices to angles
    peakAngles = np.zeros((len(peaks), len(angleAxes)))
    for i in range(len(angleAxes)):
        # Just in case we are in the last pixel and round up, we should mod the length
        # of the axis
        peakAngles[:, i] = angleAxes[i][np.round(peaks[:, i]).astype(np.int64) % len(angleAxes[i])]

    # Convert from spherical to cartesian
    # Use 1 as the radius since we want unit vectors
    peakSphericalCoords = np.array([[1, *p] for p in peakAngles])
    
    return sphericalToCartesian(peakSphericalCoords)


# Directions for the faces of a cube
DISCRETE_DIR_VECTORS = np.array([[1, 0, 0], [-1, 0, 0],
                                 [0, 1, 0], [0, -1, 0],
                                 [0, 0, 1], [0, 0, -1]])
"""
@private
"""


def discretizeDirectionVectors(dirVectors, basisVectors=None):
    """
    Turn an arbitrary set of direction vectors in `d`-dimensions
    `(x,y,z...)` into a `2*d` dimensional vector representing the
    contributions to `(+x,-x,+y,-y,+z,-z,...)`

    We need `2*d` values since if we were to just project the vectors
    along each basis vector, symmetric structures would not
    show up.

    eg. If we have a line, we could have the following direction
    vectors:

        [ 1,0,0]
        [-1,0,0]

    Just projecting these into the basis vectors and summing would give
    the discretized result `[0,0,0]`, while using only positive contributions
    to the positive and negative bases gives `[1,1,0,0,0,0]`, properly identifying
    that the direction vectors point in `x` and `-x`.

    Parameters
    ----------
    dirVectors : numpy.ndarray[N, d]
        Array of N direction vectors.

    basisVectors : numpy.ndarray[d, d], optional
        The basis vectors to which the arbitrary vectors
        should be projected into.

        If not provided, cartesian basis will be used.

    Returns
    -------
    discreteDirVector : numpy.ndarray[2 x d]
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
        assert np.shape(basisVectors)[0] == np.shape(basisVectors)[1], \
               f'Invalid basis provided (shape={np.shape(basisVectors)}); should have shape ({dim},{dim})'

        basis = basisVectors

    # The basis vectors should be given as d vectors, so we should
    # add in the negatives of the vectors as well.
    allBasisVectors = np.concatenate([(b, -b) for b in basis], axis=0)

    discreteVector = np.zeros(2 * dim)

    for i in range(len(dirVectors)):
        currentVec = np.dot(dirVectors[i], allBasisVectors.T).T
        currentVec[currentVec < 0] = 0

        discreteVector += currentVec

    return discreteVector

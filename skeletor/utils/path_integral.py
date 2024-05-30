import numpy as np
import matplotlib.pyplot as plt

import numba

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
    d = np.array(field).ndim

    if not hasattr(latticeSpacing, '__iter__'):
        spacing = np.repeat(latticeSpacing, d)
    else:
        spacing = latticeSpacing

    if d == 2 and not debug:
        return _pathIntegralAlongField2D(field, path, spacing, fieldOffset)
    elif d == 3 and not debug:
        return _pathIntegralAlongField3D(field, path, spacing, fieldOffset)
    else:
        return _pathIntegralAlongFieldMulti(field, path, spacing, fieldOffset, debug)

@numba.njit()
def _pathIntegralAlongField2D(field, path, latticeSpacing=1, fieldOffset=None):
    """
    numba-optimized function to compute a path integral in 2D; designed
    to be called by `pathIntegralAlongField()` only.

    Parameters
    ----------
    field : numpy.ndarray[N,M]
        Field over which to compute the path integral.

    path : numpy.ndarray[L,2]
        L ordered points representing a path
        through the field.

    latticeSpacing : float, or numpy.ndarray[2]
        The lattice spacing for the discretized field;
        can be a single value for all dimensions, or different
        values for each dimension.

    fieldOffset : numpy.ndarray[2] or None
        The position of the bottom left corner
        of the discrete lattice on which the field exists.

    Returns
    -------
    result : float
    """
    d = field.ndim

    # Scale the path to have no units
    scaledPath = path.astype(np.float64)

    if fieldOffset is not None:
        scaledPath -= fieldOffset

    for i in range(d):
        scaledPath[:,i] /= latticeSpacing[i]

    nearbyIndices = []
    for i in range(len(scaledPath)):
        possibleIndices = []
        for j in range(d):
            possibleIndices.append(np.array([np.floor(scaledPath[i,j]), np.ceil(scaledPath[i,j])]))
        totalCombinations = int(np.prod(np.array([np.float64(len(p)) for p in possibleIndices])))

        # Have to manually index to make numba happy
        result = np.zeros((totalCombinations, d))
        result[:,0] = np.repeat(possibleIndices[0], totalCombinations // len(possibleIndices[1]))
        result[:,1] = list(possibleIndices[1])*(totalCombinations // len(possibleIndices[0]))

        nearbyIndices.append(result)

    fieldValuesAlongPath = np.zeros(len(scaledPath))

    for i in range(len(scaledPath)):
        localPoints = nearbyIndices[i]
        # Compute distances to each nearby point
        # Add some tiny amount to avoid divide by zero issues
        localDistances = np.sqrt(np.sum((scaledPath[i] - localPoints)**2, axis=-1)) + 1e-10
        interpolationContributions = localDistances / np.sum(localDistances)

        # Have to do some weird indexing to make numba happy, but generally this is
        # just the dot product between interpolationContributions and field[all indices]
        for j in range(2**d):
            index = (int(localPoints[j][0]), int(localPoints[j][1]))
            fieldValuesAlongPath[i] += interpolationContributions[j] * field[index]

    # We need to weigh our numerical integration by the step size
    # We just to do a centered step scheme. ie. the interpolated value computed
    # above for point i "counts" for half of the path approaching point i, and
    # half of the path leaving point i. Thus, the first and last point are weighed
    # only half, since they don't have an incoming or outgoing path each.
    # We also have to scale back to the original lattice spacing.
    unscaledPath = scaledPath[1:] - scaledPath[:-1]
    for i in range(d):
        unscaledPath[:,i] *= latticeSpacing[i]

    pathSpacings = np.sqrt(np.sum((unscaledPath)**2, axis=-1))
    symSpacing = (pathSpacings[:-1] + pathSpacings[1:])/2
    symSpacing = np.concatenate((np.array([pathSpacings[0]/2]), symSpacing, np.array([pathSpacings[-1]/2])))

    pathIntegral = np.sum(symSpacing*fieldValuesAlongPath)

    return pathIntegral


@numba.njit()
def _pathIntegralAlongField3D(field, path, latticeSpacing=1, fieldOffset=None):
    """
    numba-optimized function to compute a path integral in 3D; designed
    to be called by `pathIntegralAlongField()` only.

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

    Returns
    -------
    result : float
    """
    d = field.ndim

    # Scale the path to have no units
    scaledPath = path.astype(np.float64)

    if fieldOffset is not None:
        scaledPath -= fieldOffset

    for i in range(d):
        scaledPath[:,i] /= latticeSpacing[i]

    nearbyIndices = []
    for i in range(len(scaledPath)):
        possibleIndices = []
        for j in range(d):
            possibleIndices.append(np.array([np.floor(scaledPath[i,j]), np.ceil(scaledPath[i,j])]))
        totalCombinations = int(np.nanprod(np.array([np.float64(len(p)) for p in possibleIndices])))

        # Have to manually index to make numba happy
        result = np.zeros((totalCombinations, d))
        result[:,0] = np.repeat(possibleIndices[0], 4)
        result[:,1] = list(possibleIndices[1]) * 4
        result[:,2] = list(np.repeat(possibleIndices[2], 2)) * 2
        #print(result)
        nearbyIndices.append(result)

    fieldValuesAlongPath = np.zeros(len(scaledPath))

    for i in range(len(scaledPath)):
        localPoints = nearbyIndices[i]
        # Compute distances to each nearby point
        # Add some tiny amount to avoid divide by zero issues
        localDistances = np.sqrt(np.sum((scaledPath[i] - localPoints)**2, axis=-1)) + 1e-10
        interpolationContributions = localDistances / np.sum(localDistances)

        # Have to do some weird indexing to make numba happy, but generally this is
        # just the dot product between interpolationContributions and field[all indices]
        for j in range(2**d):
            index = (int(localPoints[j][0]), int(localPoints[j][1]), int(localPoints[j][2]))
            fieldValuesAlongPath[i] += interpolationContributions[j] * field[index]

    # We need to weigh our numerical integration by the step size
    # We just to do a centered step scheme. ie. the interpolated value computed
    # above for point i "counts" for half of the path approaching point i, and
    # half of the path leaving point i. Thus, the first and last point are weighed
    # only half, since they don't have an incoming or outgoing path each.
    # We also have to scale back to the original lattice spacing.
    unscaledPath = scaledPath[1:] - scaledPath[:-1]
    for i in range(d):
        unscaledPath[:,i] *= latticeSpacing[i]

    pathSpacings = np.sqrt(np.sum((unscaledPath)**2, axis=-1))
    symSpacing = (pathSpacings[:-1] + pathSpacings[1:])/2
    symSpacing = np.concatenate((np.array([pathSpacings[0]/2]), symSpacing, np.array([pathSpacings[-1]/2])))

    pathIntegral = np.sum(symSpacing*fieldValuesAlongPath)

    return pathIntegral

def _pathIntegralAlongFieldMulti(field, path, latticeSpacing=1, fieldOffset=None, debug=False):
    """
    Unoptimized path integral function, but can be used in
    an arbitrary spatial dimension, and give debug information.

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
    d = field.ndim

    # Scale the path to have no units
    scaledPath = path.astype(np.float64)

    if fieldOffset is not None:
        scaledPath -= fieldOffset

    scaledPath /= latticeSpacing

    nearbyIndices = []
    for i in range(len(scaledPath)):
        possibleIndices = [np.floor(scaledPath[i]), np.ceil(scaledPath[i])]
        # We don't need to worry about duplicates if floor(i) == ceil(i)
        # since we normalize the contributions at the end.
        possibleIndices = np.array(possibleIndices).T

        # This next part is mostly copied from itertools' product
        # function.
        result = [[]]
        for pool in possibleIndices:
           result = [x+[y] for x in result for y in pool]

        nearbyIndices.append(np.array(result))

    fieldValuesAlongPath = np.zeros(len(scaledPath))

    for i in range(len(scaledPath)):
        localPoints = nearbyIndices[i]
        # Compute distances to each nearby point
        # Add some tiny amount to avoid divide by zero issues
        localDistances = np.sqrt(np.sum((scaledPath[i] - localPoints)**2, axis=-1)) + 1e-10

        interpolationContributions = localDistances / np.sum(localDistances)
        # Now weight the values at the nearby points by their distances
        fieldValuesAlongPath[i] = np.nansum(interpolationContributions * field[tuple(localPoints.astype(np.int32).T)])

    # We need to weigh our numerical integration by the step size
    # We just to do a centered step scheme. ie. the interpolated value computed
    # above for point i "counts" for half of the path approaching point i, and
    # half of the path leaving point i. Thus, the first and last point are weighed
    # only half, since they don't have an incoming or outgoing path each.
    # We also have to scale back to the original lattice spacing.
    pathSpacings = np.sqrt(np.sum(((scaledPath[1:] - scaledPath[:-1]) * latticeSpacing)**2, axis=-1))
    symSpacing = (pathSpacings[:-1] + pathSpacings[1:])/2
    symSpacing = np.concatenate((np.array([pathSpacings[0]/2]), symSpacing, np.array([pathSpacings[-1]/2])))

    pathIntegral = np.sum(symSpacing*fieldValuesAlongPath)

    if debug and d == 2:
        axes = [np.arange(d) for d in np.shape(field)]
        points = np.array(np.meshgrid(*axes, indexing='ij')).T
        points = points.reshape(np.product(points.shape[:-1]), points.shape[-1])

        plt.imshow(field) # imshow reverses reads y,x, while the other two do x,y
        plt.scatter(*points.T[::-1], s=1, c='white')
        plt.plot(*scaledPath.T[::-1], '-o', c='red', markersize=2)
        plt.colorbar()
        plt.show()

    return pathIntegral


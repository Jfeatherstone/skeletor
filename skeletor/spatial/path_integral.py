import numpy as np
import matplotlib.pyplot as plt

import numba


def lineIntersection(a1, b1, a2, b2):
    """
    Compute the intersection of two `d`-dimensional lines of the form:

        a + b*t

    Parameters
    ----------
    a1, a2 : numpy.ndarray[d]
        Intercept points of each line.

    b1, b2 : numpy.ndarray[d]
        Direction vectors of each line.

    Returns
    -------
    distance : float
        The closest pass distance between the two lines.
    """
    # Compute the cross product
    crossProd = np.cross(b1, b2)
    if crossProd == 0:
        return np.nan

    # Dot the cross product unit vector into the vector from intercept to intercept
    distance = np.dot(a2 - a1, crossProd) / np.sqrt(np.sum(crossProd**2))
    return distance


def pathIntegralAlongField(field, path, latticeSpacing=1, fieldOffset=None, debug=False):
    """
    Computes the path integral along a scalar discretized field in any
    dimension.

    Uses linear interpolation along the path, so works the best for smoothly-
    varying fields.

    Does not interpolate the steps along the path, so the input path steps
    should be appropriately broken up.

    This method is optimized using `numba` for two- and three-dimensional
    fields. In that case, this method is just a wrapper to make sure the
    types are all proper before calling the actual optimized method.

    Note
    ----

    Make sure that all of the quantities passed to this method use the same
    ordering of dimensions! For example, if you are integrating across an
    image, these often use `y`,`x` convention, whereas you may be tempted to
    put your path information in `x`,`y` format (physics convention).

    Parameters
    ----------
    field : numpy.ndarray[N, M, ...]
        Field over which to compute the path integral.

    path : numpy.ndarray[L, d]
        L ordered points representing a path
        through the field.

    latticeSpacing : float or numpy.ndarray[d]
        The lattice spacing for the discretized field;
        can be a single value for all dimensions, or different
        values for each dimension.

    fieldOffset : numpy.ndarray[d], optional
        The position of the bottom left corner of the discrete lattice
        on which the field exists. Assumed to be the origin if not provided.

    debug : bool, optional
        Whether to plot diagnostic information about the field
        and path. Only works if field is two dimensional.

    Returns
    -------
    result : float
        The path integral value.
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
        scaledPath[:, i] /= latticeSpacing[i]

    nearbyIndices = []
    for i in range(len(scaledPath)):
        possibleIndices = []
        for j in range(d):
            belowIndex = np.floor(scaledPath[i, j])
            aboveIndex = np.ceil(scaledPath[i, j])
           
            # So this should just be [] but there is a chance
            # that this list could contain no elements, which numba does
            # not like at all. So we have to give some indication of what
            # type this list will hold, without actually giving it any elements.
            possibilities = [0 for _ in range(0)]

            # Make sure that the indices are valid
            if belowIndex >= 0 and belowIndex < field.shape[j]:
                possibilities += [belowIndex]
            if aboveIndex >= 0 and aboveIndex < field.shape[j]:
                possibilities += [aboveIndex]
            
            possibleIndices.append(np.array(possibilities))

        totalCombinations = len(possibleIndices[0]) * len(possibleIndices[1])

        # Have to manually index to make numba happy
        result = np.zeros((totalCombinations, d))
        result[:, 0] = np.repeat(possibleIndices[0], totalCombinations // len(possibleIndices[1]))
        result[:, 1] = list(possibleIndices[1]) * (totalCombinations // len(possibleIndices[0]))

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
        unscaledPath[:, i] *= latticeSpacing[i]

    pathSpacings = np.sqrt(np.sum((unscaledPath)**2, axis=-1))
    symSpacing = (pathSpacings[:-1] + pathSpacings[1:]) / 2
    symSpacing = np.concatenate((np.array([pathSpacings[0] / 2]), symSpacing, np.array([pathSpacings[-1] / 2])))

    pathIntegral = np.sum(symSpacing * fieldValuesAlongPath)

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
    # This was written in a more general case,
    # so some of the code will look like it isn't specific
    # to 3D, but it has been made so here.
    d = field.ndim

    # Scale the path to have no units
    scaledPath = path.astype(np.float64)

    if fieldOffset is not None:
        scaledPath -= fieldOffset

    for i in range(d):
        scaledPath[:, i] /= latticeSpacing[i]

    nearbyIndices = []
    for i in range(len(scaledPath)):
        possibleIndices = []
        for j in range(d):
            belowIndex = np.floor(scaledPath[i, j])
            aboveIndex = np.ceil(scaledPath[i, j])
           
            # So this should just be [] but there is a chance
            # that this list could contain no elements, which numba does
            # not like at all. So we have to give some indication of what
            # type this list will hold, without actually giving it any elements.
            possibilities = [0 for _ in range(0)]

            # Make sure that the indices are valid
            if belowIndex >= 0 and belowIndex < field.shape[j]:
                possibilities += [belowIndex]
            if aboveIndex >= 0 and aboveIndex < field.shape[j]:
                possibilities += [aboveIndex]
            
            possibleIndices.append(np.array(possibilities))
        
        # Count up how many unique sets of indices we can create from the
        # possibilities for each axis
        totalCombinations = len(possibleIndices[0]) * len(possibleIndices[1]) * len(possibleIndices[2])

        # Have to manually index to make numba happy
        # TODO: Make sure this works if the z dimension is near the edge
        # I have tested it for x and y, but not the other one yet... if you have issues
        # around this area, it might be that.
        result = np.zeros((totalCombinations, d))
        result[:, 0] = np.repeat(possibleIndices[0], len(possibleIndices[1]) * len(possibleIndices[2]))
        result[:, 1] = list(possibleIndices[1]) * len(possibleIndices[0]) * len(possibleIndices[2])
        result[:, 2] = list(np.repeat(possibleIndices[2], len(possibleIndices[1]))) * len(possibleIndices[0])
        
        nearbyIndices.append(result)

    fieldValuesAlongPath = np.zeros(len(scaledPath))

    for i in range(len(scaledPath)):
        localPoints = nearbyIndices[i]
        if len(localPoints) == 0:
            continue

        # Compute distances to each nearby point
        # Add some tiny amount to avoid divide by zero issues
        localDistances = np.sqrt(np.sum((scaledPath[i] - localPoints)**2, axis=-1)) + 1e-8

        interpolationContributions = localDistances / np.sum(localDistances)

        # Have to do some weird indexing to make numba happy, but generally this is
        # just the dot product between interpolationContributions and field[all indices]
        for j in range(len(localPoints)):
            fieldValuesAlongPath[i] += interpolationContributions[j] * field[int(localPoints[j][0]),
                                                                             int(localPoints[j][1]),
                                                                             int(localPoints[j][2])]

    # We need to weigh our numerical integration by the step size
    # We just to do a centered step scheme. ie. the interpolated value computed
    # above for point i "counts" for half of the path approaching point i, and
    # half of the path leaving point i. Thus, the first and last point are weighed
    # only half, since they don't have an incoming or outgoing path each.
    # We also have to scale back to the original lattice spacing.
    unscaledPath = scaledPath[1:] - scaledPath[:-1]
    for i in range(d):
        unscaledPath[:, i] *= latticeSpacing[i]

    pathSpacings = np.sqrt(np.sum((unscaledPath)**2, axis=-1))
    symSpacing = (pathSpacings[:-1] + pathSpacings[1:]) / 2
    symSpacing = np.concatenate((np.array([pathSpacings[0] / 2]), symSpacing, np.array([pathSpacings[-1] / 2])))

    pathIntegral = np.sum(symSpacing * fieldValuesAlongPath)

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
            result = [x + [y] for x in result for y in pool]

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
    symSpacing = (pathSpacings[:-1] + pathSpacings[1:]) / 2
    symSpacing = np.concatenate((np.array([pathSpacings[0] / 2]), symSpacing, np.array([pathSpacings[-1] / 2])))

    pathIntegral = np.sum(symSpacing * fieldValuesAlongPath)

    if debug and d == 2:
        axes = [np.arange(d) for d in np.shape(field)]
        points = np.array(np.meshgrid(*axes, indexing='ij')).T
        points = points.reshape(np.product(points.shape[:-1]), points.shape[-1])

        plt.imshow(field)  # imshow reverses reads y,x, while the other two do x,y
        plt.scatter(*points.T[::-1], s=1, c='white')
        plt.plot(*scaledPath.T[::-1], '-o', c='red', markersize=2)
        plt.colorbar()
        plt.show()

    return pathIntegral

import numpy as np
from scipy.signal import convolve


def courseGrainField(points,
                     values=None,
                     defaultValue=0,
                     latticeSpacing=None,
                     fieldSize=None,
                     fixedBounds=None,
                     kernel='gaussian',
                     kernelSize=5,
                     subsample=None,
                     returnSpacing=False,
                     returnCorner=False):
    """
    Course grains a collection of values at arbitrary points,
    into a discrete field.

    If `values` is not provided, course-grained field is the point density.

    Parameters
    ----------
    points : numpy.ndarray[N, d]
        Spatial positions of N points in d-dimensional space.

    values : numpy.ndarray[N, [k]] or func(points)->numpy.ndarray[N, [k]], optional
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

        If not provided, returned field will be the point density.

    defaultValue : float or numpy.ndarray[k]
        The default value of the course-grained field, ie the value of the
        field when there are no points found in that lattice cell;
        probably `0` for most applications.

    latticeSpacing : float, optional
        The spacing of lattice points for the course-grained field.

        If not provided, will be chosen such that every dimension
        has 100 lattice points (implying potentially hetereogeneous
        spacing across dimensions).

    fieldSize : array_like(d), optional
        The size of the lattice to create; alternative option instead
        of giving a specific spacing to define the lattice.

        If a lattice spacing is given, this overrides the field size.

    fixedBounds : numpy.ndarray[d], optional
        The bounds of the field to define the discretized
        grid over. If not provided, will be calculated based on the
        extrema of the provided points.

    kernel : {'gaussian', 'ones'} or numpy.ndarray[A, A]
        The kernel to course-grain the field with. `'gaussian'`
        option is implemented as default, but a custom matrix
        can be provided. If using default gaussian option,
        kernel size can be set with `kernelSize`. `'ones'` will create
        a square matrix of 1 values.

    kernelSize : int
        The kernel size to use if `kernel='gaussian'`.
        If a custom kernel is provided, this has no effect.

    returnSpacing : bool
        Whether to return the spacing of the lattice alongside the field.

    returnCorner : bool
        Whether to return the corner of the lattice alongside the field.

    Returns
    -------
    field : numpy.ndarray[dims]
        Course grained field, with `d` or `d+1` dimensions, depending on
        if the field was course grained based on point density (`d`),
        or based on vector values (`d+1`).

    spacing : numpy.ndarray[3], optional
        The spacing of the lattice in each dimension. Only returned
        if `returnSpacing=True`.

    corner : numpy.ndarray[3], optional
        The corner of the lattice closest to the origin. Only returned
        if `returnCorner=True`.

    Notes
    -----

    The field calculation involves using fourier transforms to compute
    the convolution of the given kernel, which often leads to zero-value
    regions having some nonzero value on the order of machine precision
    (~1e-15). If you plan to use the result in an expression of the sort:

        np.where(field > 0)

    or something similar, it is highly recommended to clamp tiny values
    or use a proper threshold:

        field[field < 1e-10] = 0
        ...
        np.where(field > 1e-10)
    """
    # TODO: Make sure this works for 1D data
    dim = np.shape(points)[-1] if len(np.shape(points)) > 1 else 1

    if dim == 1:
        points = np.array(points)[:, None]
    
    if not hasattr(fixedBounds, '__iter__'):
        occupiedVolumeBounds = np.array(list(zip(np.min(points, axis=0), np.max(points, axis=0))))
    else:
        occupiedVolumeBounds = np.array(fixedBounds)

    # Add a small epsilon so we don't round up over the lattice size
    epsilon = 1e-6
    
    # Create a lattice with the selected scale for that cube
    if latticeSpacing is not None:
        spacing = latticeSpacing
        # We also have to correct the occupied volume bounds if we were provided with
        # a fixed set of bounds. Otherwise, we will end up with an extra bin at the
        # end
        if hasattr(fixedBounds, '__iter__'):
            occupiedVolumeBounds[:, 1] -= spacing

    elif latticeSpacing is None and hasattr(fieldSize, '__iter__'):
        # If we are given a specific size of the lattice, we use that
        # And the -1 is because we later add 1 
        spacing = (occupiedVolumeBounds[:, 1] - occupiedVolumeBounds[:, 0]) / (np.array(fieldSize) - 1) + epsilon

    else:
        # If we are given nothing, we choose such that the every axis has
        # 100 lattice points (arbitrary, but reasonable)
        spacing = (occupiedVolumeBounds[:, 1] - occupiedVolumeBounds[:, 0]) / 100

    # I was having some index out of bounds issues, so I think adding a tiny epsilon
    # to the range of the field helps resolve that.
    # Same with the +1
    fieldDims = np.ceil(1 + (occupiedVolumeBounds[:, 1] - occupiedVolumeBounds[:, 0] + epsilon) / (spacing))
    fieldDims = fieldDims.astype(np.int64)

    # Calculate which lattice cell each scatter point falls into
    latticePositions = np.floor((points - occupiedVolumeBounds[:, 0]) / spacing).astype(np.int64)

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
    # print(tuple(latticePositions[0]))
    for i in range(np.shape(points)[0]):
        fieldArr[tuple(latticePositions[i])] += valArr[i]

    # Now smooth over the field
    if kernel == 'gaussian':
        singleAxis = np.arange(kernelSize)
        kernelGrid = np.meshgrid(*np.repeat([singleAxis], np.shape(points)[-1], axis=0))
        # kernelGrid = np.meshgrid(singleAxis, singleAxis, singleAxis)
        # No 2 prefactor in the gaussian denominator because I want the kernel to
        # decay nearly to 0 at the corners
        kernelArr = np.exp(-np.sum([(kernelGrid[i] - (kernelSize - 1) / 2.)**2 for i in range(np.shape(points)[-1])],
                                   axis=0) / (kernelSize))
        # Now account for however many dimensions k we have
        # kernelArr = np.repeat([kernelArr] if k > 1 else kernelArr, k, axis=0)

    elif kernel == 'ones':
        kernelArr = np.ones([kernelSize for _ in range(dim)])

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
        returnResult += [occupiedVolumeBounds[:, 0]]

    return returnResult if len(returnResult) > 1 else convolution

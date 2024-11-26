import numpy as np


def pointsToImage(points, imageSize, kernelSize=1, minPointsPerPixel=1, padding=0):
    r"""
    NOTE: This function is now deprecated, and should be replaced with
    `skeletor.spatial.courseGrainField()`.


    Convert a set of points [N,d] into a
    binarized image [H,W[,L]] with a value of `1`
    if a point exists within a voxel, and `0` otherwise.

    Works on either 2D or 3D point clouds.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        N points in \(d \in [2,3] \) dimensions to be converted
        to an image.

    imageSize : (H,W) or (H,W,L)
        Size of the image in number of pixels for each
        dimension. At least one point will always be
        found on each face of the shape (ie. the points
        will be spread out maximally within the space).

    kernelSize : odd int
        The size of the kernel when placing points in the image.

        For each point, `kernelSize**2` pixels/voxels in the image
        will be given a value of `1`.

    minPointsPerPixel : int
        The minimum number of points located in a pixel/voxel
        for that pixel/voxel to be counted as having a value of `1`.
        Can be used to make the processing more noise resistant.

    padding : int
        The number of pixels to keep empty around the edge of the
        image.

    Returns
    -------

    img : numpy.ndarray[H,W] or numpy.ndarray[H,W,L]
        Image representation of the point cloud.
    """
    # TODO: Change from an integer kernel to a float one.

    nDim = np.shape(points)[-1]

    # Make sure the image has the correct number of dimensions
    assert len(imageSize) == nDim, f'{nDim} dimensional points attempted to be placed in {len(imageSize)} dimensional image.'
    # Make sure the kernel is odd
    assert kernelSize % 2 == 1, "Kernel must be even!"

    # Subtract off the minimum in each direction, and then
    # scale according to the given image shape.
    scaledPoints = points - np.min(points, axis=0)
    scaledPoints = scaledPoints / np.max(scaledPoints, axis=0)
    # We have to subtract one from the image size to account for
    # zero-indexing
    scaledPoints = padding + scaledPoints * (np.array(imageSize) - 1 - 2*padding)

    # Convert to integers
    scaledPoints = np.floor(scaledPoints).astype(np.int32)

    kernel = np.arange(-(kernelSize-1)//2, (kernelSize-1)//2+1)

    fullImage = np.zeros(imageSize, dtype=np.uint8)

    # The cardinal directions for the kernel
    if nDim == 2:
        directions = np.array([(i, j) for i in kernel for j in kernel])
    elif nDim == 3:
        directions = np.array([(i, j, k) for i in kernel for j in kernel for k in kernel])

    for p in scaledPoints:
        for d in directions:
            # Skip if we are out of bounds
            if True in (p+d-imageSize - padding >= 0) or True in (p+d - padding < 0):
                continue
            fullImage[tuple(p+d)] += 1

    fullImage[fullImage < minPointsPerPixel] = 0
    fullImage[fullImage > 0] = 1

    return fullImage


def imageToPoints(image, threshold=0):
    """
    Generate a list of indices (points) where the
    pixels/voxels in an image are greater than a certain
    value.

    Parameters
    ----------
    image : numpy.ndarray[H,W] or numpy.ndarray[H,W,L]
        A 2D or 3D image.

    threshold : float
        The threshold value for a pixel/voxel to be
        considered a point.

    Returns
    -------
    points : numpy.ndarray[N,d]
        Point representation of the image.
    """
    return np.array(np.where(image > threshold)).T

r"""
Implements --  or rather, wraps (`scipy`'s implementation)[https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html]
with some pre- and postprocessing -- the \( L_1 \) medial axis thinning algorithm described in Lee
et al. (1994). This algorithm works on either 2D or 3D point clouds.

This method is one of the oldest in the field, and other than being easy
to apply, doesn't have many advantages compared to modern methods.

Typical Applications
--------------------
1. Simple, solid objects, eg. the objects in the original paper or (scipy's manual page)[https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html].

Overview of Method
------------------
1. Convert an unordered set of points into a 2D or 3D image. For more
    information, see `skeletor.utils.pointsToImage()`.
2. Identify pixels or voxels that are on the edge of a feature.
3. Remove pixels or voxels from the previous step sequentially to
    preserve connectivity as best as possible.
4. Repeat steps 2 and 3 until no more pixels or voxels can be removed.
5. Convert the skeletonized image into a set of points and an adjacency
    matrix denoting which points are neighbors.

Parameters
----------
This method only requires two parameters, which control how the set of points is turned into an image
during preprocessing.

`imageMaxDim` is the length of the largest side of the image in pixels/voxels.

`kernelSize` is the size of the blur kernel applied when registering points in the image.

References
----------
Lee, T. C., Kashyap, R. L., & Chu, C. N. (1994). Building Skeleton Models via 3-D Medial Surface Axis Thinning Algorithms. CVGIP: Graphical Models and Image Processing, 56(6), 462–478. https://doi.org/10.1006/cgip.1994.1042

"""

import numpy as np
import matplotlib.pyplot as plt

import skimage
from scipy.spatial import KDTree

from skeletor.utils import pointsToImage, imageToPoints
from skeletor.utils import plotSpatialGraph


def skeletonize_medialThinning(points, imageMaxDim=100, kernelSize=1, minPointsPerPixel=1, debug=False):
    r"""
    \( L_1 \) medial axis thinning procedure to skeletonize a 2D or 3D set of
    points.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Unordered point cloud of 2 or 3 dimensions.

    imageMaxDim : int 
        The number of pixels/voxels in the largest dimension
        of the point cloud. The other 1 or 2 dimensions will
        be less than or equal to this value.

    kernelSize : odd int 
        The kernel used in turning the array of points into an
        image. See `skeletor.utils.pointsToImage()` for more information.

    debug : bool
        Whether to print diagnostic plots (`True`) or not (`False`).

    Returns
    -------
    skeleton : numpy.ndarray[M,d]
        Skeletonized points. Arbitrary ordering, but
        always the same order as the adjacency matrix.

    adjMat : numpy.ndarray[M,M]
        Adjacency matrix with `1` values representing
        connections, and `0` otherwise.
    """
    dim = np.shape(points)[-1]

    # Compute the aspect ratio of our point cloud
    dimExtents = np.max(points, axis=0) - np.min(points, axis=0)

    imageSize = dimExtents * (imageMaxDim / np.max(dimExtents))
    imageSize = imageSize.astype(np.int32)

    # Turn our points into an image
    image = pointsToImage(points, imageSize, kernelSize, minPointsPerPixel)

    # Now apply scipy's algorithm
    # 'lee' refers to the paper from 1994 cited above
    skeleton = skimage.morphology.skeletonize(image, method='lee')
    
    # Now convert our image back to a collection of points
    skelPoints = imageToPoints(skeleton).astype(np.float64)

    # And we can determine which points are neighbors based on their
    # distance from each other. There are other ways to do this,
    # but generally we will take adjacent (including diagonally) pixels to be neighbors
    # (this method is in-spirit of the original work I feel)
    # eye is the identity, since the diagonal should always be
    # 1's in an adjacency matrix.
    adjMat = np.eye(len(skelPoints), dtype=np.uint8)

    # This type of calculation is most efficiently done using a kdtree
    kdTree = KDTree(skelPoints)

    # sqrt(2) + eps for radius so we only find points that are directly
    # adjacent or diagonally adjacent.
    neighborIndices = kdTree.query_ball_point(skelPoints, np.sqrt(2)+1e-5)
    # The [1:] is to remove the self connection, since
    # we already added the identity matrix in earlier.
    properIndices = np.array([(i,j) for j in range(len(skelPoints)) for i in neighborIndices[j][1:]], dtype=np.int32).T

    # Scale the coordinates back
    # We do this after calculating neighbors, since we need the well-
    # defined scale of the pixel indices to do that
    skelPoints = (skelPoints / imageSize) * dimExtents + np.min(points, axis=0)
   
    # Populate the adjacency matrix (if we have any neighbors)
    if len(properIndices) > 0:
        adjMat[tuple(properIndices[0]), tuple(properIndices[1])] = 1

    if debug:
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(1, 4, 1, projection='3d' if dim == 3 else None)
        ax.scatter(*points.T)
        ax.set_title('Point Cloud')

        ax = fig.add_subplot(1, 4, 2, projection='3d' if dim == 3 else None)
        if dim == 3:
            ax.voxels(image, alpha=.8)
        elif dim == 2:
            ax.imshow(image)

        ax.set_title('Input Image')

        ax = fig.add_subplot(1, 4, 3, projection='3d' if dim == 3 else None)
        if dim == 3:
            ax.voxels(skeleton, alpha=.8)
        elif dim == 2:
            ax.imshow(skeleton)

        ax.set_title('Skeletonized Image')

        ax = fig.add_subplot(1, 4, 4, projection='3d' if dim == 3 else None)
        plotSpatialGraph(skelPoints, adjMat, ax)
        ax.set_title('Skeleton Graph')

        fig.tight_layout()
        plt.show()

    return skelPoints, adjMat

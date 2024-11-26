import numpy as np


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
    volumeSize = (occupiedVolumeBounds[:, 1] - occupiedVolumeBounds[:, 0])*1.01
    boxSize = volumeSize / nBoxes**(1/points.shape[-1])  # [x, y, z, ...]

    if cubes:
        # If we are partitioning into cubes, then we have to choose the dimension of the
        # side; we choose the finest dimension, because that seems reasonable.
        boxSize = np.repeat(np.min(boxSize), points.shape[-1])
    
    boxIdentities = np.floor((points - occupiedVolumeBounds[:, 0]) / boxSize).astype(np.int64)

    # Now change box identities from (i,j,k) to just i
    boxLabels = [tuple(t) for t in np.unique(boxIdentities, axis=0)]  # (i,j,k)
    # dictionary: {(i,j,k) : l}
    boxLabelConversion = dict(zip(boxLabels, np.arange(len(boxLabels))))
    linearBoxIdentities = np.array([boxLabelConversion[tuple(label)] for label in boxIdentities])  # l
    
    if returnIndices:
        # Upper left corner of the boxes
        # boxCorners = [tuple(occupiedVolumeBounds[:,0] + t*boxSize) for t in np.unique(boxIdentities, axis=0)]
        boxIndices = [tuple(t) for t in np.unique(boxIdentities, axis=0)]
        # Note that this conversion is slightly different than before since we
        # don't want the corner for each point, but for each box; see docstring
        boxCornersConversion = dict(zip(boxLabels, boxIndices))
        inverseLabelConversion = {v: k for k, v in boxLabelConversion.items()}
        linearBoxCorners = np.array([
                           boxCornersConversion[inverseLabelConversion[label]] for label in np.unique(linearBoxIdentities)
                           ])*boxSize + occupiedVolumeBounds[:, 0]

        return boxSize, linearBoxIdentities, linearBoxCorners
    
    return boxSize, linearBoxIdentities


def rotationMatrix(theta, phi, psi):
    """
    Generate the rotation matrix corresponding to rotating
    a point in 3D space.
    """
    return np.array([[np.cos(theta)*np.cos(psi),
                      np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi),
                      np.sin(phi)*np.sin(psi) - np.cos(psi)*np.cos(phi)*np.sin(theta)],
                     [-np.cos(theta)*np.sin(psi),
                      np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi),
                      np.sin(phi)*np.cos(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)],
                     [np.sin(theta),
                      -np.sin(phi)*np.cos(theta),
                      np.cos(phi)*np.cos(theta)]])


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

    sphericalPoints = np.zeros((nPoints, dim))
    # See the page on n-spheres for these equations:
    # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    # Radius
    sphericalPoints[:, 0] = np.sqrt(np.sum(arrPoints**2, axis=-1))
    for i in range(1, dim-1):
        sphericalPoints[:, i] = np.arctan2(np.sqrt(np.sum(arrPoints[:, i:]**2, axis=-1)), arrPoints[:, i-1])
    # The unique angle
    sphericalPoints[:, -1] = np.arctan2(arrPoints[:, -1], arrPoints[:, -2])

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
        cartesianPoints[:, i] = arrPoints[:, 0] * np.product(np.sin(arrPoints[:, 1:i+1]), axis=-1)  \
                                * np.cos(arrPoints[:, i+1])

    # The last one is different
    cartesianPoints[:, -1] = arrPoints[:, 0] * np.product(np.sin(arrPoints[:, 1:]), axis=-1)

    # Remove extra dimensions if you only have a single point
    return cartesianPoints[0] if (nPoints == 1 and len(np.shape(points)) == 1) else cartesianPoints

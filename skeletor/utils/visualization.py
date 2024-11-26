import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
import itertools


def visualizeRegionAroundPoints(scatterPoints, pIndex, innerRadius=1, outerRadius=2, innerColor='tab:red', outerColor='tab:grey', centerColor='tab:blue', lineColor=None, s=[2, 5, 30]):
    """
    Plot a point in a point cloud, as well as the local neighborhood
    of that point

    Legacy carryover method. Will likely be removed soon.
    """

    kdTree = KDTree(scatterPoints)

    innerPoints = scatterPoints[kdTree.query_ball_point(scatterPoints[pIndex], innerRadius)]
    outerPoints = scatterPoints[kdTree.query_ball_point(scatterPoints[pIndex], outerRadius)]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d' if np.shape(scatterPoints)[-1] == 3 else None)

    ax.scatter(*outerPoints.T, s=s[0], c=outerColor)
    ax.scatter(*innerPoints.T, s=s[1], c=innerColor)
    ax.scatter(*scatterPoints[pIndex], s=s[2], c=centerColor)

    ax.set_xlim([scatterPoints[pIndex, 0] - outerRadius, scatterPoints[pIndex, 0] + outerRadius])
    ax.set_ylim([scatterPoints[pIndex, 1] - outerRadius, scatterPoints[pIndex, 1] + outerRadius])
    if np.shape(scatterPoints)[-1] == 3:
        ax.set_zlim([scatterPoints[pIndex, 2] - outerRadius, scatterPoints[pIndex, 2] + outerRadius])

    if lineColor is not None:
        for i in range(len(innerPoints)):
            ax.plot(*list(zip(scatterPoints[pIndex], innerPoints[i])), c=lineColor)
 
    return fig


def getBoxLines(corner, boxSize, basis=None):
    """
    Get the set of lines representing the edges of a box in 3D.

    Parameters
    ----------
    corner : numpy.ndarray[3]
        Position of the corner closest to the origin.

    boxSize : numpy.ndarray[3]
        The size of the box in each direction.

    basis : numpy.ndarray[3,3] or None
        Basis along which the box is aligned, or None for default
        Cartesian alignment.

    Returns
    -------
    lines : numpy.ndarray[12,2,3]
        Start and end points of each of the 12 lines
        that comprise the box edges.
    """
    dim = len(corner)

    if not hasattr(basis, '__iter__') and basis is None:
        # Cartesian basis (rows of the identity)
        basis = np.eye(dim)
    else:
        assert np.shape(basis)[0] == np.shape(basis)[1], f'Invalid basis provided (shape={np.shape(basis)}); should have shape ({dim},{dim})'

    # Unit cube (sorta, side lengths are actually 2)
    r = [-1, 1]
    directions = np.array(list(itertools.product(*[r for _ in range(dim)])))

    # Transform to given basis
    directions = np.array([np.dot(d, basis) for d in directions])

    # Choose only lines that have a magnitude of 2 (since we have a sorta unit cube),
    # removing diagonal lines.
    # For some godforsaken reason, using 64 bit floats will
    # identify two side lengths as different even though they are
    # the same (no idea why that's an issue here, it's not like I'm
    # using super tiny side lengths...) so we have to cast to 32 bit
    # floats.
    lines = np.array([c for c in itertools.combinations(directions, 2) if np.sqrt(np.sum((c[1]-c[0])**2)).astype(np.float32) == r[1]-r[0]])

    # Now account for corner and boxsize
    lines = [((c[0]+1)*boxSize/2 + corner, (c[1]+1)*boxSize/2 + corner) for c in lines]

    return np.array(lines)


def plotBox(corner, boxSize, ax=None, basis=None, **kwargs):
    """
    Plot a box in 3D using matplotlib.

    Parameters
    ----------
    corner : numpy.ndarray[3]
        Position of the corner closest to the origin.

    boxSize : numpy.ndarray[3]
        The size of the box in each direction.

    ax : matplotlib.pyplot.axis or None
        The axis on which to plot the box. If None, a new
        axis will be created.

    basis : numpy.ndarray[3,3] or None
        Basis along which the box is aligned, or None for default
        Cartesian alignment.

    kwargs
        Other keyword arguments for the `matplotlib.pyplot.plot()` method.

    Returns
    -------
    figure : matplotlib.pyplot.figure
        Current figure.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    lines = getBoxLines(corner, boxSize, basis)

    for v1, v2 in lines:
        ax.plot(*zip(v1, v2), **kwargs)

    return plt.gcf()


def plotSpatialGraph(points, adjMat, ax=None, scatterKwargs={'c': "tab:blue"}, lineKwargs={'c': "tab:red"}):
    """
    Plot a spatially embedded graph using matplotlib.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Locations of graph nodes.

    adjMat : numpy.ndarray[N,N]
        The unweighted adjacency matrix for the graph.

    ax : matplotlib.pyplot.axis or None
        The axis on which to plot the graph. If None, a new
        axis will be created.

    scatterKwargs : dict
        Keyword arguments for the `matplotlib.pyplot.scatter()` function.

    lineKwargs : dict
        Keyword arguments for the `matplotlib.pyplot.plot()` function.

    Returns
    -------
    figure : matplotlib.pyplot.figure
        Current figure.
    """
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if np.shape(points)[-1] == 3 else None)

    ax.scatter(*points.T, **scatterKwargs)

    for i in range(len(adjMat)):
        edgeIndices = np.where(adjMat[i] > 0)[0]
        for j in edgeIndices:
            # Skip the self-connections
            if i == j:
                continue
            ax.plot(*list(zip(points[i], points[j])), **lineKwargs)

    return plt.gcf()

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
import itertools

def visualizeRegionAroundPoints(scatterPoints, pIndex, innerRadius=15, outerRadius=100, innerColor='tab:red', outerColor='tab:grey', centerColor='tab:blue', lineColor=None, s=[2,5,30]):

    kdTree = KDTree(scatterPoints)

    innerPoints = scatterPoints[kdTree.query_ball_point(scatterPoints[pIndex], innerRadius)]
    outerPoints = scatterPoints[kdTree.query_ball_point(scatterPoints[pIndex], outerRadius)]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(outerPoints[:,0], outerPoints[:,1], outerPoints[:,2], s=s[0], c=outerColor)
    ax.scatter(innerPoints[:,0], innerPoints[:,1], innerPoints[:,2], s=s[1], c=innerColor)
    ax.scatter(scatterPoints[pIndex,0], scatterPoints[pIndex,1], scatterPoints[pIndex,2], s=s[2], c=centerColor)

    ax.set_xlim([scatterPoints[pIndex,0] - 1.5*outerRadius, scatterPoints[pIndex,0] + 1.5*outerRadius])
    ax.set_ylim([scatterPoints[pIndex,1] - 1.5*outerRadius, scatterPoints[pIndex,1] + 1.5*outerRadius])
    ax.set_zlim([scatterPoints[pIndex,2] - 1.5*outerRadius, scatterPoints[pIndex,2] + 1.5*outerRadius])

    if not lineColor is None:
        for i in range(len(innerPoints)):
            ax.plot(*list(zip(scatterPoints[pIndex], innerPoints[i])), c=lineColor)
 

    return fig

CARTESIAN_BASIS = np.array([[1,0,0],
                            [0,1,0],
                            [0,0,1]])

def plotBox(corner, boxSize, ax=None, basis=CARTESIAN_BASIS, **kwargs):
    """
    """
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    # Unit cube (sorta, side lengths are actually 2)
    r = [-1, 1]
    directions = np.array(list(itertools.product(r, r, r)))

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

    for v1, v2 in lines:
        ax.plot3D(*zip(v1, v2), **kwargs)

    return plt.gcf()

def plotSpatialGraph(points, adjMat, ax=None, scatterKwargs={'c':"tab:blue"}, lineKwargs={'c':"tab:red"}):
    """
    """
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    ax.scatter(*points.T, **scatterKwargs)

    for i in range(len(adjMat)):
        edgeIndices = np.where(adjMat[i] > 0)[0]
        for j in edgeIndices:
            # Skip the self-connections
            if i == j:
                continue
            ax.plot(*list(zip(points[i], points[j])), **lineKwargs)

    return plt.gcf()


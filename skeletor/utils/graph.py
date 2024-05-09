import numpy as np

from anytree import Node

def detectConnectedStructures(adjMat):
    """

    """

    structureIdentity = np.zeros(len(adjMat), dtype=np.int64) - 1
    numStructures = -1

    for i in range(len(adjMat)):

        if structureIdentity[i] >= 0:
            continue

        numStructures += 1
        structureIdentity[i] = numStructures 
        # Add the initial points
        pointsToCheck = list(np.where(adjMat[i] > 0)[0])

        while len(pointsToCheck) > 0:
            potentialPoints = np.where(adjMat[pointsToCheck[0]] > 0)[0]
            potentialPoints = [p for p in potentialPoints if structureIdentity[p] == -1]

            structureIdentity[pointsToCheck[0]] = numStructures

            pointsToCheck = pointsToCheck[1:] + potentialPoints


    return structureIdentity

def identifyCycles(adjMat, N):
    """
    Identify cycles of (exactly) length N in the graph
    described by an adjacency matrix.
    
    Parameters
    ----------
    adjMat : numpy.ndarray[M,M]
        Adjacency matrix for M points, where
        a value adjMat[i,j] > 0 signifies points
        i and j have an edge between them.
        
    N : int
        The length of cycles to find.
        
    Returns
    -------
    cycles : numpy.ndarray[k,N]
        Indices of points that form k
        cycles of length N.
    """

    if N <= 2:
        raise Exception(f'Invalid cycle number ({N} <= 2) provided!')    

    cycles = []
        
    for i in range(len(adjMat)):
        nodes = [[Node(i)]]

        neighbors = [ind for ind in np.where(adjMat[i] > 0)[0] if ind != i]
        
        nodes += [[Node(ind, parent=nodes[0][0]) for ind in neighbors]]
        
        for level in range(1,N):
            newNodes = []
            #print(level, nodes[level])
            
            for j in range(len(nodes[level])):
                #print([ind for ind in np.where(adjMat[nodes[level][j].name] > 0)[0]])
                #print([n.name for n in nodes[-1][j].ancestors[-N+2:]])

                neighbors = [ind for ind in np.where(adjMat[nodes[-1][j].name] > 0)[0] if not ind in [n.name for n in nodes[-1][j].ancestors[-N+2:]]]
                newNodes += [Node(ind, parent=nodes[level][j]) for ind in neighbors if nodes[level][j].name != i]

            if len(newNodes) > 0:
                nodes += [newNodes]
            else:
                break
           
        #print(nodes)
        currentNodeCycles = [n for levels in nodes for n in levels if n.name == i and n != nodes[0][0]]
        
        # Convert to list of indices
        currentNodeCycles = [[an.name for an in n.ancestors] for n in currentNodeCycles]
        
        cycles += currentNodeCycles
        
    # Find unique entries
    # Note that we can't just sort the indices, since cycles of
    # length > 3 have a specific ordering, so we first have
    # to identify which ones involve the same N points, then
    # take the first instance of that one (ensuring the order doesn't
    # change).
    # The quickest way to do this is just to take a very non-linear but symmetric
    # function of the indices, which should uniquely identify each unordered set of
    # indices.
    # The choice of nonlinear function doesn't really matter
    cycleIdentities = [np.sum((np.array(c) + 2)**8) for c in cycles]
    uniqueCycleIdentities = np.unique(cycleIdentities)
    uniqueCycleIndices = [np.where(cycleIdentities == uci)[0][0] for uci in uniqueCycleIdentities]

    return np.array(cycles)[uniqueCycleIndices]


def computeCyclePathLength(points, cycles):
    """

    Parameters
    ----------
    points : numpy.ndarray[M,d]
        M vertices of a graph in d dimensions.

    cycles : numpy.ndarray[k,N]
        Array of indices for k cycles
        of length N.
    """

    distances = np.zeros(len(cycles))

    for i in range(len(cycles)):

        cyclePoints = points[cycles[i]]
        edges = np.concatenate((cyclePoints[1:] - cyclePoints[:-1], [cyclePoints[-1] - cyclePoints[0]]), axis=0)
        distances[i] = np.sum(np.sqrt(np.sum(edges**2)))

    return distances



import numpy as np
import os
import pathlib

import matplotlib.pyplot as plt

# The test datasets should be placed in the same directory this file is in
# (likely `skeletor/data/`)
TEST_DATASET_DIR = str(pathlib.Path(__file__).parent.resolve())

# Dataset names and brief descriptions
TEST_DATASETS_2D = {
                    '2d_curve_1':'Two dimensional high frequnecy periodic-looking wave drawn with a mouse.',
                    '2d_curve_2':'Two dimensional low frequency periodic-looking wave drawn with a mouse.',
                   }

TEST_DATASETS_3D = {
                    'wireframe_cube_1':'Simple wireframe cube with small noise.',
                    'wireframe_cube_2':'Wireframe cube with small noise rotated 45 degrees on one axis.',
                    'wireframe_cube_3':'Wireframe cube with small noise rotated 10 degrees on one axis.',
                    'wireframe_cube_4':'Wireframe cube with small noise rotated 45 degrees on two axes.',
                    'double_wireframe_cube_1':'Small wireframe cube inscribed in a large one.',
                    'double_wireframe_cube_2':'Medium wireframe cube inscribed in a slightly larger one.',
                    'simple_tree':'A basic tree scan from MarcSchotman/skeletons-from-poincloud.',
                   }

TEST_DATASETS_ALL = TEST_DATASETS_2D | TEST_DATASETS_3D

def loadTestDataset(name, downsample=False, randomOrder=False, extraNoise=False):
    """
    Load any of the test point clouds.

    Parameters
    ----------
    name : str
        The name of a test dataset; see `printTestDatasets()`.

    downsample : int or False
        The factor to downsample the data by.

    randomOrder : int or False
        The seed for the random generator to order
        the points, or `False` for the arbitrary order
        the points were saved in.

    extraNoise : float or False
        The scale of noise added to the data in terms of the
        fraction of the total system size, or `False` for no
        noise to be added.

        Any value over approx. `0.25` will result in very
        little information being left from the original data.
    """
    assert name in TEST_DATASETS_ALL.keys(), f'Dataset \'{name}\' not recognized; see \'printTestDatasets()\' for available options.'

    if not downsample:
        dsFactor = 1
    else:
        dsFactor = int(downsample)

    if randomOrder and type(randomOrder) is int:
        np.random.seed(randomOrder)

    with open(os.path.join(TEST_DATASET_DIR, f'{name}.npy'), 'rb') as f:
        rawData = np.load(f)

    if randomOrder:
        order = np.arange(rawData.shape[0])
        np.random.shuffle(order)
        rawData = rawData[order]

    if extraNoise:
        systemScale = np.max(rawData, axis=0) - np.min(rawData, axis=0)
        for i in range(rawData.shape[-1]):
            rawData[:,i] += np.random.uniform(-extraNoise*systemScale[i], extraNoise*systemScale[i], size=len(rawData))

    return rawData[::dsFactor,:]


def printTestDatasets():
    """
    Print out the available test datasets.
    """
    print('**Available datasets**')
    print(f'Data location: {TEST_DATASET_DIR}\n')

    maxNameLength = np.max([len(k) for k in TEST_DATASETS_ALL.keys()])
    print('2D Datasets:')
    for k,v in TEST_DATASETS_2D.items():
        print(f'{k}{"."*(maxNameLength+5-len(k))}{v}')

    print('\n3D Datasets:')
    for k,v in TEST_DATASETS_3D.items():
        print(f'{k}{"."*(maxNameLength+5-len(k))}{v}')

def plotTestDatasets():
    """
    Show scatter plots of all of the available test datasets.
    """
    datasetNames2D = list(TEST_DATASETS_2D.keys()) 
    fig = plt.figure(figsize=(len(datasetNames2D)*3, 4))

    for i in range(len(datasetNames2D)):
        ax = fig.add_subplot(1, len(datasetNames2D), i+1)
        points = loadTestDataset(datasetNames2D[i])
        ax.scatter(*points.T)
        ax.set_title(f'{datasetNames2D[i]}\nPoints: {len(points)}')

    fig.tight_layout()
    plt.show()

    datasetNames3D = list(TEST_DATASETS_3D.keys()) 
    fig = plt.figure(figsize=(len(datasetNames3D)*3, 4))

    for i in range(len(datasetNames3D)):
        ax = fig.add_subplot(1, len(datasetNames3D), i+1, projection='3d')
        points = loadTestDataset(datasetNames3D[i])
        ax.scatter(*points.T)
        ax.set_title(f'{datasetNames3D[i]}\nPoints: {len(points)}')
    
    fig.tight_layout()
    plt.show()


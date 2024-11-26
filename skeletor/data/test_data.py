import numpy as np
import os
import pathlib

import matplotlib.pyplot as plt

# The test datasets should be placed in the same directory this file is in
# (likely `skeletor/data/`)
TEST_DATASET_DIR = str(pathlib.Path(__file__).parent.resolve())
"""
@private
No need to show this variable in the documentation as it will
vary from system to system.
"""

# Dataset names and brief descriptions
TEST_DATASETS_2D = {
                    '2d_curve_1': 'Two dimensional medium frequnecy periodic-looking wave drawn with a mouse.',
                    '2d_curve_2': 'Two dimensional low frequency periodic-looking wave drawn with a mouse.',
                    '2d_curve_3': 'Two dimensional low frequency wave with minor gaps.',
                    '2d_curve_4': 'Slightly undersampled two dimensional low frequency wave.',
                    '2d_curve_5': 'Two tangent, undersampled two dimensional low frequency waves.',
                   }
"""
Dictionary of all 2D datasets.
"""

TEST_DATASETS_3D = {
                    'wireframe_cube_1': 'Simple wireframe cube with small noise.',
                    'wireframe_cube_2': 'Wireframe cube with small noise rotated 45 degrees on one axis.',
                    'wireframe_cube_3': 'Wireframe cube with small noise rotated 10 degrees on one axis.',
                    'wireframe_cube_4': 'Wireframe cube with small noise rotated 45 degrees on two axes.',
                    'double_wireframe_cube_1': 'Small wireframe cube inscribed in a large one.',
                    'double_wireframe_cube_2': 'Medium wireframe cube inscribed in a slightly larger one.',
                    'simple_tree': 'A basic tree scan from MarcSchotman/skeletons-from-poincloud.',
                    'orb_web_scan': 'An orb-weaver web (more or less 2D) scan embedded in 3D.',
                   }
"""
Dictionary of all 3D datasets.
"""

TEST_DATASETS_ALL = TEST_DATASETS_2D | TEST_DATASETS_3D
"""
Dictionary of all datasets.
"""


def loadTestDataset(name, downsample=None, randomOrder=None, extraNoise=None):
    """
    Load any of the test point clouds.

    See `printTestDatasets()` for available point cloud names.

    Parameters
    ----------
    name : str
        The name of a test dataset; see `printTestDatasets()`.

    downsample : int, optional
        The factor to downsample the data by. If not provided,
        no downsampling will be performed.

    randomOrder : int, optional
        The seed for the random generator to order the points. If
        not provided, will be given in the arbitrary order
        the points were saved in.

    extraNoise : float, optional
        The scale of noise added to the data in terms of the
        fraction of the total system size. If not provided, no
        noise will be added.

        Any value over approx. `0.25` will result in very
        little information being left from the original data.
    """
    assert name in TEST_DATASETS_ALL.keys(), \
            f'Dataset \'{name}\' not recognized; see \'printTestDatasets()\' for available options.'

    if downsample is not None:
        dsFactor = int(downsample)
    else:
        dsFactor = 1

    if randomOrder is not None and type(randomOrder) is int:
        np.random.seed(randomOrder)

    with open(os.path.join(TEST_DATASET_DIR, f'{name}.npy'), 'rb') as f:
        rawData = np.load(f)

    if randomOrder is not None:
        order = np.arange(rawData.shape[0])
        np.random.shuffle(order)
        rawData = rawData[order]

    if extraNoise is not None:
        systemScale = np.max(rawData, axis=0) - np.min(rawData, axis=0)
        for i in range(rawData.shape[-1]):
            rawData[:, i] += np.random.uniform(-extraNoise*systemScale[i], extraNoise*systemScale[i], size=len(rawData))

    return rawData[::dsFactor, :]


def printTestDatasets():
    """
    Print out the available test datasets.
    """
    print('**Available datasets**')
    print(f'Data location: {TEST_DATASET_DIR}\n')

    maxNameLength = np.max([len(k) for k in TEST_DATASETS_ALL.keys()])
    print('2D Datasets:')
    for k, v in TEST_DATASETS_2D.items():
        print(f'{k}{"."*(maxNameLength+5-len(k))}{v}')

    print('\n3D Datasets:')
    for k, v in TEST_DATASETS_3D.items():
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

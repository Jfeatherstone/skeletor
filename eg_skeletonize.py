"""
This is an example script on how you might use the library
to analyze a dataset from the command line.

Usage
-----

python3 eg_skeletonize.py <data_path>

Use the --help flag for more information.
"""

import numpy as np

import os

import pickle

import skeletor.skeleton as sk
from skeletor.data import loadPointCloud

import argparse

SKELETONIZATION_METHODS = {'medial': sk.MedialThinningSkeleton,
                           'octree': sk.Octree,
                           'adaptive_octree': sk.AdaptiveOctree,
                           'contraction': sk.LaplacianContractionSkeleton}

# These are kwargs I have manually chosen for a specific dataset
SKELETON_KWARGS = {}

SKELETON_KWARGS['medial'] = {'imageMaxDim': 600,
                             'kernelSize': 5,
                             'minPointsPerPixel': 0.01
                             }

SKELETON_KWARGS['octree'] = {'nBoxes': 10000
                            }

SKELETON_KWARGS['adaptive_octree'] = {'maxPointsPerBox': 0.01
                                     }

SKELETON_KWARGS['contraction'] = {'initialAttraction': 5,
                                  'initialContraction': 1,
                                  'maxContraction': 32,
                                  'contractionAmplificationFactor': 2,
                                  'maxIterations': 20,
                                  }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(dest='inputPath', type=str)

    parser.add_argument('-d', dest='outputDir', type=str, help='Output directory for the data', default='./')
    parser.add_argument('-oe', dest='outputExtension', type=str, help='Extra text to add at the end of the ouput name', default='')
    parser.add_argument('-m', dest='method', type=str, help=f'Method to use for skeletonization; options are: {list(SKELETONIZATION_METHODS.keys())}', default='medial')
    parser.add_argument('--slim', dest='slim', action='store_true', help='Reduce output file size (significantly) by not including original point data.', default=False)

    args = parser.parse_args()

    # Load in the data (most formats are fine; see loadPointCloud for more
    # info
    data = loadPointCloud(args.inputPath)

    assert args.method in list(SKELETONIZATION_METHODS.keys()), \
            f'Invalid skeletonization method provided: {args.method}; see --help for more info.'

    # Call the skeletonization method with the preset kwargs
    skeleton = SKELETONIZATION_METHODS[args.method](data, **SKELETON_KWARGS[args.method])
    skeleton.generateSkeleton()

    # The points can be quite large (~1GB for a large dataset) so we give
    # the option to drop them from the saved file to save space.
    if args.slim:
        skeleton.points = None

    outputName = os.path.join(args.outputDir, [p for p in args.inputPath.split('/') if p][-1] + args.outputExtension + '_skeleton.pckl')
    with open(outputName, 'wb') as f:
        pickle.dump(skeleton, f)

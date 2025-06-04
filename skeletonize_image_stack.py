#!/bin/bin/env/ python3
"""
A script to perform medial axis thinning on a folder of images.

Medial axis thinning is implemented in scikit-image's skeletonize()
function, and images are loaded using dask.

Dask divides matrices up into smaller chunks, so you should be
able to run this script on a moderate computer (~4GB ram), but
it may take a long time.

No command line arguments are taken; parameters must be edited
within the file itself.

Will save the output to a file which is a pickled version of
a sparse COO array.

Note that this script does not use any of the code implemented
in the rest of the library; this was made to be the control
for which the other methods are compared to.
"""
import numpy as np
import matplotlib.pyplot as plt

import os

from pepe.preprocess import checkImageType

from skimage.morphology import skeletonize

from tqdm import tqdm

import dask.array as da
from dask_image.imread import imread
from dask_image.ndmeasure import label
from dask_image.ndfilters import convolve, gaussian_filter

import functools
import operator
import dask.dataframe as dd

import sparse
import pickle

if __name__ == '__main__':

    ###################################
    # Parameters
    # Directory where the pictures are located
    dataFolder = '/bucket/BandiU/Jack/data/scans/2024-10-11_LG_A_PNG/'
    imageExtension = "png"
    # Intensity threshold in images
    threshold = 5
    # Downsample factor on both the images and the image sizes
    dsFactor = 2
    # The conversion info for turning distances in
    # pixels/image stack steps to real units
    # (Set to 1 for no conversion)
    CAMERA_MM_PER_PIXEL = 0.0370
    STAGE_MM_PER_STEP = 0.00625 * 20
  
    # The base scaling factor that every dimension is multiplied by
    baseFactor = 50

    # Smoothing kernel size (odd number)
    # (set to 1 for off)
    smoothKernel = 3
    
    # End parameters
    ###################################

    print('step/pixel:')
    print(CAMERA_MM_PER_PIXEL / STAGE_MM_PER_STEP)

    images = imread(f'{dataFolder}*.{imageExtension}')

    # Grayscale
    images = np.max(images, axis=-1)

    # Remove the mask
    images = images[:-1]

    images = images[::dsFactor,::dsFactor,::dsFactor]


    smoothedImages = gaussian_filter(images, sigma=smoothKernel)

    binImages = smoothedImages > threshold

    dims = images.shape
    xDim, yDim = dims[1], dims[2]

    factor = np.array([baseFactor * CAMERA_MM_PER_PIXEL / STAGE_MM_PER_STEP,
                       baseFactor*xDim/yDim,
                       baseFactor])  # even numbers

    chunksize = np.array(binImages.shape)//factor

    rechunkBinImages = binImages.rechunk(chunksize)

    print(rechunkBinImages)
    print(rechunkBinImages.nbytes)

    skel = da.map_overlap(skeletonize, rechunkBinImages)
    
    sm = sparse.COO(skel.compute())

    with open([p for p in dataFolder.split('/') if p][-1] + '_sparse_skeleton.pckl', 'wb') as f:
        pickle.dump(sm, f)

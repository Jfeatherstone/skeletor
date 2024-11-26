import numpy as np

import os
import cv2

import dask.dataframe as dd
import dask.array as da

from dask_image.imread import imread
from dask_image.ndmeasure import label
from dask_image.ndfilters import convolve, gaussian_filter

import sparse

IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'nef']

def thresholdStatic(image, threshold=0):
    """
    Static thresholding function for an image.

    Parameters
    ----------
    image : array_like
        An image to threshold.

    threshold : float
        The value below which the image pixels will be set to zero.
    """
    image[image <= threshold] = 0
    return array


def loadPointCloud(inputPath, delimiter=',', downsample=None, randomOrder=None, extraNoise=None):
    """
    Load in a point cloud from a file.

    Supports csv, npy, and pcd formats.

    Parameters
    ----------
    inputPath : str
        Path to the input file.

    delimiter : str
        The delimiter if the input file is in csv/txt format.

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

    Returns
    -------
    points : numpy.ndarray[N,d]
        The point cloud.
    """
    inputExt = inputPath.split('.')[-1].lower()

    # Load in the data
    if inputExt == 'csv':
        rawData = np.genfromtxt(inputPath, delimiter=delimiter)
        
    elif inputExt == 'npy':
        with open(inputPath, 'rb') as f:
            rawData = np.load(f)

    elif inputExt == 'pcd':
        rawData = np.asarray(o3d.io.read_point_cloud(inputPath).points)
    
    else:
        raise Exception(f'Unsupported input format: {inputExt}')

    # Downsample, randomize, and add noise (if desired)
    if downsample is not None:
        dsFactor = int(downsample)
    else:
        dsFactor = 1

    if randomOrder is not None and type(randomOrder) is int:
        np.random.seed(randomOrder)

    if randomOrder is not None:
        order = np.arange(rawData.shape[0])
        np.random.shuffle(order)
        rawData = rawData[order]

    if extraNoise is not None:
        systemScale = np.max(rawData, axis=0) - np.min(rawData, axis=0)
        for i in range(rawData.shape[-1]):
            rawData[:, i] += np.random.uniform(-extraNoise*systemScale[i], extraNoise*systemScale[i], size=len(rawData))

    return rawData[::dsFactor, :]


def loadImages(inputPath, format='array', thresholdFunc=None, binarize=False):
    """
    Load in an image or set of images to perform skeletonization on.

    Parameters
    ----------
    inputPath : str
        The path to the file(s) to load in. Possible options include
        either a raster image file (jpg, png, tif, nef), or a directory
        containing raster image files. If a directory is provided, the
        images will be stacked according to alphabetical sorting of
        the filenames.

        If a directory is provided, all images should be the same size.

    format : {'array', 'dask', 'sparse'}
        The format to return the files in.

        `'array'` means that the image or image stack will be returned as
        a regular 2D or 3D `numpy.ndarray`.

        `'dask'` means that the image or image stack will be returned as
        a `dask.array.array`. This is a filetype that allows the user to
        break up a very large array into chunks to perform operations in
        a piecewise manner. Note that this filetype does not load data
        until a calculation is requested, so it will seem much faster to
        load a large dataset, though it is actually just queuing the 
        loading for a later time. For more information, see the dask
        documentation: https://docs.dask.org/en/latest/array.html.

        `'sparse'` means that the image or image stack will be returned
        as a `sparse.COO` matrix. This format only stores a selection of
        values in the matrix as a dictionary, which is beneficial when
        the number of non-zero* elements is much less than the overall
        size of the matrix.

        *In this context, we refer to 'non-zero' elements as elements that
        are relevant to the user; this doesn't necessarily actually mean
        they are everything except what is non-zero. For example, you
        could apply a static threshold on the image intensity, and
        everything below that value is ignored; for more information,
        see the `thresholdFunc` kwarg.

    thresholdFunc : function(array_like) -> array_like(bool)
        A vectorized function that takes in an array_like, and returns a
        mask or thresholded version of that image. Not necessarily
        required when loading images as a numpy or dask array, but highly
        recommended when using the sparse representation (unless your
        image already has lots of zeros in it).

        For example, the most basic approach would be to threshold over a
        static value:

            def func(image):
                image[image <= SOME_CONSTANT] = 0
                return image

            loadImages(path, ..., thresholdFunc=func)

        The function should take no extra args or kwargs; if you have a
        function that requires extra parameters, you can wrap the function
        or use a lambda expression:

            def funcWithKwargs(image, a, b, c):
                ...

            A, B, C = ...
            func = lambda image : funcWithKwargs(image, A, B, C)

            loadImages(path, ..., thresholdFunc=func)

    """
    isSingleImage = False
    isDirectory = False

    # First, we want to get a list of all the image files that we
    # want to load in.

    baseName = os.path.basename(inputPath)
    inputExtension = baseName.split('.')[-1] if '.' in baseName else ''

    # If we are given a single image file, this is quite easy, it is
    # exactly just the file we are given.
    if inputExtension in IMAGE_EXTENSIONS:
        isSingleImage = True
        allImagePaths = [inputPath]

    # Otherwise, if we have a directory, we need to look inside to
    # find all of the image files.
    isDirectory = os.path.isdir(inputPath)
    if isDirectory:
        subfiles = os.listdir(inputPath)
        # First, make sure there is a '.' in the name
        imageSubfiles = np.sort([f for f in subfiles if '.' in f])
        # Now split off the extensions
        imageSubfileExtensions = [f.split('.')[-1] for f in imageSubfiles]
        # Remove empty values
        imageSubfileExtensions = [ext for ext in imageSubfileExtensions if ext]

        isImageFile = [ext in IMAGE_EXTENSIONS for ext in imageSubfileExtensions]

        allImagePaths = [os.path.join(inputPath, img) for img in imageSubfiles[isImageFile]]

    assert isSingleImage or (isDirectory and len(allImagePaths) > 0), \
            f'Invalid file path provided: {inputPath}'

    # Now, allImagePaths contains all of the images files we want to load
    # in, regardless of how inputPath was provided.

    # Dask format
    if format == 'dask':
        # Read in files (or actually, queue the reading of files)
        # Dask supports regex so we don't actually need the list of files
        # we just created.
        if isDirectory:
            image = imread(inputPath + '/*')
        else:
            image = imread(inputPath)

        # Apply our threshold function if provided
        if thresholdFunc is not None:
            image = thresholdFunc(image)

        # Binarize if requested
        if binarize:
            image = image > 0

        return image

    # Numpy format
    if format == 'array':
        # If we don't have a threshold function we define a dummy function to
        # call on each frame
        if thresholdFunc is not None:
            threshold = thresholdFunc
        else:
            def threshold(image):
                return image

        # Need to read in an image first to get the size
        testImage = cv2.imread(allImagePaths[0])
        fullImage = np.zeros((len(allImagePaths), *testImage.shape), dtype=bool if binarize else np.uint8)

        # Read in each image and save it in an array
        for i in range(len(allImagePaths)):
            # If we don't have a threshold function, this is just a dummy
            currImage = threshold(cv2.imread(allImagePaths[i]))

            # Binarize if necessary
            if binarize:
                currImage = currImage > 0

            fullImage[i] = currImage

        # If we only have a single file, we should collapse the first
        # dimension, since probably the user doesn't want it
        if len(allImagePaths) == 1:
            fullImage = fullImage[0]

        return fullImage

    # Sparse format
    if format == 'sparse':
        # Similar approach to numpy loading, but instead of saving
        # the whole images, we just save the location where they are
        # nonzero

        # If we don't have a threshold function we define a dummy function
        # to call on each frame
        if thresholdFunc is not None:
            threshold = thresholdFunc
        else:
            def threshold(image):
                return image

        # Need to read in an image first to get the size
        testImage = cv2.imread(allImagePaths[0])

        # We could need two coordinates to define  point
        # (grayscale image) or three coordinates (colored image)
        coordLength = len(testImage.shape) 

        nonzeroCoords = np.zeros((0,coordLength), dtype=np.int16)
        nonzeroValues = np.zeros(0, dtype=np.uint8)

        # Read in each image and save it in an array
        for i in range(len(allImagePaths)):
            # If we don't have a threshold function, this is just a dummy
            currImage = threshold(cv2.imread(allImagePaths[i]))

            # Binarize if necessary
            if binarize:
                currImage = currImage > 0

            # Find where the image is greater than zero, and save the value
            coords = np.where(currImage > 0)
            values = currImage[coords]
            coords = np.array(coords).T

            nonzeroCoords = np.concatenate((nonzeroCoords, coords))
            nonzeroValues = np.concatenate((nonzeroValues, values))

        fullImage = sparse.COO(coords=nonzeroCoords,
                               values=nonzeroValues,
                               shape=(len(allImagePaths), *testImage.shape), dtype=bool if binarize else np.uint8)

        # If we only have a single file, we should collapse the first
        # dimension, since probably the user doesn't want it
        if len(allImagePaths) == 1:
            fullImage = fullImage[0]

        return fullImage

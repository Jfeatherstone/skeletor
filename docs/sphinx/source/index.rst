.. skeletor documentation master file, created by
   sphinx-quickstart on Thu Nov  7 15:56:25 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

skeletor documentation
======================

``skeletor`` is a library for performing point clouds skeletonization. We implement several state of the art
methods, which we believe to be especially important considering many publications/algorithms don't provide
open source implementations.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    Home <main/overview>
    Uses <main/uses>
    Methods <main/methods>
    Other skeletonization code <main/other_code>

.. toctree::
    :maxdepth: 2
    :caption: API:

    skeletor  <reference/api>
    skeletor.skeleton <reference/skeleton>
    skeletor.spatial <reference/spatial>
    skeletor.utils <reference/utils>
    skeletor.data <reference/data>

The library is divided into a few sub-packages to make navigation easier. Generally, if you are interested in
a particular skeletonization method, I would first read through that page/source code, and refer to the
other sub-packages only as needed.

====================================  ======================================================
Subpackage                            Description
====================================  ======================================================
skeleton                              Skeletonization algorithms
spatial                               Spatial operations on point clouds
utils                                 Utility functions for preprocessing and analysis
data                                  Test data sets and format conversion
====================================  ======================================================

"""
This subpackage contains all of the actual skeletonization methods implemented
in the library.

All of these implementations are objected-oriented, with a base class `SkeletonBase`.

See each individual page for detailed algorithms, how to select parameters, and more.
"""
from .medial_thinning import *
from .basic_octree import *
from .adaptive_octree import *
from .laplacian_contraction import *
from .octree_contraction import *

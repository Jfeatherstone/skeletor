r"""
Functions to compute spatial quantities, like path integrals or
course-grained fields.

.. currentmodule:: skeletor.spatial

.. autosummary::

   courseGrainField
   findDominantHistogramDirections

 Angular Histogram Functions
----------------------------

| Name | Description |
|-|-|
| `courseGrainField <spatial/angular_histogram.html#courseGrainField>`_ (field[, values, defaultValue, ...])  |  Course grain a set of points into a discrete field. |
| `angularHistogramAroundPoint <spatial/angular_histogram.html#angularHistogramAroundPoint>`_ (points, center[, adjArr, ...])  |  Compute a spherical histogram of the directions to neighbor points for a single point. |
| `findDominantHistogramDirections <spatial/angular_histogram.html#findDominantHistogramDirections>`_ (hist, angleAxes[, peakFindPrevalence, ...])  |  Perform peak finding specific to an spherical histogram.  |
| `discretizeDirectionVectors <spatial/angular_histogram.html#discretizeDirectionVectors>`_ (dirVectors[, basisVectors])  | Project a set of vectors on the normals of a cube.  |

 Transform Functions
--------------------

| Name | Description |
|-|-|
| `rotationMatrix <spatial/misc.html#rotationMatrix>`_ (theta, phi, psi)  |  Compute a 3D rotation matrix.  |
| `cartesianToSpherical <spatial/misc.html#cartesianToSpherical>`_ (points)  |  Convert points in cartesian space to generalized spherical coordinates.  |
| `sphericalToCartesian <spatial/misc.html#sphericalToCartesian>`_ (points)  |  Convert points in generalized spherical coordinates to cartesian space.  |

 Misc Functions
---------------

| Name | Description |
|-|-|
| `partitionIntoBoxes(points, nBoxes[, cubes, ...])`                            | Partition a set of points into boxes based on spatial position.                        |
| `pathIntegralAlongField(field, path[, latticeSpacing, ...])`                  | Compute a path integral through a discrete field.                                      |
| `lineIntersection(a1, b1, a2, b2)`                                            | Compute the intersection points of two lines.                                          |
| `calculateAdjacencyMatrix(points, neighborDistance)`                          | Compute the adjacency matrix based on spatial proximity.                               |

"""

from .path_integral import *
from .misc import *
from .angular_histogram import *
from .course_grain import *

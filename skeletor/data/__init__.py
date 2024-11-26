"""
To test the skeletonization methods, we provide several data sets, comprised of either artificial or real data.

Below is a brief description of the available datasets; for more information, see `skeletor.data.printTestDatasets()` or `skeletor.data.plotTestDatasets()`.

2D Datasets
-----------

| Name         | Description                                                                | Points | Source     | Preview |
|--------------|----------------------------------------------------------------------------|--------|------------|---------|
| `2d_curve_1` | Two dimensional medium frequnecy periodic-looking wave drawn with a mouse. | 2935   | Artificial |         |
| `2d_curve_2` | Two dimensional low frequency periodic-looking wave drawn with a mouse.    | 868    | Artificial |         |
| `2d_curve_3` | Two dimensional low frequency wave with minor gaps.                        | 1593   | Artificial |         |
| `2d_curve_4` | Slightly undersampled two dimensional low frequency wave.                  | 772    | Artificial |         |
| `2d_curve_5` | Two tangent, undersampled two dimensional low frequency waves.             | 1569   | Artificial |         |

3D Datasets
-----------

| Name                      | Description                                                     | Points | Source                                | Preview |
|---------------------------|-----------------------------------------------------------------|--------|---------------------------------------|---------|
| `wireframe_cube_1`        | Simple wireframe cube with small noise.                         | 5137   | Artificial                            |         |
| `wireframe_cube_2`        | Wireframe cube with small noise rotated 45 degrees on one axis. | 5137   | Artificial                            |         |
| `wireframe_cube_3`        | Wireframe cube with small noise rotated 10 degrees on one axis. | 5137   | Artificial                            |         |
| `wireframe_cube_4`        | Wireframe cube with small noise rotated 45 degrees on two axes. | 5137   | Artificial                            |         |
| `double_wireframe_cube_1` | Small wireframe cube inscribed in a large one.                  | 10274  | Artificial                            |         |
| `double_wireframe_cube_2` | Medium wireframe cube inscribed in a slightly larger one.       | 10274  | Artificial                            |         |
| `simple_tree`             | A basic tree scan.                                              | 1444   | [MarcSchotman/skeletons-from-poincloud](https://github.com/MarcSchotman/skeletons-from-poincloud) |         |
| `orb_web_scan`            | An orb-weaver web (more or less 2D) scan embedded in 3D.        | 62577  | Light sheet scanning data (original)  |         |

All files are stored as numpy binaries, since this is a nicely compressed format, though data can be converted to other formats using the `convertPointcloud()` method (also available with a CLI interface).

If you want to test a method on all available datasets, you can get the name of every available dataset as the keys of the `skeletor.data.TEST_DATASETS_ALL` dictionary:

    for d in skeletor.data.TEST_DATASETS_ALL.keys():
        points = skeletor.data.loadDataset(d)

        ...

Or similarly you could use `TEST_DATASETS_2D` or `TEST_DATASETS_3D` in the same way.
"""

from .load_data import *
from .test_data import *
from .convert_pc import *

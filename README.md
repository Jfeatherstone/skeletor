# Skeletor

This package implements several different *skeletonization* algorithms for 2D/3D point clouds. Skeletonization is the process of turning an unordered point cloud into a spatially embedded graph.

Note: Skeletonization can mean several other processes, so we clarify that this library implements algorithms to extract 1D graphs from 2D or 3D unordered sets of points.

### Algorithms

- L<sub>1</sub> Medial Axis Thinning [WIP]
    - Originally described in Lee et al. (1994), this process considers a point cloud as a 2D or 3D image, and iteratively removes points close to the edges of the shape until only a single axis is left.
-

### References

Lee, T. C., Kashyap, R. L., & Chu, C. N. (1994). Building Skeleton Models via 3-D Medial Surface Axis Thinning Algorithms. CVGIP: Graphical Models and Image Processing, 56(6), 462–478. https://doi.org/10.1006/cgip.1994.1042


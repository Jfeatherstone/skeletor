# Skeletor

This package implements several different *skeletonization* algorithms for 2D/3D point clouds.

Note: Skeletonization can mean several other processes, so we clarify that this library implements algorithms to extract 1D graphs from 2D or 3D unordered sets of points. Further, this library has an emphasis on point clouds with noisy, wirelike structures (eg. trees, spiderwebs, etc.) though the algorithms can be applied in many contexts.

### Algorithms

Each of the algorithms and the specific implementations here are described more in detail on the respective wiki pages.

- L<sub>1</sub> Medial Axis Thinning
    - Originally described in [1], this process considers a point cloud as a 2D or 3D image, and iteratively removes points close to the edges of the shape until only a single axis is left. This algorithm is actually implemented in [scikit-image](https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html); here we add some preprocessing for applying this method to a point cloud (instead of an image) but otherwise wrap their method.
- Octree Generation
    - This technique, used in the SkelTre algorithm [2], divides the space into a set of boxes, and then tries to establish connectivity criteria between adjacent boxes based on the positions on points therein. We also include an implementation that creates boxes adaptively, allowing non-uniformity in size/placement.
- Laplacian Contraction
    - By assigning points an attraction to both other positions and to their original positions, this technique looks to globally contract points without losing the geometric structure of the point cloud. See [3].

### Related Work

- [hough3d](https://github.com/Jfeatherstone/hough3d): Identify lines in 3D space.
- 

### References

[1] Lee, T. C., Kashyap, R. L., & Chu, C. N. (1994). Building Skeleton Models via 3-D Medial Surface Axis Thinning Algorithms. CVGIP: Graphical Models and Image Processing, 56(6), 462–478. https://doi.org/10.1006/cgip.1994.1042

[2] Bucksch, A., Lindenbergh, R., & Menenti, M. (2012). SkelTre: Robust skeleton extraction from imperfect point clouds. The Visual Computer, 26, 1283–1300. https://doi.org/10.1007/s00371-010-0520-4

[3] Cao, J., Tagliasacchi, A., Olson, M., Zhang, H., & Su, Z. (2010). Point Cloud Skeletons via Laplacian-Based Contraction. Proc. of IEEE Conf. on Shape Modeling and Applications.


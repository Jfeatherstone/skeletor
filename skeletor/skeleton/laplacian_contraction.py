r"""
Implements Laplacian contraction skeletonization method.

Based on Lukas Meyer's implementation here, under the MIT License:
https://github.com/meyerls/pc-skeletor

MIT License

Copyright (c) 2022 Lukas Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import matplotlib.pyplot as plt

import robust_laplacian
import open3d as o3d

import scipy.sparse.linalg as sla
import scipy.sparse as sparse
from scipy.spatial import KDTree

from typing import Union


from .skeleton_base import SkeletonBase

class LaplacianContractionSkeleton(SkeletonBase):

    def __init__(self,
                 points,
                 initialContraction: float = 1,
                 initialAttraction: float = 0.5,
                 maxContraction: float = 2048,
                 maxAttraction: float = 1024,
                 contractionAmplificationFactor: float = 5,
                 tolerance: float = 1e-3,
                 maxIterations: int = 20,
                 verbose: bool = False,
                 debug: bool = False):
        """
        Perform Laplacian-based contraction on a point cloud
        to generate a skeleton.
        """
        # Super class will take care of the basic stuff
        super().__init__(points, verbose, debug)

        # Here we just need to set the parameters specific to this skeletonization
        # method
        self.initialContraction = initialContraction
        self.initialAttraction = initialAttraction
        self.maxContraction = maxContraction
        self.maxAttraction = maxAttraction
        self.contractionAmplificationFactor = contractionAmplificationFactor
        self.tolerance = tolerance
        self.maxIterations = maxIterations

        self.contractedPoints = None

    def _contract(self, points, L, positionalWeights, laplacianWeights):
        """
        Perform least squares sparse solving for the Laplacian-based contraction.

        Returns:
            The contracted point cloud.
        """
        # Define Weights
        WL = sparse.diags(laplacianWeights)  # I * laplacian_weighting
        WH = sparse.diags(positionalWeights)

        A = sparse.vstack([L.dot(WL), WH]).tocsc()
        b = np.vstack([np.zeros((points.shape[0], 3)), WH.dot(points)])

        A_new = A.T @ A

        # Solve each dimension separately
        solvedAxes = [sla.spsolve(A_new, A.T @ b[:,i], permc_spec='COLAMD') for i in range(self.dim)]
        # If we are in 2D, just add back in the previous z dimension (no need to solve it since
        # we will throw it away eventually)
        if self.dim == 2:
            solvedAxes += [list(points[:,2])]
        ret = np.vstack(solvedAxes).T

        if (np.isnan(ret)).all():
            #logging.warn('Matrix is exactly singular. Stopping Contraction.')
            ret = points

        return ret

    def contractPointCloud(self):
        """
        """
        # If we have a 2D point cloud, we should ideally compute things
        # properly in 2D, but the robust_laplacian library only works
        # for 3D point clouds, and I am not capable of implementing
        # my own version of that...
        # So we just project our points into 3D, and clamp the z values
        # at zero at every step.
        if self.dim == 2:
            projectedPoints = np.zeros((len(self.points), 3))
            projectedPoints[:,:2] = self.points
        else:
            projectedPoints = self.points

        # Generate the Laplacian and Mass matrix using the robust_laplacian library.
        # The kwargs are just the default values recommended by the package.
        # L and M will be N x N square matrices where N is the number of points
        L, M = robust_laplacian.point_cloud_laplacian(projectedPoints, mollify_factor=1e-5, n_neighbors=min(30, len(self.points)-1))
        # We will be comparing the diagonal of the mass matrix to previous
        # iterations, so we need to store these
        M_list = [M.diagonal()]

        #plt.spy(L, markersize=1)
        #plt.show()

        # Initial weight matrices
        positionalWeights = self.initialAttraction * np.ones(M.shape[0])
        laplacianWeights = self.initialContraction * 10 ** 3 * np.sqrt(np.mean(M.diagonal())) * np.ones(M.shape[0])

        iteration = 0
        volumeRatio = np.mean(M_list[-1]) / np.mean(M_list[0])
        currPoints = projectedPoints

        while volumeRatio > self.tolerance:
            #pbar.set_description(
            #    "Volume ratio: {}. Contraction weights: {}. Attraction weights: {}. Progress {}".format(
            #        volume_ratio, np.mean(laplacian_weights), np.mean(positional_weights), self.algo_type))
            #logging.debug('Laplacian Weight: {}'.format(laplacian_weights))
            #logging.debug('Mean Positional Weight: {}'.format(np.mean(positional_weights)))

            # Contract the point cloud according to the current Laplacian
            # and weights
            newPoints = self._contract(points=currPoints,
                                       L=L,
                                       positionalWeights=positionalWeights,
                                       laplacianWeights=laplacianWeights)

            # If wthere is no change in the points, end
            if (newPoints == currPoints).all():
                break
            else:
                currPoints = newPoints

                if self.dim == 2:
                    currPoints[:,-1] = 0

            # Add the new diagonal term to the running list
            M_list.append(M.diagonal())
            volumeRatio = np.mean(M_list[-1]) / np.mean(M_list[0])

            # Update laplacian weights with amplification factor
            laplacianWeights *= self.contractionAmplificationFactor
            # Update positional weights with the ratio of the first Mass matrix and the current one.
            positionalWeights = positionalWeights * np.sqrt((M_list[0] / M_list[-1]))

            # Clip weights
            laplacianWeights = np.clip(laplacianWeights, 0.1, self.maxContraction)
            positionalWeights = np.clip(positionalWeights, 0.1, self.maxAttraction)


            # Calculate the new laplacian and mass matrix
            L, M = robust_laplacian.point_cloud_laplacian(currPoints, mollify_factor=1e-5, n_neighbors=min(30, len(self.points)-1))

            iteration += 1
            if iteration == self.maxIterations:
                break

        #plt.spy(L, markersize=1)
        #plt.show()

        # If we have a 2D cloud, we need to clip off the third dimension
        self.contractedPoints = currPoints[:,:self.dim]


    def generateSkeleton(self):
        """
        """
        # Contract the point cloud if this hasn't been done already
        if not hasattr(self.contractedPoints, '__iter__'):
            self.contractPointCloud()

        # Merge points that are very close
        kdTree = KDTree(self.contractedPoints)
        nnDistances, nnIndices = kdTree.query(self.contractedPoints, 2)
        avgNNDistance = np.mean(nnDistances[:,1])

        # Threshold is twice the average nearest neighbor distance
        mergeRadius = 1.0*avgNNDistance
        newPointIdentities = np.arange(len(self.contractedPoints))
        # Find the points within the merge distance for each point
        mergeIndices = kdTree.query_ball_point(self.contractedPoints, mergeRadius)
        for i in range(len(self.contractedPoints)):
            if len(mergeIndices[i]) > 0 and newPointIdentities[i] == i:
                for j in mergeIndices[i]:
                    newPointIdentities[j] = i

        mergedPoints = np.array([np.mean(self.contractedPoints[newPointIdentities == i], axis=0) for i in np.unique(newPointIdentities)])

        if self.verbose:
            print(f'Removed {len(self.contractedPoints) - len(mergedPoints)} points through merging.')
        
        self.skeletonPoints = mergedPoints
        #self.skeletonPoints = self.contractedPoints
        # Establish initial connectivity
        # TODO: Convert to sparse representation
        self.skeletonAdjMat = np.diag(np.ones(len(self.skeletonPoints)))

        return self.skeletonPoints, self.skeletonAdjMat


class LaplacianBasedContractionBase(SkeletonBase):
    """
    Base class for laplacian based algorithms

    """

    def __init__(self,
                 algo_type: str,
                 point_cloud: Union[str, o3d.geometry.PointCloud, dict],
                 init_contraction: int,
                 init_attraction: float,
                 max_contraction: float,
                 max_attraction: int,
                 step_wise_contraction_amplification: Union[float, str],
                 termination_ratio: float,
                 max_iteration_steps: int,
                 debug: bool,
                 verbose: bool,
                 contraction_type):
        super().__init__(verbose, debug)

        if init_attraction == 0 and init_contraction == 0:
            raise ValueError(
                'Both initial parameters (init_attraction and init_contraction) are set to 0. This is an invalid set'
                'of values. e.g. init_attraction=0.5 and init_contraction=1. Exiting.')
        elif init_attraction == 0:
            raise ValueError(
                'The attraction term is set to 0. This leads to a point cloud contracted in the coordinate origin. Exiting.')
        elif init_contraction / (init_attraction + 1e-12) > 1e6 and init_contraction != 0:
            raise ValueError(
                'The ratio between init_attraction and init_contraction is to large! This results in a point cloud '
                'contracted to the coordinate origin. Please set the value more balanced (max. 1e6!). Current ratio:'
                ' {}. Exiting.'.format(init_attraction / init_contraction))

        # Name of the algorithm
        self.algo_type = algo_type

        # Set or load point cloud to apply algorithm
        if isinstance(point_cloud, str):
            self.pcd: o3d.geometry.PointCloud = load_pcd(filename=point_cloud)
        elif isinstance(point_cloud, dict):
            # Currently only two classes are supported!
            if isinstance(point_cloud['trunk'], str):
                self.trunk: o3d.geometry.PointCloud = load_pcd(filename=point_cloud['trunk'])
            else:
                self.trunk: o3d.geometry.PointCloud = point_cloud['trunk']

            if isinstance(point_cloud['branches'], str):
                self.branches: o3d.geometry.PointCloud = load_pcd(filename=point_cloud['branches'])
            else:
                self.branches: o3d.geometry.PointCloud = point_cloud['branches']
        elif isinstance(point_cloud, o3d.geometry.PointCloud):
            self.pcd: o3d.geometry.PointCloud = point_cloud
        else:
            raise TypeError('Type {} is not supported!'.format(type(point_cloud)))

        # Main parameters for the laplacian based contraction
        self.param_termination_ratio = termination_ratio
        self.param_max_contraction = max_contraction
        self.param_max_attraction = max_attraction
        self.param_init_contraction = init_contraction
        self.param_init_attraction = init_attraction
        self.max_iteration_steps = max_iteration_steps

        # Type of contraction. This differs in LBC and S-LBC!
        self.contraction_type = contraction_type
        self.step_wise_contraction_amplification = step_wise_contraction_amplification

        self.graph_k_n = 15

    def set_amplification(self):

        # Set amplification factor of contraction weights.
        if isinstance(self.step_wise_contraction_amplification, str):
            if self.step_wise_contraction_amplification == 'auto':
                num_pcd_points = np.asarray(self.pcd.points).shape[0]

                if num_pcd_points < 1000:
                    contraction_amplification = 1
                    termination_ratio = 0.01
                elif num_pcd_points < 1e4:
                    contraction_amplification = 2
                    termination_ratio = 0.007
                elif num_pcd_points < 1e5:
                    contraction_amplification = 5
                    termination_ratio = 0.005
                elif num_pcd_points < 0.5 * 1e6:
                    contraction_amplification = 5
                    termination_ratio = 0.004
                elif num_pcd_points < 1e6:
                    contraction_amplification = 5
                    termination_ratio = 0.003
                else:
                    contraction_amplification = 8
                    termination_ratio = 0.0005

                self.param_contraction_amplification = contraction_amplification
            else:
                raise ValueError('Value: {} Not found!'.format(self.step_wise_contraction_amplification))
        else:
            self.param_contraction_amplification = self.step_wise_contraction_amplification

    def visualize(self):
        geometry = [self.contracted_point_cloud, self.topology, self.pcd]
        visualize(geometry=geometry, window_name=self.algo_type, point_size=2, line_width=5, background_color=[1, 1, 1])

    def __debug_skeletonization(self, pcd_points):
        import matplotlib
        skeleton = points2pcd(pcd_points)
        L, M = robust_laplacian.point_cloud_laplacian(pcd_points, mollify_factor=1e-5, n_neighbors=30)
        skeleton_vis = copy(skeleton)
        # Mean curvature flow: LP = 2HN
        mean_curvature_flow = L @ pcd_points
        # Scale normals for visualization
        skeleton_vis.normals = o3d.utility.Vector3dVector(5 * mean_curvature_flow)

        area = M.diagonal()
        area = np.clip(area, M.diagonal().mean() - 3 * M.diagonal().std(), M.diagonal().mean() + 3 * M.diagonal().std())

        norm = matplotlib.colors.Normalize(vmin=area.min(), vmax=area.max(), clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.viridis)

        color = list(map(mapper.to_rgba, area))

        skeleton_vis.colors = o3d.utility.Vector3dVector(np.asarray(color)[:, :3])

        self.pcd.normals = o3d.utility.Vector3dVector(np.asarray([[0, 0, 0]]))

        # o3d.visualization.draw_geometries([skeleton_vis], point_show_normal=True)
        o3d.visualization.draw_geometries([self.pcd, skeleton_vis], point_show_normal=True)

    def extract_skeleton(self):
        self.set_amplification()

        pcd_points = np.asarray(self.pcd.points)

        # Log values
        logging.info('PCD #points: {}'.format(pcd_points.shape[0]))
        logging.debug('Init value contraction: {}'.format(self.param_init_contraction))
        logging.debug('Init value attraction: {}'.format(self.param_init_attraction))
        logging.debug('Max contraction value: {}'.format(self.param_max_contraction))
        logging.debug('Max attraction value: {}'.format(self.param_max_attraction))
        logging.debug('Termination ratio: {}'.format(self.param_termination_ratio))
        logging.debug('Amplification factor: {}'.format(self.param_contraction_amplification))

        L, M = robust_laplacian.point_cloud_laplacian(pcd_points, mollify_factor=1e-5, n_neighbors=30)
        M_list = [M.diagonal()]

        # Init weights
        positional_weights = self.param_init_attraction * np.ones(M.shape[0])
        laplacian_weights = self.param_init_contraction * 10 ** 3 * np.sqrt(np.mean(M.diagonal())) * np.ones(M.shape[0])

        if self.debug:
            self.__debug_skeletonization(pcd_points)

        iteration = 0
        volume_ratio = np.mean(M_list[-1]) / np.mean(M_list[0])
        pbar = tqdm(total=self.max_iteration_steps)

        pcd_points_current = pcd_points
        while np.mean(M_list[-1]) / np.mean(M_list[0]) > self.param_termination_ratio:
            pbar.set_description(
                "Volume ratio: {}. Contraction weights: {}. Attraction weights: {}. Progress {}".format(
                    volume_ratio, np.mean(laplacian_weights), np.mean(positional_weights), self.algo_type))
            logging.debug('Laplacian Weight: {}'.format(laplacian_weights))
            logging.debug('Mean Positional Weight: {}'.format(np.mean(positional_weights)))

            pcd_points_new = self.contraction_type(pcd_points=pcd_points_current,
                                                   L=L,
                                                   laplacian_weighting=laplacian_weights,
                                                   positional_weighting=positional_weights)

            if (pcd_points_new == pcd_points_current).all():
                break
            else:
                pcd_points_current = pcd_points_new

            if self.debug:
                self.__debug_skeletonization(pcd_points_current)

            volume_ratio = np.mean(M_list[-1]) / np.mean(M_list[0])

            # Update laplacian weights with amplification factor
            laplacian_weights *= self.param_contraction_amplification
            # Update positional weights with the ration of the first Mass matrix and the current one.
            positional_weights = positional_weights * np.sqrt((M_list[0] / M.diagonal()))

            # Clip weights
            laplacian_weights = np.clip(laplacian_weights, 0.1, self.param_max_contraction)
            positional_weights = np.clip(positional_weights, 0.1, self.param_max_attraction)

            M_list.append(M.diagonal())

            iteration += 1
            pbar.update(1)

            try:
                L, M = robust_laplacian.point_cloud_laplacian(pcd_points_current, mollify_factor=1e-5, n_neighbors=30)
            except RuntimeError as er:
                print(er)
                break

            if iteration == self.max_iteration_steps:
                break

        print('Contraction is Done.')
        logging.info('Contraction is Done.')

        self.contracted_point_cloud = points2pcd(pcd_points_current)

        return pcd_points_current

    def __extract_skeletal_graph(self, skeletal_points: np.ndarray):
        def extract_mst(points: np.ndarray):
            mst = mist.GetMST(x=points[:, 0], y=points[:, 1], z=points[:, 2])
            degree, edge_length, branch_length, branch_shape, edge_index, branch_index = mst.get_stats(
                include_index=True, k_neighbours=self.graph_k_n)

            return degree, edge_length, branch_length, branch_shape, edge_index, branch_index

        _, _, _, _, edge_index, _ = extract_mst(points=skeletal_points)

        # Convert to Graph
        mst_graph = nx.Graph(edge_index.T.tolist())
        for idx in range(mst_graph.number_of_nodes()):
            mst_graph.nodes[idx]['pos'] = skeletal_points[idx].T

        return mst_graph

    def __simplify_graph(self, graph):

        G_simplified, node_pos, node_idx = simplify_graph(graph)
        skeleton_cleaned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(node_pos)))
        skeleton_cleaned.paint_uniform_color([0, 0, 1])
        skeleton_cleaned_points = np.asarray(skeleton_cleaned.points)

        mapping = {}
        for node in G_simplified:
            pcd_idx = np.where(skeleton_cleaned_points == G_simplified.nodes[node]['pos'])[0][0]
            mapping.update({node: pcd_idx})

        return nx.relabel_nodes(G_simplified, mapping), skeleton_cleaned_points

    def extract_topology(self):
        contracted_point_cloud_zero_artifact = deepcopy(self.contracted_point_cloud)

        # Artifacts at zero
        pcd_contracted_tree = o3d.geometry.KDTreeFlann(self.contracted_point_cloud)
        idx_near_zero = np.argmin(np.linalg.norm(np.asarray(contracted_point_cloud_zero_artifact.points), axis=1))
        if np.linalg.norm(contracted_point_cloud_zero_artifact.points[idx_near_zero]) <= 0.01:
            [k, idx, _] = pcd_contracted_tree.search_radius_vector_3d(
                contracted_point_cloud_zero_artifact.points[idx_near_zero], 0.01)
            self.contracted_point_cloud = contracted_point_cloud_zero_artifact.select_by_index(idx, invert=True)
        contracted_point_cloud = np.asarray(self.contracted_point_cloud.points)

        # Compute points for farthest point sampling
        self.fps_points = int(contracted_point_cloud.shape[0] * 0.1)
        self.fps_points = max(self.fps_points, 15)

        # Sample with farthest point sampling
        self.skeleton = self.contracted_point_cloud.farthest_point_down_sample(num_samples=self.fps_points)
        skeleton_points = np.asarray(self.skeleton.points)

        if (np.isnan(contracted_point_cloud)).all():
            print('Element is NaN!')

        self.skeleton_graph = self.__extract_skeletal_graph(skeletal_points=skeleton_points)
        self.topology_graph, topology_points = self.__simplify_graph(graph=self.skeleton_graph)

        self.topology = o3d.geometry.LineSet()
        self.topology.points = o3d.utility.Vector3dVector(topology_points)
        self.topology.lines = o3d.utility.Vector2iVector(list((self.topology_graph.edges())))

        return self.topology

    def export_results(self, output: str):
        os.makedirs(output, exist_ok=True)
        path_contracted_pcd = os.path.join(output, '01_point_cloud_contracted_{}'.format(self.algo_type) + '.ply')
        o3d.io.write_point_cloud(path_contracted_pcd, self.contracted_point_cloud)

        path_skeleton = os.path.join(output, '02_skeleton_{}'.format(self.algo_type) + '.ply')
        o3d.io.write_point_cloud(filename=path_skeleton, pointcloud=self.skeleton)

        path_topology = os.path.join(output, '03_topology_{}'.format(self.algo_type) + '.ply')
        o3d.io.write_line_set(filename=path_topology, line_set=self.topology)

        path_skeleton_graph = os.path.join(output, '04_skeleton_graph_{}'.format(self.algo_type) + '.gpickle')
        nx.write_gpickle(G=self.skeleton_graph, path=path_skeleton_graph)

        path_topology_graph = os.path.join(output, '05_topology_graph_{}'.format(self.algo_type) + '.gpickle')
        nx.write_gpickle(G=self.topology_graph, path=path_topology_graph)


class LBC(LaplacianBasedContractionBase):
    """
    Laplacian-Based Contraction (LBC)
    ---------------------------------

    Original implementation of Cao et al.

    Paper: https://taiya.github.io/pubs/cao2010cloudcontr.pdf
    Original Matlab Code: https://github.com/taiya/cloudcontr
    """

    def __init__(self,
                 point_cloud: Union[str, o3d.geometry.PointCloud],
                 init_contraction: float = 1.,
                 init_attraction: float = 0.5,
                 max_contraction: int = 2048,
                 max_attraction: int = 1024,
                 step_wise_contraction_amplification: Union[float, str] = 'auto',
                 termination_ratio: float = 0.003,
                 max_iteration_steps: int = 20,
                 down_sample: float = -1,
                 filter_nb_neighbors: int = 20,
                 filter_std_ratio: float = 2.0,
                 debug: bool = False,
                 verbose: bool = False):
        super().__init__(self.__class__.__name__,
                         point_cloud,
                         init_contraction,
                         init_attraction,
                         max_contraction,
                         max_attraction,
                         step_wise_contraction_amplification,
                         termination_ratio,
                         max_iteration_steps,
                         debug,
                         verbose,
                         self.__least_squares_sparse)

        # Down sampling point cloud for faster contraction.
        if down_sample != -1:
            self.pcd = self.pcd.voxel_down_sample(down_sample)

        # Filter point cloud as outliers might distort the skeletonization algorithm
        if filter_nb_neighbors and filter_std_ratio:
            self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors=filter_nb_neighbors,
                                                              std_ratio=filter_std_ratio)

        if self.debug:
            o3d.visualization.draw_geometries([self.pcd], window_name="Default Point Cloud")

    def __least_squares_sparse(self, pcd_points, L, laplacian_weighting, positional_weighting):
        """
        Perform least squares sparse solving for the Laplacian-based contraction.

        Args:
            pcd_points: The input point cloud points.
            L: The Laplacian matrix.
            laplacian_weighting: The Laplacian weighting matrix.
            positional_weighting: The positional weighting matrix.

        Returns:
            The contracted point cloud.
        """
        # Define Weights
        WL = sparse.diags(laplacian_weighting)  # I * laplacian_weighting
        WH = sparse.diags(positional_weighting)

        if self.debug:
            plt.figure(figsize=(10, 10))

            plt.spy(L, ms=1)
            plt.title('Laplacian Matrix', fontsize=40)
            plt.show()

        A = sparse.vstack([L.dot(WL), WH]).tocsc()
        b = np.vstack([np.zeros((pcd_points.shape[0], 3)), WH.dot(pcd_points)])

        A_new = A.T @ A

        if self.verbose:
            plt.spy(A_new, ms=0.1)
            plt.title('A_new: A.T @ A')
            plt.show()

        x = sla.spsolve(A_new, A.T @ b[:, 0], permc_spec='COLAMD')
        y = sla.spsolve(A_new, A.T @ b[:, 1], permc_spec='COLAMD')
        z = sla.spsolve(A_new, A.T @ b[:, 2], permc_spec='COLAMD')

        ret = np.vstack([x, y, z]).T

        if (np.isnan(ret)).all():
            logging.warn('Matrix is exactly singular. Stopping Contraction.')
            ret = pcd_points

        if self.debug:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            mean_curvature_flow = L @ pcd_points
            # Scale normals for visualization
            pcd.normals = o3d.utility.Vector3dVector(mean_curvature_flow / 5)

            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        return ret


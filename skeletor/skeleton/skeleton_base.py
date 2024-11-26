"""
Base class for skeletonization methods.

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

from typing import Union
import logging
import os

import open3d as o3d
import networkx as nx

from skeletor.utils import plotSpatialGraph

import sparse

class SkeletonBase(object):

    #############################
    ## Variables to be defined
    #############################
    # The following variables should be defined by the inheriting
    # class for this base class to function properly.

    points = np.zeros((0,0))
    """
    numpy.ndarray[N,d]

    Set of points comprising the point cloud.
    """
    
    skeletonPoints = np.zeros((0, 0))
    """
    numpy.ndarray[M,d]

    Set of points comprising the skeleton.
    """
    
    # Adjacency matrix noting which points of the
    # skeleton are connected to which others
    # sparse.COO[M,M]
    skeletonAdjMat = sparse.COO([], shape=(0, 0))
    """
    sparse.COO[M,M]

    Adjacency matrix for the skeleton. Stored as a sparse matrix
    since there may be a potential very large amount of skeleton points.
    Can always be converted to a dense matrix with:

        skeletonAdjMat.todense()
    """

    skeletonGraph = nx.Graph()
    """
    networkx.Graph()

    Graph representation of the skeleton.
    """

    topologyPoints = np.zeros((0, 0))
    """
    numpy.ndarray[L,d]

    Points for the topological graph, ie. graph with all degree 2 vertices
    removed and their neighbors joined.

    Computed by this base class's function `generateTopology()`.
    """

    topologyAdjMat = sparse.COO([], shape=(0, 0))
    """
    sparse.COO[M,M]

    Adjacency matrix for the topological graph. Stored as a sparse matrix.
    """

    topologyGraph = nx.Graph()
    """
    networkx.Graph()

    Graph representation of the topological graph.
    """

    def __init__(self,
                 points: Union[np.ndarray, None] = None,
                 verbose: bool = False,
                 debug: bool = False):
        r"""
        The base class for a skeletonization algorithm, which implements
        methods to plot the skeleton and generate images/movies.

        Parameters
        ----------
        points : numpy.ndarray[N,d]
            Point cloud.

        verbose : bool
            Whether or not to print information during generation.

        debug : bool
            Whether or not to plot debug figures during generation.

        """

        self.verbose: bool = verbose
        self.debug: bool = debug

        # The only real benefit of giving the points to this
        # template class is to note the dimension.
        if hasattr(points, '__iter__'):
            self.points = points
            self.dim = np.shape(points)[-1]
        else:
            self.points = None
            self.dim = None

        # TODO: Update this
        #if self.verbose:
        #    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
        #else:
        #    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



    def generateTopology(self,
                         keepSpatialEmbedding: bool = False):
        """
        Create a copy of the skeleton graph with all vertices of
        degree 2 removed.

        The spatial embedding can be carried over as well, but likely
        doesn't mean very much anymore.

        Parameters
        ----------
        keepSpatialEmbedding : bool
            Whether to carry the spatial embedding from the skeleton
            over to the topology graph (True) or to leave the
            topology graph unembedded (False).

        Returns
        -------
        topology : networkx.Graph
            Graph representing topology of the skeleton.

            Also saved as self.topologyGraph
        """
        # TODO: Remove all degree 2 nodes and then make a networkx graph
        pass


    def save(self, output):
        """
        Not implemented! 

        Save the resulting skeleton in to a file. Can be written

        Parameters
        ----------
        output : str
            Output file path, including filename. If no
            extension is given, `.ply` will be used.

        Returns
        -------
        None
        """
        # TODO
        pass


    def plot(self,
             plotPoints: bool = True,
             plotSkeleton: bool = True,
             backend: str = 'mpl',
             ax=None,
             **kwargs):
        """
        Plot the point cloud, skeleton, or other information.

        Parameters
        ----------
        plotPoints : bool
            Whether to plot the point cloud.

        plotSkeleton : bool
            Whether to plot the derived skeleton nodes and edges.

        backend : {'mpl', 'o3d'}
            Which plotting library to use: matplotlib (`'mpl'`) or
            Open3D (`'o3d'`).

            Open3D is recommended for larger point clouds, as it renders
            much faster than matplotlib.

        ax : matplotlib.pyplot.axis, optional
            The axis on which to plot the requested information, if
            using the matplotlib backend. If not provided, a new axis
            will be created.

        kwargs : kwargs
            Other keyword arguments specific to a given skeletonization
            subclass. See individual documentation on
            `_o3d_extra_plot_geometries()` or `_mpl_extra_plot_geometries()`
            for each subclass for more information.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Figure object, if using matplotlib backend.

        If using open3d, nothing will be returned.
        """
        # 2D plotting always uses matplotlib
        if self.dim == 2:

            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot()
            else:
                fig = plt.gcf()

            if plotSkeleton and len(self.skeletonPoints) > 0:
                plotSpatialGraph(self.skeletonPoints, self.skeletonAdjMat, ax=ax, scatterKwargs={"c": 'tab:blue', "alpha": 0.6})

            if plotPoints:
                ax.scatter(*self.points.T, alpha=0.05, c='tab:purple')

            # Plot extra stuff if necessary
            self._mpl_extra_plot_geometries(plotPoints, plotSkeleton, ax, **kwargs)

            return fig

        # For 3D we can choose between mpl and o3d
        elif self.dim == 3:

            if backend == 'mpl':
                if ax is None:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                else:
                    fig = plt.gcf()

                if plotSkeleton and len(self.skeletonPoints) > 0:
                    plotSpatialGraph(self.skeletonPoints, self.skeletonAdjMat, ax=ax, scatterKwargs={"c": 'tab:blue', "alpha": 0.8})

                if plotPoints:
                    ax.scatter(*self.points.T, alpha=0.05, c='tab:purple')

                # Plot extra stuff if necessary
                self._mpl_extra_plot_geometries(plotPoints, plotSkeleton, ax, **kwargs)

                return fig

            elif backend == 'o3d':
                geometry = []

                if plotSkeleton and len(self.skeletonPoints) > 0:
                    skelPointCloud = o3d.geometry.PointCloud()
                    skelPointCloud.points = o3d.utility.Vector3dVector(self.skeletonPoints)
                    skelPointCloud.paint_uniform_color(np.array([31, 119, 180])/255.) # Tab:blue color
                    geometry.append(skelPointCloud)

                    # TODO: Allow support for weighted edges (maybe use color?)
                    skelEdges = o3d.geometry.LineSet()
                    skelEdges.points = o3d.utility.Vector3dVector(self.skeletonPoints)

                    # Find all indices in the adjacency matrix that are greater than 0
                    # Code for numpy
                    pairConnections = np.array(np.where(self.skeletonAdjMat > 0), dtype=np.int64).T
                    # Code for sparse
                    #pairConnections = self.skeletonAdjMat.coords.T
                    # Sort them, so that we can remove duplicates
                    pairConnections = [np.sort(p) for p in pairConnections]
                    # And remove duplicates
                    pairConnections = np.unique(pairConnections, axis=0)

                    skelEdges.lines = o3d.utility.Vector2iVector(pairConnections)

                    geometry.append(skelEdges)

                if plotPoints:
                    pointCloud = o3d.geometry.PointCloud()
                    pointCloud.points = o3d.utility.Vector3dVector(self.points)
                    pointCloud.paint_uniform_color(np.array([148, 103, 189])/255.) # Tab:purple color
                    geometry.append(pointCloud)

                # If any of the implementations of this base class want to
                # add extra elements to plot, we have support for that.
                # For example, if an octree algorithm wants to plot the
                # boxes into which points are divided
                geometry = geometry + self._o3d_extra_plot_geometries(plotPoints, plotSkeleton, **kwargs)

                o3d.visualization.draw_geometries(geometry)


            else:
                print(f'Unrecognized backend ({backend}), please use either \'mpl\' or \'o3d\'.')


    def _o3d_extra_plot_geometries(self, plotPoints, plotSkeleton, **kwargs):
        """
        Extra geometries to be plotted by the plot() function.

        To be implemented in the inheriting class if desired.
        In this case, should return a list of Open3D geometries.

        Returns
        -------
        list(open3d.geometry)
        """

        return []


    def _mpl_extra_plot_geometries(self, plotPoints, plotSkeleton, ax, **kwargs):
        """
        Extra geometries to be plotted by the plot() function.

        To be implemented in the inheriting class if desired.
        In this case, use the given axis `ax` to plot desired quantities.

        Returns
        -------
        None
        """

        return None


    def animate(self,
                outputFile=None,
                plotPoints=True,
                plotSkeleton=True,
                rotationSpeed=5,
                maxFrames=None,
                fps=30, pointSize=7,
                lineWidth=5,
                zoom=0.8,
                initialRotation=(0, -520),
                backgroundColor=[1,1,1],
                crop=False,
                cropPadding=10,
                outputFormat='gif',
                **kwargs):
        """
        Create an animation of the point cloud and/or skeleton rotating.

        Uses Open3D as the graphical framework.

        Parameters
        ----------
        outputFile : str, optional
            The name of the animation file that will be saved. If it includes
            an (acceptable) extension, the output type will be automatically
            chosen; otherwise, the output format will be set by the
            kwarg `outputFormat`.

            If `None`, no file will be saved, the animation will just be
            played.

        plotPoints : bool
            Whether to plot the point cloud.

        plotSkeleton : bool
            Whether to plot the derived skeleton nodes and edges.

        rotationSpeed : int
            The speed of the rotation during the animation.

        maxFrames : int, optional
            The total number of frames to save; if `None`, no limit will
            be applied.

            Note that I can't figure out how to exit a window automatically,
            so the window itself will continue existing until the user manually
            presses 'q', but no more than `maxFrames` will actually be saved.

        fps : int
            The frames per second of the output animation.

        pointSize : float
            The size of points for point clouds.

        lineWidth : float
            The width of lines for line sets (WIP).

        zoom : float
            The zoom setting for the view.

        initialRotation : (theta, phi)
            Initial rotation angles. This is based off of
            `open3d.visualization.ViewControl.rotate` method, which
            takes the rotation values in pixels your cursor has
            moved across the screen... not ideal.

            Roughly 520 pixels gives you a 90 degree rotation.

        backgroundColor : [float, float, float]
            The background color for the animation.

        crop : bool, optional
            Whether to crop the animation to a minimal window size (plus
            padding) that only includes pixels that actually change
            during the animation.

        cropPadding : int
            The amount of padding to add around the animation if
            `crop=True`.

        outputFormat : 'gif' or 'mp4'
            The output format for the animation. Overridden if an extension is
            directly included in the file name (`outputFile`).
            
        """

        if self.dim == 2:
            raise Exception('Animation not supported for 2D point clouds')

        # Import these here since they arent needed in the rest of this
        # file and this is sortof a debug method
        import cv2
        from PIL import Image

        #############################################
        # Create our geometries
        #############################################
        geometry = []

        if plotSkeleton and len(self.skeletonPoints) > 0:
            skelPointCloud = o3d.geometry.PointCloud()
            skelPointCloud.points = o3d.utility.Vector3dVector(self.skeletonPoints)
            skelPointCloud.paint_uniform_color(np.array([31, 119, 180])/255.) # Tab:blue color
            geometry.append(skelPointCloud)

            # TODO: Allow support for weighted edges (maybe use color?)
            skelEdges = o3d.geometry.LineSet()
            skelEdges.points = o3d.utility.Vector3dVector(self.skeletonPoints)

            # Find all indices in the adjacency matrix that are greater than 0
            # Code for numpy
            pairConnections = np.array(np.where(self.skeletonAdjMat > 0), dtype=np.int64).T
            # Code for sparse
            #pairConnections = self.skeletonAdjMat.coords.T
            # Sort them, so that we can remove duplicates
            pairConnections = [np.sort(p) for p in pairConnections]
            # And remove duplicates
            pairConnections = np.unique(pairConnections, axis=0)

            skelEdges.lines = o3d.utility.Vector2iVector(pairConnections)

            geometry.append(skelEdges)

        if plotPoints:
            pointCloud = o3d.geometry.PointCloud()
            pointCloud.points = o3d.utility.Vector3dVector(self.points)
            pointCloud.paint_uniform_color(np.array([148, 103, 189])/255.) # Tab:purple color
            geometry.append(pointCloud)

        # If any of the implementations of this base class want to
        # add extra elements to plot, we have support for that.
        # For example, if an octree algorithm wants to plot the
        # boxes into which points are divided
        geometry = geometry + self._o3d_extra_plot_geometries(plotPoints, plotSkeleton, **kwargs)

        # Correct for rotation
        # TODO: Find a better way to do this (can I rotate the view?)
        #for i in range(len(geometry)):
        #    geometry[i].points = o3d.utility.Vector3dVector(np.asarray(geometry[i].points)[[0,2,1]])

        #############################################
        # Setup the animation
        #############################################
        ALLOWED_FORMATS = ['gif', 'mp4']

        if outputFile is not None:
            # Try to infer the output format from the file name
            potentialExtension = outputFile.split('.')[-1].lower()
            if potentialExtension in ALLOWED_FORMATS:
                outputFormat = potentialExtension
                fullOutputPath = outputFile
            else:
                # Otherwise, we have to append whatever the provided extension is
                outputFormat = outputFormat.lower()

                # But first make sure the provided format is allowed
                if outputFormat not in ALLOWED_FORMATS:
                    raise Exception(f'Invalid output format specified: {outputFormat}; available options are {ALLOWED_FORMATS}.')

                fullOutputPath = f'{outputFile}.{outputFormat}'

        images = []
        global initialRotationApplied
        initialRotationApplied = False

        def update_view(vis):
            global initialRotationApplied

            opt = vis.get_render_option()
            opt.background_color = np.asarray(backgroundColor)

            vis.get_render_option().point_size = pointSize
            vis.get_render_option().line_width = lineWidth

            ctr = vis.get_view_control()

            if not initialRotationApplied:
                ctr.rotate(*initialRotation)
                initialRotationApplied = True

            ctr.rotate(rotationSpeed, 0)
            ctr.set_zoom(zoom)

            if outputFile is not None and (maxFrames is None or len(images) < maxFrames):
                image = (np.array(vis.capture_screen_float_buffer(False))*255).astype(np.uint8)
                images.append((image).astype(np.uint8))
            
            return False

        o3d.visualization.draw_geometries_with_animation_callback(geometry,
                                                                  update_view)

        if outputFile is None:
            return


        if crop:
            # If we want to crop, we should see if there are any pixels that only
            # ever take on the value of the background color
            usedPixels = np.sum(np.array(images) - np.array(backgroundColor)*255, axis=0)
            usedPixels = np.sum(usedPixels**2, axis=-1)

            usedIndices = np.array(np.where(usedPixels > 1e-8))
            cropBounds = [max(np.min(usedIndices[0])-cropPadding, 0),
                          min(np.max(usedIndices[0])+cropPadding, images[0].shape[0]),
                          max(np.min(usedIndices[1])-cropPadding, 0),
                          min(np.max(usedIndices[1])+cropPadding, images[0].shape[1])]

        else:
            cropBounds = [0, images[0].shape[0], 0, images[0].shape[1]]

        for i in range(len(images)):
            images[i] = images[i][cropBounds[0]:cropBounds[1],cropBounds[2]:cropBounds[3]]

        if outputFormat == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(fullOutputPath, fourcc, fps, (images[0].shape[1], images[0].shape[0]))

            for image in images:
                video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            video.release()

        if outputFormat == 'gif':
            for i in range(len(images)):
                images[i] = Image.fromarray(images[i])

            images[0].save(fullOutputPath, save_all=True, append_images=images[1:], loop=0, duration=int(1/fps*1e3))

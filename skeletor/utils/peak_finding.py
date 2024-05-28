"""
Code taken from https://github.com/Jfeatherstone/pepe
"""

import numpy as np
import matplotlib.pyplot as plt

import numba
import colour

from scipy.signal import convolve

from pepe.topology import spatialClusterCenters, spatialClusterLabels

def findPeaks2D(data, minPeakPrevalence=None, normalizePrevalence=True):
    """
    Identify peaks in 2-dimensional data using persistent topology.
    Peak prevalence is calculated as the age of topological features,
    which is not necessarily the same as the actual height of the peaks.

    Originally adapted from [1] (and partially from [2]), but adjusted for data with
    an specifically 2 dimensions. For background on persistence, see: [3] for
    application of persistence to signal processing, or [4] for general domain knowledge
    (these are just my personal recommendations, probably plenty of other resources out there).
    May struggle to find peaks in noisier data if the fluctuations due to
    noise are large enough to create a spiky-looking profile. This is because
    such fluctuations do represent topological objects, even if they are not ones
    of any physical relevance; this method does not distinguish between the two.
    In this case, smoothing functions are recommended to be used on the data before
    it is passed to this method.

    Method is optimized using `numba`.

    Parameters
    ----------
    data : np.ndarray[d]
        An array of data points within which to identify peaks in d-dimensional space.

    minPeakPrevalence : float or None
        The minimum prevalence of a peak (maximum - minimum) that will be
        returned to the user. If `normalizePrevalence` is True, this should
        be in the range `[0, 1]`, representing a percentage of the full range
        the data occupies.

    normalizePrevalence : bool
        Whether to normalize the prevalences such that they represent the
        percent of the data range that is spanned by a peak (True) or not (False).
        For example, if normalized, a prevalence of .4 would mean that the peak
        spans 40% of the total range of the data.

        Note that this normalization happens before using the minPeakPrevalence
        kwarg to clip smaller peaks, and thus the minimum value should be
        specified as a percent (eg. .3 for 30%) if the normalizePrevalence
        kwarg is True.

    Returns
    -------
    peakPositions : np.ndarray[N,d]
        The indices of peaks in the provided data, sorted from most
        to least prevalent (persistent).

    peakPrevalences : np.ndarray[N]
        The prevalence of each peak in the provided data, or the persistence
        of the topological feature. If `normalizePrevalence` is True, this will
        be normalized to the domain `[0, 1]`; otherwise, these values are
        equivalent to the maximum data value minus the minimum data value evaluated
        over all points that identify with a given peak.

    References
    ----------
    [1] Huber, Stefan. Persistent Topology for Peak Detection. 
    <https://www.sthu.org/blog/13-perstopology-peakdetection/index.html>

    [2] Huber, Stefan. Topological peak detection in two-dimensional data.
    <https://www.sthu.org/code/codesnippets/imagepers.html>

    [3] Huber, S. (2021). Persistent Homology in Data Science. In P. Haber,
    T. Lampoltshammer, M. Mayr, & K. Plankensteiner (Eds.), Data Science – Analytics
    and Applications (pp. 81–88). Springer Fachmedien. <https://doi.org/10.1007/978-3-658-32182-6_13>

    [4] Edelsbrunner, H., & Harer, J. (2010). Computational topology: An introduction.
    Chapter 7: Persistence. p. 149-156. American Mathematical Society. ISBN: 978-0-8218-4925-5
    """ 
    # Go through each value in our data, starting from highest and ending
    # with lowest. This is important because we will be merging points into
    # groups as we go, and so starting with the highest values means that
    # points will almost never have to be overwritten.
    # This is a bit complex here because we could have any number of dimensions, but
    # the gist is that we sort a flattened version of the array from highest to lowest,
    # then turn those 1d indices into Nd indices, then pair the indices together.
    # Each element of this array will be a set of indices that can index a single
    # element of data. 
    # eg. data[sortedIndices[0]] will be the largest value (for any number of dimensions)
    sortedIndices = np.dstack(np.unravel_index(np.argsort(data.flatten())[::-1], data.shape))[0]
    # To be able to actually index data with an element, they need to all be tuples
    sortedIndices = [tuple(si) for si in sortedIndices]
  
    # Numba gets mad if we pass lists between methods, so we have to turn our list
    # into a numba (typed) list
    typedSortedIndices = numba.typed.List(sortedIndices)

    peakMembership, peakBirthIndices = _findPeaks2DIter(data, typedSortedIndices)

    # Calculate the prevalence of each peak as the height of the birth points minus the
    # height of the death point
    # We could do this using the arrays we've stored along the way, but it's easier just to take the
    # max/min heights directly from the data
    peakPrevalences = np.array([data[peakBirthIndices[i]] - np.min(data[peakMembership == i]) for i in range(len(peakBirthIndices))])
    # Also note that I have made the decision here to normalize these prevalences by the total range
    # of the data, such that a prevalence of .6 means that the peak spans 60% of the range of the data.
    # This can be altered with the normalizePrevalence kwarg
    if normalizePrevalence:
        dataRange = np.max(data) - np.min(data)
        peakPrevalences /= dataRange

    # Cut off small peaks, if necessary
    if minPeakPrevalence is not None:
        peakBirthIndices = [peakBirthIndices[i] for i in range(len(peakBirthIndices)) if peakPrevalences[i] > minPeakPrevalence]
        peakPrevalences = np.array([peakPrevalences[i] for i in range(len(peakPrevalences)) if peakPrevalences[i] > minPeakPrevalence])

    # Sort the peaks by their prevalence
    #order = np.argsort(peakPrevalences)[::-1]
    #peakPositions = peakPositions[order]
    #peakPrevalences = peakPrevalences[order]

    return (peakBirthIndices, peakPrevalences)


@numba.njit(cache=False)
def _findPeaks2DIter(data, sortedIndices):
    """
    Iterative part of 2D peak finding, optimized using `numba`.
    As usual with these types of methods, there may be some statements
    that generally would be written better/easier another way, but end up
    looking weird because of a particular `numba` requirement.
    Not meant to be used outside of `pepe.topology.findPeaks2D()`.
    """
    # This array contains the indices of the peak that each point
    # belongs to (assuming it does belong to a peak)
    # We start it at -1 such taht we can check if a point belongs to
    # a certain peak with peakMembership[i,j,...] >= 0
    peakMembership = np.zeros_like(data, dtype=np.int16) - 1
    peakBirthIndices = []

    # I've avoided using the index i here since we are iterating over sets of indices,
    # not just a single number
    for si in sortedIndices:
        # See if any neighbors have been assigned to a peak
        # No options for expanding which points are considered neighbors here
        # since the 8 surrounding points should be fine.
        assignedNeighbors = [(si[0]+i, si[1]+j) for i in [0, 1, -1] for j in [0, 1, -1]][1:]
        # Cut off points outside the domain, or that haven't been assigned yet
        assignedNeighbors = [n for n in assignedNeighbors if n[0] >= 0 and n[1] >= 0 and n[0] < data.shape[0] and n[1] < data.shape[1] and peakMembership[n] >= 0]

        # If there aren't any assigned neighbors yet, then we create a new
        # peak
        if len(assignedNeighbors) == 0:
            peakMembership[si] = len(peakBirthIndices)
            peakBirthIndices.append(si)

        # If only a single one has been assigned, or all of the assigned peaks have the
        # same membership, then this point joins that peak
        elif len(assignedNeighbors) == 1 or len(np.unique(np.array([peakMembership[n] for n in assignedNeighbors]))) == 1:
            peakMembership[si] = peakMembership[assignedNeighbors[0]]

        # Otherwise, we have to resolve a conflict between multiple, in which the
        # oldest one gains the new point.
        else:      
            # Find which one is the oldest
            order = np.argsort(np.array([data[peakBirthIndices[peakMembership[n]]] for n in assignedNeighbors]))[::-1]
            # New point joins oldest peak
            peakMembership[si] = peakMembership[assignedNeighbors[order[0]]]

    return peakMembership, peakBirthIndices


def approxPeakFind(discreteField, smoothing=0.05, threshold=1, lengthScale=0.15, periodic=False, debug=False):
    """
    Locate the maxima of a discrete field by finding clusters of points
    with a value greather than a threshold.

    Parameters
    ----------
    discreteField : numpy.ndarray[i,j,k,...]
        d-dimensional scalar field to perform peak finding on.

    smoothing : float
        Smoothing parameter that determines the size of the
        smoothing kernel applied to the discrete field.

        Calculates the kernel size as the diagonal length of the
        field times this parameter.

    threshold : float
        Factor to multiply the mean of the nonzero values in the
        discrete field by to determine which points could comprise a
        maximum.

    periodic : bool
        Whether the discrete field wraps around to itself at the edges.

    debug : bool
        Whether to plot debug information about the peak finding.

    Returns
    -------
    peaks : numpy.ndarray[N,d]
        Indices for each axis of the N detected peaks.
    """
    dim = len(np.shape(discreteField))

    smoothingKernel = int(np.sqrt(np.sum(np.array(np.shape(discreteField))**2))*smoothing)
    # Make sure it is odd
    smoothingKernel += (smoothingKernel+1) % 2
    
    # Use a gaussian profile (normalization is chosen as `smoothingKernel`
    # arbitrarily)
    kernelProfile = np.exp(-np.arange(-(smoothingKernel-1)//2, (smoothingKernel-1)//2+1)**2 / smoothingKernel)

    if dim > 1:
        kernel = np.multiply.outer(*[kernelProfile for _ in range(dim)])
    else:
        kernel = kernelProfile

    # If our field is periodic, we need to pad each side with the opposing
    # side.
    if periodic:
        padding = (smoothingKernel+1)//2
        paddedField = np.pad(discreteField, padding, mode='wrap')
        smoothedField = convolve(paddedField, kernel, mode='same')
        
        # Cut out the padding
        for i in range(dim):
            smoothedField = np.take(smoothedField, np.arange(padding, np.shape(paddedField)[i] - padding + 1), axis=i)
        
    else:
        smoothedField = convolve(discreteField, kernel, mode='same')

    # Now take points above the average multiplied by some factor
    #smoothedField[smoothedField == 0] = np.nan
    includePoints = np.array(np.where(smoothedField > np.nanmean(smoothedField)*threshold)).T
    
    if len(includePoints) == 0:
        return np.array([])

    # Cluster them together
    #centers, weights = spatialClusterCenters(includePoints, l=.1, wrapPoints=np.shape(smoothedField) if periodic else None, return_weights=True)
    # Renormalize weights
    #weights = weights / np.sum(weights)
    labels = spatialClusterLabels(includePoints, l=lengthScale, wrapPoints=np.shape(smoothedField) if periodic else None)
    centers = np.zeros((int(np.max(labels)+1), dim))

    for i in range(len(centers)):
        centers[i] = includePoints[labels == i][np.argmax([smoothedField[tuple(ind)] for ind in includePoints[labels == i]])]
        
    # Sort by weight
    #order = np.argsort(weights)
    #centers = centers[order]
    #weights = weights[order]
    
    if debug:
        if dim == 1:
            plt.plot(smoothedField)
            for i in range(int(np.max(labels)+1)):
                for j in includePoints[np.where(labels == i)]:
                    plt.axvline(j, c=str(colour.Color(pick_for=i)), linestyle='--', alpha=.5)

                plt.axvline(centers[i], c='tab:red')
            plt.show()

        if dim == 2:
            # Cluster them together
            #labels = spatialClusterLabels(includePoints, l=.1, wrapPoints=np.shape(smoothedField) if periodic else None)
        
            plt.imshow(smoothedField)
            for i in range(int(np.max(labels)+1)):
                plt.scatter(*includePoints[np.where(labels == i)].T[::-1], alpha=.3)
            plt.scatter(*centers.T[::-1], c='tab:red')
            plt.show()

    return centers


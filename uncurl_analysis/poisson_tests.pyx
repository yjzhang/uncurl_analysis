cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, log, log2, log10, exp2

from scipy.special import ndtr
from scipy import stats

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def log_wald_poisson_test(np.ndarray[double, ndim=1] data1,
        np.ndarray[double, ndim=1] data2,
        double counts1=0,
        double counts2=0):
    if counts1 == 0:
        counts1 = len(data1)
    if counts2 == 0:
        counts2 = len(data2)
    cdef double X1 = data1.sum()
    cdef double X0 = data2.sum()
    if X1 == 0 and X0 == 0:
        return 0.5, 1.0
    # add a 'pseudocount' of 0.5
    X1 += 0.5
    X0 += 0.5
    counts1 += 0.5
    counts2 += 0.5
    cdef double d = counts1/counts2
    cdef double W3 = (log(X1/X0) - log(d))*sqrt(1.0/X0 + 1.0/X1)
    if np.isnan(W3):
        return 0.5, 1.0
    # normal CDF
    # ndtr is much more computationally efficient than norm.cdf
    # on one test case, using norm.cdf took ~13 seconds
    # while using ndtr took ~4 seconds
    cdef double pv = 1 - ndtr(W3)
    #pv = 1 - scipy.stats.norm.cdf(W3)
    cdef double ratio = X1/(X0*d)
    return pv, ratio

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def uncurl_poisson_test_1_vs_rest(np.ndarray[double, ndim=2] m, np.ndarray[double, ndim=2] w, str mode='counts'):
    """
    Calculates 1-vs-rest ratios and p-values for all genes.

    mode can be 'cells' or 'counts'. In 'cells', the observation time in the
    Poisson test is the number of cells. In 'counts', the observation time
    is the total reads in the cells.

    Returns two arrays: all_pvs and all_ratios, of shape (genes, clusters).
    """
    # TODO: make this more efficient...
    cdef np.ndarray[Py_ssize_t, ndim=1] clusters = w.argmax(0)
    cdef Py_ssize_t genes = m.shape[0]
    cdef Py_ssize_t cells = w.shape[1]
    cdef Py_ssize_t g
    all_pvs = []
    all_ratios = []
    cluster_cell_counts = None
    if mode == 'counts':
        cell_counts = np.zeros(cells)
        cluster_cell_counts = np.zeros(w.shape[0])
        for i in range(cells):
            cell_counts[i] = np.sum(m.dot(w[:,i]))
        for k in range(w.shape[0]):
            cluster_cell_counts[k] = cell_counts[clusters==k].sum()
    for g in range(genes):
        cluster_pvs, cluster_ratios = poisson_test_1_vs_rest(m, w, clusters, g, cell_counts=cluster_cell_counts)
        all_pvs.append(cluster_pvs)
        all_ratios.append(cluster_ratios)
    return np.array(all_pvs), np.array(all_ratios)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def poisson_test_1_vs_rest(np.ndarray[double, ndim=2] m,
        np.ndarray[double, ndim=2] w,
        np.ndarray[long, ndim=1] clusters,
        Py_ssize_t gene_index,
        np.ndarray[double, ndim=1] cell_counts):
    cdef Py_ssize_t n_clusters = w.shape[0]
    cdef Py_ssize_t k
    cdef np.ndarray[double, ndim=1] gene_matrix = np.dot(m[gene_index, :], w)
    cluster_pvs = []
    cluster_ratios = []
    for k in range(n_clusters):
        cluster_cells = (clusters == k)
        in_cluster = gene_matrix[cluster_cells]
        if len(in_cluster) == 0:
            cluster_pvs.append(0.0)
            cluster_ratios.append(0.0)
        else:
            not_in_cluster = gene_matrix[~cluster_cells]
            if cell_counts is not None:
                counts1 = cell_counts[k]
                counts2 = cell_counts[:k].sum() + cell_counts[k+1:].sum()
                pv, ratio = log_wald_poisson_test(in_cluster, not_in_cluster, counts1, counts2)
            else:
                pv, ratio = log_wald_poisson_test(in_cluster, not_in_cluster)
            cluster_pvs.append(pv)
            cluster_ratios.append(ratio)
    return cluster_pvs, cluster_ratios

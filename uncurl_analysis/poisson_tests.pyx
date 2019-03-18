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
    """
    Source: [source here]
    """
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
def log_wald_poisson_test_counts(double X1,
        double X0,
        double counts1=0,
        double counts2=0):
    """
    Same as log-wald poisson test, but takes in counts
    rather than matrices of data.
    """
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
    cdef Py_ssize_t n_clusters = w.shape[0]
    cdef Py_ssize_t genes = m.shape[0]
    cdef Py_ssize_t cells = w.shape[1]
    cdef Py_ssize_t g, i, j, k
    cdef double pv, ratio, in_cluster_counts, not_in_cluster_counts, counts1, counts2
    cdef np.ndarray[double, ndim=1] cell_counts = np.zeros(cells)
    # cluster_cell_counts is the total counts of each cluster if mode=='counts', or the number of cells in each cluster otherwise.
    cdef np.ndarray[double, ndim=1] cluster_cell_counts = np.zeros(n_clusters)
    cdef np.ndarray[double, ndim=1] gene_matrix
    # cluster_gene_counts is the counts of each cluster for one gene
    cdef np.ndarray[double, ndim=1] cluster_gene_counts = np.zeros(n_clusters)
    cdef np.ndarray[double, ndim=2] all_pvs = np.zeros((genes, n_clusters))
    cdef np.ndarray[double, ndim=2] all_ratios = np.zeros((genes, n_clusters))
    if mode == 'counts':
        for i in range(cells):
            cell_counts[i] += m.dot(w[:,i]).sum()
        for k in range(n_clusters):
            cluster_cell_counts[k] = cell_counts[clusters==k].sum()
    else:
        for k in range(n_clusters):
            cluster_cell_counts[k] = (clusters==k).sum()
    for g in range(genes):
        cluster_gene_counts = np.zeros(n_clusters)
        gene_matrix = np.dot(m[g, :], w)
        for i in range(cells):
            j = clusters[i]
            cluster_gene_counts[j] += gene_matrix[i]
        for k in range(n_clusters):
            in_cluster_counts = cluster_gene_counts[k]
            not_in_cluster_counts = cluster_gene_counts[:k].sum() + cluster_gene_counts[k+1:].sum()
            counts1 = cluster_cell_counts[k]
            counts2 = cluster_cell_counts[:k].sum() + cluster_cell_counts[k+1:].sum()
            pv, ratio = log_wald_poisson_test_counts(in_cluster_counts, not_in_cluster_counts, counts1, counts2)
            all_pvs[g, k] = pv
            all_ratios[g, k] = ratio
    return all_pvs, all_ratios

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def uncurl_poisson_test_pairwise(np.ndarray[double, ndim=2] m, np.ndarray[double, ndim=2] w, str mode='counts'):
    """
    Pairwise Poisson tests between all clusters.

    Returns:
        all_pvs, all_ratios: two arrays of shape (genes, k, k) indicating the
        p-values between two clusters.

    """
    cdef np.ndarray[Py_ssize_t, ndim=1] clusters = w.argmax(0)
    cdef Py_ssize_t n_clusters = w.shape[0]
    cdef Py_ssize_t genes = m.shape[0]
    cdef Py_ssize_t cells = w.shape[1]
    cdef Py_ssize_t g, i, j, k, k1, k2
    cdef double pv, ratio, in_cluster_counts, not_in_cluster_counts, counts1, counts2
    cdef np.ndarray[double, ndim=1] cell_counts = np.zeros(cells)
    # cluster_cell_counts is the total counts of each cluster if mode=='counts', or the number of cells in each cluster otherwise.
    cdef np.ndarray[double, ndim=1] cluster_cell_counts = np.zeros(n_clusters)
    cdef np.ndarray[double, ndim=1] gene_matrix
    # cluster_gene_counts is the counts of each cluster for one gene
    cdef np.ndarray[double, ndim=1] cluster_gene_counts = np.zeros(n_clusters)
    cdef np.ndarray[double, ndim=3] all_pvs = np.zeros((genes, n_clusters, n_clusters))
    cdef np.ndarray[double, ndim=3] all_ratios = np.zeros((genes, n_clusters, n_clusters))
    if mode == 'counts':
        for i in range(cells):
            cell_counts[i] += m.dot(w[:,i]).sum()
        for k in range(n_clusters):
            cluster_cell_counts[k] = cell_counts[clusters==k].sum()
    else:
        for k in range(n_clusters):
            cluster_cell_counts[k] = (clusters==k).sum()
    for g in range(genes):
        cluster_gene_counts = np.zeros(n_clusters)
        gene_matrix = np.dot(m[g, :], w)
        for i in range(cells):
            j = clusters[i]
            cluster_gene_counts[j] += gene_matrix[i]
        for k1 in range(n_clusters):
            in_cluster_counts = cluster_gene_counts[k1]
            counts1 = cluster_cell_counts[k1]
            for k2 in range(n_clusters):
                not_in_cluster_counts = cluster_gene_counts[k2]
                counts2 = cluster_cell_counts[k2]
                pv, ratio = log_wald_poisson_test_counts(in_cluster_counts, not_in_cluster_counts, counts1, counts2)
                all_pvs[g, k1, k2] = pv
                all_ratios[g, k1, k2] = ratio
    return all_pvs, all_ratios


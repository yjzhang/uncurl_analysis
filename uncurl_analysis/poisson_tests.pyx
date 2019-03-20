cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, log, log2, log10, exp2

from scipy.special import ndtr, stdtr


cdef inline double t_test(double s1, double s2, double v1, double v2,
        double n1, double n2):
    """
    Computes a two-sample one-sided t-test, returning the p-value.

    Args:
        s1 - sum of sample 1
        s2 - sum of sample 2
        v1 - variance of sample 1
        v2 - variance of sample 2
        n1 - cell counts of sample 1
        n2 - cell counts of sample 2
    """
    cdef double m1 = s1/n1
    cdef double m2 = s2/n2
    cdef double t_test_statistic = (m1 - m2)/sqrt(v1/n1 + v2/n2)
    # TODO: we really should try Welch's dof procedure
    cdef double dof = max(n1 + n2 - 2, 1)
    cdef double pval = 1 - stdtr(dof, t_test_statistic)
    cdef double ratio = m1/m2
    return pval, ratio

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def log_wald_poisson_test(np.ndarray[double, ndim=1] data1,
        np.ndarray[double, ndim=1] data2,
        double counts1=0,
        double counts2=0):
    """
    Source: this is the W3 statistic from https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.1949

    See also: https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-5-S3-S1
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
    cdef double W3 = (log(X1/X0) - log(d))/sqrt(1.0/X0 + 1.0/X1)
    if np.isnan(W3):
        return 0.5, 1.0
    # normal CDF
    # ndtr is much more computationally efficient than norm.cdf
    # on one test case, using norm.cdf took ~13 seconds
    # while using ndtr took ~4 seconds
    cdef double pv = 1 - ndtr(W3)
    cdef double ratio = X1/(X0*d)
    return pv, ratio

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def log_wald_poisson_test_counts(double X1,
        double X0,
        double counts1,
        double counts2):
    """
    Same as log_wald_poisson_test, but takes in counts
    rather than arrays of data.
    """
    if X1 == 0 and X0 == 0:
        return 0.5, 1.0
    # add a 'pseudocount' of 0.5
    X1 += 0.5
    X0 += 0.5
    counts1 += 0.5
    counts2 += 0.5
    cdef double d = counts1/counts2
    cdef double W3 = (log(X1/X0) - log(d))/sqrt(1.0/X0 + 1.0/X1)
    if np.isnan(W3):
        return 0.5, 1.0
    # normal CDF
    # ndtr is much more computationally efficient than scipy.stats.norm.cdf
    # on one test case, using scipy.stats.norm.cdf took ~13 seconds
    # while using ndtr took ~4 seconds
    cdef double pv = 1 - ndtr(W3)
    cdef double ratio = X1/(X0*d)
    return pv, ratio


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def uncurl_test_1_vs_rest(np.ndarray[double, ndim=2] m, np.ndarray[double, ndim=2] w, str mode='counts',
        str test='poisson',
        np.ndarray[Py_ssize_t, ndim=1] clusters=None):
    """
    Calculates 1-vs-rest ratios and p-values for all genes.

    mode can be 'cells' or 'counts'. In 'cells', the observation time in the
    Poisson test is the number of cells. In 'counts', the observation time
    is the total reads in the cells.

    Returns two arrays: all_pvs and all_ratios, of shape (genes, clusters).
    """
    if clusters is None:
        clusters = w.argmax(0)
    cdef int use_t_test = 0
    if test == 't':
        use_t_test = 1
    cdef Py_ssize_t n_clusters = len(set(clusters))
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
    # these are only necessary for t-tests
    cdef np.ndarray[double, ndim=1] cluster_means
    cdef np.ndarray[double, ndim=1] cluster_variances
    # outputs
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
        if use_t_test:
            cluster_means = np.zeros(n_clusters)
            cluster_variances = np.zeros(n_clusters)
        for i in range(cells):
            j = clusters[i]
            cluster_gene_counts[j] += gene_matrix[i]
            if use_t_test:
                pass
        for k in range(n_clusters):
            in_cluster_counts = cluster_gene_counts[k]
            not_in_cluster_counts = cluster_gene_counts[:k].sum() + cluster_gene_counts[k+1:].sum()
            counts1 = cluster_cell_counts[k]
            counts2 = cluster_cell_counts[:k].sum() + cluster_cell_counts[k+1:].sum()
            if use_t_test:
                pass
            else:
                pv, ratio = log_wald_poisson_test_counts(in_cluster_counts, not_in_cluster_counts, counts1, counts2)
            all_pvs[g, k] = pv
            all_ratios[g, k] = ratio
    return all_pvs, all_ratios

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def uncurl_test_pairwise(np.ndarray[double, ndim=2] m, np.ndarray[double, ndim=2] w, str mode='counts',
        str test='poisson',
        np.ndarray[Py_ssize_t, ndim=1] clusters=None):
    """
    Pairwise Poisson tests between all clusters.

    Returns:
        all_pvs, all_ratios: two arrays of shape (genes, k, k) indicating the
        p-values between two clusters.

    """
    if clusters is None:
        clusters = w.argmax(0)
    cdef int use_t_test = 0
    if test == 't':
        use_t_test = 1
    cdef Py_ssize_t n_clusters = len(set(clusters))
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
    # these are only necessary for t-tests
    cdef np.ndarray[double, ndim=1] cluster_means
    cdef np.ndarray[double, ndim=1] cluster_variances
    # outputs
    cdef np.ndarray[double, ndim=3] all_pvs = np.zeros((genes, n_clusters, n_clusters))
    cdef np.ndarray[double, ndim=3] all_ratios = np.zeros((genes, n_clusters, n_clusters))
    # TODO: if test is 't', calculate mean and variance across cells
    # use an online mean/variance algorithm, just for computational efficiency?
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
        if use_t_test:
            cluster_means = np.zeros(n_clusters)
            cluster_variances = np.zeros(n_clusters)
        for i in range(cells):
            j = clusters[i]
            cluster_gene_counts[j] += gene_matrix[i]
            if use_t_test:
                pass
        for k1 in range(n_clusters):
            in_cluster_counts = cluster_gene_counts[k1]
            counts1 = cluster_cell_counts[k1]
            for k2 in range(n_clusters):
                not_in_cluster_counts = cluster_gene_counts[k2]
                counts2 = cluster_cell_counts[k2]
                if use_t_test:
                    pass
                else:
                    pv, ratio = log_wald_poisson_test_counts(in_cluster_counts, not_in_cluster_counts, counts1, counts2)
                all_pvs[g, k1, k2] = pv
                all_ratios[g, k1, k2] = ratio
    return all_pvs, all_ratios


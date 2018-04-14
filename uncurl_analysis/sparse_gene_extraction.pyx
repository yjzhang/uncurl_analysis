cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, log2, log10, exp2

from scipy import sparse
from scipy.special import xlogy
from scipy import stats

ctypedef fused int2:
    short
    int
    long
    long long

ctypedef fused DTYPE_t:
    float
    double

ctypedef fused numeric:
    int
    long
    long long
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_weighted_cluster_means(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[double, ndim=2] W,
        Py_ssize_t cells,
        Py_ssize_t genes):
    """
    Returns a 2d array of cluster means weighted by W.
    """
    cdef numeric[:] data_ = data
    cdef int2[:] indices_ = indices
    cdef int2[:] indptr_ = indptr
    cdef double[:,:] W_ = W
    cdef int2 g, c, start_ind, end_ind, i2
    cdef int2 k
    cdef int K = W.shape[0]
    cdef double[:,:] cluster_means = np.zeros((genes, K))
    cdef double[:] cluster_cell_counts = np.zeros(K)
    for c in range(cells):
        start_ind = indptr_[c]
        end_ind = indptr_[c+1]
        for k in range(K):
            cluster_cell_counts[k] += W_[k, c]
            for i2 in range(start_ind, end_ind):
                g = indices_[i2]
                cluster_means[g, k] += data_[i2]*W_[k, c]
    for g in range(genes):
        for k in range(K):
            cluster_means[g, k] = cluster_means[g, k]/cluster_cell_counts[k]
    return cluster_means, cluster_cell_counts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_unweighted_cluster_means(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes):
    """
    returns a cluster-specific means matrix, given labels.
    """
    cdef numeric[:] data_ = data
    cdef int2[:] indices_ = indices
    cdef int2[:] indptr_ = indptr
    cdef int2 g, c, start_ind, end_ind, i2
    cdef int k
    labels_set = set(labels)
    cdef int2 K = len(labels_set)
    cdef double[:,:] cluster_means = np.zeros((genes, K))
    cdef double[:] cluster_cell_counts = np.zeros(K)
    for c in range(cells):
        k = labels[c]
        cluster_cell_counts[k] += 1
        start_ind = indptr_[c]
        end_ind = indptr_[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices_[i2]
            cluster_means[g, k] += data_[i2]
    for g in range(genes):
        for k in range(K):
            cluster_means[g, k] = cluster_means[g, k]/cluster_cell_counts[k]
    return cluster_means, cluster_cell_counts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_c_scores(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=0):
    """
    Find overexpressed genes given a csc matrix...
    """
    if eps == 0:
        eps = 10.0/cells
    cdef double[:,:] cluster_means
    cluster_means, _ = csc_unweighted_cluster_means(
            data, indices, indptr, labels, cells, genes)
    labels_set = set(labels)
    cdef int2 K = len(labels_set)
    scores = {}
    cdef int2 g, k
    for k in labels_set:
        scores[k] = []
    cdef double max_k, max_k2
    for g in range(genes):
        for k in range(K):
            max_k = cluster_means[g, k] + eps
            max_k2 = 0
            for k2 in range(K):
                if k2 != k:
                    if cluster_means[g,k2] > max_k2:
                        max_k2 = cluster_means[g,k2]
            scores[k].append((g, max_k/(max_k2 + eps)))
    return scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_weighted_c_scores(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[double, ndim=2] W,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=0):
    """
    Find overexpressed genes given a csc matrix and W, using weighted means.
    """
    cdef double[:,:] cluster_means
    cluster_means, _ = csc_weighted_cluster_means(
            data, indices, indptr, W, cells, genes)
    cdef int K = W.shape[0]
    scores = {}
    cdef int2 g, k
    for k in range(K):
        scores[k] = []
    cdef double max_k, max_k2
    for g in range(genes):
        for k in range(K):
            max_k = cluster_means[g, k] + eps
            max_k2 = 0
            for k2 in range(K):
                if k2 != k:
                    if cluster_means[g,k2] > max_k2:
                        max_k2 = cluster_means[g,k2]
            scores[k].append((g, max_k/(max_k2 + eps)))
    return scores


def t_test(double m1, double m2, double v1, double v2,
        double n1, double n2):
    """
    Computes a two-sample t-test, returning the p-value.
    """
    cdef double t_test_statistic = (m1 - m2)/sqrt(v1/n1 + v2/n2)
    # TODO: we really should try Welch's dof procedure
    cdef int dof = int(round(n1 + n2 - 2))
    cdef double pval = 1 - stats.t.cdf(t_test_statistic, dof)
    return pval


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_weighted_t_test(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[double, ndim=2] W,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=1.0):
    """
    Returns the pairwise t-test score and p-val between each pair of clusters, for
    all genes.
    """
    cdef double[:,:] cluster_means
    cdef double[:,:] cluster_variances
    cdef double[:] cluster_cell_counts
    cdef np.ndarray[double, ndim=1] log_data = np.log2(data + eps)
    cluster_means, cluster_cell_counts = csc_weighted_cluster_means(
            log_data, indices, indptr, W, cells, genes)
    # calculate variance... var = E[X]^2 - E[X^2]
    cdef int2 g, k
    cdef np.ndarray[double, ndim=1] log_data_sq = log_data**2
    cdef double[:,:] cluster_sq_means
    cluster_sq_means, _ = csc_weighted_cluster_means(
            log_data_sq, indices, indptr, W, cells, genes)
    cdef int K = W.shape[0]
    cdef double[:,:,:] scores = np.zeros((K, K, genes))
    cdef double[:,:,:] pvals = np.zeros((K, K, genes))
    cdef double mean_k, mean_k2, var_k, var_k2
    for g in range(genes):
        for k in range(K):
            mean_k = cluster_means[g, k]
            var_k = cluster_means[g, k]**2 - cluster_sq_means[g, k]
            for k2 in range(K):
                mean_k2 = cluster_means[g, k2]
                var_k2 = mean_k2**2 - cluster_sq_means[g, k2]
                scores[k, k2, g] = mean_k - cluster_means[g, k2]
                pvals[k, k2, g] = t_test(mean_k, cluster_means[g, k2], var_k,
                        var_k2,
                        cluster_cell_counts[k],
                        cluster_cell_counts[k2])
    return np.asarray(scores), np.asarray(pvals)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_unweighted_t_test(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=1.0):
    """
    Returns the pairwise t-test score and p-val between each pair of clusters, for
    all genes.
    """
    cdef double[:,:] cluster_means
    cdef double[:,:] cluster_variances
    cdef double[:] cluster_cell_counts
    cdef np.ndarray[double, ndim=1] log_data = np.log2(data + eps)
    cluster_means, cluster_cell_counts = csc_unweighted_cluster_means(
            log_data, indices, indptr, labels, cells, genes)
    # calculate variance... var = E[X]^2 - E[X^2]
    cdef int2 g, k
    cdef np.ndarray[double, ndim=1] log_data_sq = np.power(log_data, 2)
    cdef double[:,:] cluster_sq_means
    cluster_sq_means, _ = csc_unweighted_cluster_means(
            log_data_sq, indices, indptr, labels, cells, genes)
    labels_set = set(labels)
    cdef int2 K = len(labels_set)
    cdef double[:,:,:] scores = np.zeros((K, K, genes))
    cdef double[:,:,:] pvals = np.zeros((K, K, genes))
    cdef double mean_k, mean_k2, var_k, var_k2
    for g in range(genes):
        for k in range(K):
            mean_k = cluster_means[g, k]
            var_k = cluster_sq_means[g, k] - mean_k**2
            for k2 in range(K):
                mean_k2 = cluster_means[g, k2]
                var_k2 = cluster_sq_means[g, k2] - mean_k2**2
                scores[k, k2, g] = mean_k - cluster_means[g, k2]
                if scores[k, k2, g] == 0:
                    pvals[k, k2, g] = 1
                else:
                    pvals[k, k2, g] = t_test(mean_k, mean_k2, var_k,
                            var_k2,
                            cluster_cell_counts[k],
                            cluster_cell_counts[k2])
    return np.asarray(scores), np.asarray(pvals)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def t_test_c_scores(np.ndarray[double, ndim=3] scores,
        np.ndarray[double, ndim=3] pvals):
    """
    Converts the output of the pairwise t-test procedure into c-scores.

    The output is two dicts: one giving c-scores for each cluster,
    and the other giving p-vals corresponding to those c-scores.
    """
    cdef Py_ssize_t k, k2, K, genes, g
    cdef double cs, min_cs, best_pval
    cdef double[:,:,:] scores_ = scores
    cdef double[:,:,:] pvals_ = pvals
    K = scores.shape[0]
    genes = scores.shape[2]
    c_scores = {}
    c_pvals = {}
    for k in range(K):
        c_scores[k] = []
        c_pvals[k] = []
    for g in range(genes):
        for k in range(K):
            min_cs = 1e10
            best_pval = 1
            for k2 in range(K):
                cs = scores_[k, k2, g]
                if k2 != k and cs <= min_cs:
                    min_cs = cs
                    best_pval = pvals_[k, k2, g]
            c_scores[k].append((g, exp2(min_cs)))
            c_pvals[k].append((g, best_pval))
    for k in range(K):
        k_c_scores = np.array([x[1] for x in c_scores[k]])
        indices = k_c_scores.argsort()[::-1]
        c_scores[k] = [c_scores[k][i] for i in indices]
        c_pvals[k] = [c_pvals[k][i] for i in indices]
    return c_scores, c_pvals

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def t_test_separation_scores(np.ndarray[double, ndim=3] scores,
        np.ndarray[double, ndim=3] pvals, double eps=1e-30):
    """
    Converts the output of the pairwise t-test procedure into separation scores.

    The output is a K x K array of pairwise separation scores.

    Citation for separation score:
    Zhang, J. M., Fan, J., Fan, H. C., Rosenfeld, D. & Tse, D. N. An interpretable framework for clustering single-cell RNA-Seq datasets. BMC Bioinformatics 19, (2018).
    """
    cdef Py_ssize_t k, k1, k2, K, genes, g
    cdef double cs, max_cs, best_pval
    K = scores.shape[0]
    genes = scores.shape[2]
    cdef double[:,:] separation_scores = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            best_pval = 1
            for g in range(genes):
                if pvals[k1, k2, g] < best_pval:
                    best_pval = pvals[k1, k2, g]
            separation_scores[k1, k2] = -log10(best_pval + eps)
    return np.array(separation_scores)

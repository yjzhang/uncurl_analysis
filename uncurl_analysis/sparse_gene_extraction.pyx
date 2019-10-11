cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, log2, log10, exp2

from scipy import sparse
from scipy.special import xlogy, stdtr

ctypedef fused int2:
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

cdf = stdtr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_weighted_cluster_means(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[double, ndim=2] W,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=1e-10):
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
    #print(W.shape[0], W.shape[1])
    #print(cells)
    #print(genes)
    cdef double[:,:] cluster_means = np.zeros((genes, K))
    cdef double[:] cluster_cell_counts = np.zeros(K) + eps
    for c in range(cells):
        start_ind = indptr_[c]
        end_ind = indptr_[c+1]
        for k in range(K):
            cluster_cell_counts[k] += W_[k, c]
            for i2 in range(start_ind, end_ind):
                g = indices_[i2]
                cluster_means[g, k] += data_[i2]*W_[k, c]
    #print('W.sum(0): ')
    #print(W.sum(0))
    #print('W.sum(1): ')
    #print(W.sum(1))
    #print('cluster_cell_counts:')
    #print(np.asarray(cluster_cell_counts))
    for g in range(genes):
        for k in range(K):
            cluster_means[g, k] = cluster_means[g, k]/(cluster_cell_counts[k] + eps)
    return cluster_means, cluster_cell_counts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_unweighted_cluster_means(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=0.0):
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
    cdef long[:] cluster_cell_counts = np.zeros(K).astype(np.int64)
    cdef double[:,:] cluster_gene_nonzeros = np.ones((genes, K))
    for c in range(cells):
        k = labels[c]
        cluster_cell_counts[k] += 1
        start_ind = indptr_[c]
        end_ind = indptr_[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices_[i2]
            cluster_means[g, k] += data_[i2]
            cluster_gene_nonzeros[g, k] += 1
    for g in range(genes):
        for k in range(K):
            if cluster_cell_counts[k] > 0:
                cluster_means[g, k] = cluster_means[g, k]/cluster_cell_counts[k]
    return cluster_means, cluster_cell_counts, cluster_gene_nonzeros

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_1_vs_rest_cluster_means(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes):
    """
    Returns 3 arrays: cluster_means, rest_cluster_means (mean of all cells
    NOT in cluster k), of shape (genes, K), and cluster_cell_counts, of shape K
    """
    cdef numeric[:] data_ = data
    cdef int2[:] indices_ = indices
    cdef int2[:] indptr_ = indptr
    cdef int2 g, c, start_ind, end_ind, i2
    cdef int k, k2
    labels_set = set(labels)
    cdef int2 K = len(labels_set)
    cdef double[:,:] cluster_means = np.zeros((genes, K))
    cdef double[:,:] rest_cluster_means = np.zeros((genes, K))
    cdef long[:] cluster_cell_counts = np.zeros(K).astype(int)
    cdef double[:,:] cluster_gene_nonzeros = np.ones((genes, K))
    cdef double[:,:] rest_cluster_gene_nonzeros = np.ones((genes, K))
    for c in range(cells):
        k = labels[c]
        cluster_cell_counts[k] += 1
        start_ind = indptr_[c]
        end_ind = indptr_[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices_[i2]
            cluster_means[g, k] += data_[i2]
            cluster_gene_nonzeros[g, k] += 1
            for k2 in range(K):
                if k2 != k:
                    rest_cluster_means[g, k2] += data_[i2]
                    rest_cluster_gene_nonzeros[g, k2] += 1
    for g in range(genes):
        for k in range(K):
            if cluster_cell_counts[k] > 0:
                cluster_means[g, k] = cluster_means[g, k]/cluster_cell_counts[k]
            rest_cluster_means[g, k] = rest_cluster_means[g, k]/(cells - cluster_cell_counts[k])
    # return nonzeros
    return cluster_means, rest_cluster_means, cluster_cell_counts, cluster_gene_nonzeros, rest_cluster_gene_nonzeros


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_1_vs_rest_lists(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes):
    """
    Returns 2 arrays: cluster_vals, rest_cluster_vals, lists
    of lists for every cluster/gene.
    """
    cdef numeric[:] data_ = data
    cdef int2[:] indices_ = indices
    cdef int2[:] indptr_ = indptr
    cdef int2 g, c, start_ind, end_ind, i2
    cdef int k, k2, remaining_length
    labels_set = set(labels)
    cdef int K = len(labels_set)
    cdef long[:] cluster_cell_counts = np.zeros(K).astype(int)
    cluster_vals = [[[] for i in range(genes)] for k in range(K)]
    rest_cluster_vals = [[[] for i in range(genes)] for k in range(K)]
    for c in range(cells):
        k = labels[c]
        cluster_cell_counts[k] += 1
        start_ind = indptr_[c]
        end_ind = indptr_[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices_[i2]
            cluster_vals[k][g].append(data_[i2])
            for k2 in range(K):
                if k2 != k:
                    rest_cluster_vals[k2][g].append(data_[i2])

    return cluster_vals, rest_cluster_vals

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
    cluster_means, _, _ = csc_unweighted_cluster_means(
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


cdef inline double t_test(double m1, double m2, double v1, double v2,
        double n1, double n2):
    """
    Computes a two-sample t-test, returning the p-value.
    """
    cdef double t_test_statistic = (m1 - m2)/sqrt(v1/n1 + v2/n2)
    # TODO: we really should try Welch's dof procedure
    cdef double dof = max(n1 + n2 - 2, 1)
    cdef double pval = 1 - cdf(dof, t_test_statistic)
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
        double eps=1.0,
        int calc_pvals=True):
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
    cdef np.ndarray[double, ndim=1] log_data_sq = log_data**2
    cdef double[:,:] cluster_sq_means
    cluster_sq_means, _ = csc_weighted_cluster_means(
            log_data_sq, indices, indptr, W, cells, genes)
    cdef int K = W.shape[0]
    cdef int g, k, k2
    cdef double[:,:,:] scores = np.zeros((K, K, genes))
    cdef double[:,:,:] pvals = np.zeros((K, K, genes))
    cdef double mean_k, mean_k2, var_k, var_k2
    for g in range(genes):
        for k in range(K):
            mean_k = cluster_means[g, k]
            var_k = cluster_sq_means[g, k] - cluster_means[g, k]**2
            for k2 in range(K):
                mean_k2 = cluster_means[g, k2]
                var_k2 = cluster_sq_means[g, k2] - mean_k2**2
                scores[k, k2, g] = mean_k - mean_k2
                # truncate this so that we don't have to perform so many
                # t-test calculations - this is solely for efficiency :(
                # print(g, k, k2, var_k, var_k2, cluster_cell_counts[k])
                if calc_pvals:
                    if (var_k == 0 and var_k2 == 0) or (cluster_cell_counts[k2] == 0 and cluster_cell_counts[k] == 0):
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
def csc_unweighted_t_test(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=1.0,
        int calc_pvals=True,
        int use_nonzeros=False):
    """
    Returns the pairwise t-test score and p-val between each pair of clusters, for
    all genes.
    """
    cdef double[:,:] cluster_means
    #base_means is non-log
    cdef double[:,:] base_means
    cdef double[:,:] cluster_variances
    cdef double[:,:] nonzeros
    cdef long[:] cluster_cell_counts
    # log_data is log1p(data)
    cdef np.ndarray[double, ndim=1] log_data = np.log2(data + 1.0)
    cluster_means, cluster_cell_counts, nonzeros = csc_unweighted_cluster_means(
            log_data, indices, indptr, labels, cells, genes)
    # calculate ratios using base_means
    base_means, cluster_cell_counts, _ = csc_unweighted_cluster_means(
            data, indices, indptr, labels, cells, genes)
    # calculate variance... var = E[X]^2 - E[X^2]
    cdef int2 g, k
    cdef np.ndarray[double, ndim=1] log_data_sq = np.power(log_data, 2)
    cdef double[:,:] cluster_sq_means
    cluster_sq_means, _, _ = csc_unweighted_cluster_means(
            log_data_sq, indices, indptr, labels, cells, genes)
    labels_set = set(labels)
    cdef int2 K = len(labels_set)
    cdef double[:,:,:] scores = np.zeros((K, K, genes))
    cdef double[:,:,:] pvals = np.zeros((K, K, genes))
    cdef double mean_k, mean_k2, var_k, var_k2, score, base_mean_k
    cdef double[:,:] cluster_vars = np.zeros((genes, K))
    for g in range(genes):
        for k in range(K):
            cluster_vars[g, k] = cluster_sq_means[g, k] - cluster_means[g, k]**2
    for g in range(genes):
        for k in range(K):
            mean_k = cluster_means[g, k]
            base_mean_k = base_means[g, k]
            var_k = cluster_vars[g, k]
            for k2 in range(K):
                mean_k2 = cluster_means[g, k2]
                var_k2 = cluster_vars[g, k2]
                score = mean_k - cluster_means[g, k2]
                scores[k, k2, g] = (base_mean_k + eps)/(base_means[g, k2] + eps)
                # truncate this so that we don't have to perform so many
                # t-test calculations - this is solely for efficiency :(
                if calc_pvals:
                    if (var_k == 0 and var_k2 == 0) or cluster_cell_counts[k2] == 0 or cluster_cell_counts[k] == 0:
                        pvals[k, k2, g] = 1
                    else:
                        # use nonzero counts as degrees of freedom?
                        pvals[k, k2, g] = t_test(mean_k, mean_k2, var_k,
                                var_k2,
                                nonzeros[g, k],
                                nonzeros[g, k2])
    return np.asarray(scores), np.asarray(pvals)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_unweighted_1_vs_rest_rank_sum_test(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=1.0,
        int calc_pvals=True,
        str mode='u'):
    """
    Output is two dicts of {cluster : [list of (gene, ratio)]}, {cluster : [list of (gene, p-val)]}, where
    the lists are sorted by descending ratio/ascending p-val.
    """
    from scipy.stats import mannwhitneyu
    # TODO: runs Mann-Whitney U-test OR Wilcoxon rank-sum test
    # cluster_means necessary to calculate ratios
    cdef double[:,:] cluster_means
    cdef double[:,:] rest_cluster_means
    cdef double[:,:] base_means
    cdef double[:,:] rest_base_means
    cdef double[:,:] cluster_variances
    cdef long[:] cluster_cell_counts
    labels_set = set(labels)
    cdef long K = len(labels_set)
    base_means, rest_base_means, cluster_cell_counts, cg_nonzeros, rest_cg_nonzeros = csc_1_vs_rest_cluster_means(
            data, indices, indptr, labels, cells, genes)
    cluster_vals, rest_cluster_vals = csc_1_vs_rest_lists(
            data, indices, indptr, labels, cells, genes)
    cdef long g, k, remaining_length_cluster, remaining_length_rest
    cdef double[:,:] ratios = np.zeros((K, genes))
    cdef double[:,:] pvals = np.zeros((K, genes))
    # the rank-sum test requires a list of values for gene g and cluster k,
    # and a list of "other" values.
    for k in range(K):
        cluster_genes = cluster_vals[k]
        rest_genes = rest_cluster_vals[k]
        for g in range(genes):
            ratios[k, g] = (base_means[g, k] + eps)/(rest_base_means[g, k] + eps)
            if ratios[k, g] > 1.0:
                try:
                    # TODO: ZERO VALUES have to be included
                    remaining_length_cluster = cluster_cell_counts[k] - len(cluster_genes[g])
                    remaining_length_rest = cells - cluster_cell_counts[k] - len(rest_genes[g])
                    stat, pval = mannwhitneyu(cluster_genes[g] + [0]*remaining_length_cluster,
                            rest_genes[g] + [0]*remaining_length_rest, alternative='greater')
                except:
                    pval = 0.99
                pvals[k, g] = pval
            else:
                pvals[k, g] = 0.99
    scores_output = {}
    pvals_output = {}
    for k in range(K):
        scores_output[k] = []
        pvals_output[k] = []
    for g in range(genes):
        for k in range(K):
            scores_output[k].append((g, ratios[k, g]))
            pvals_output[k].append((g, pvals[k, g]))
    # this section sorts all the c-scores and pvals.
    for k in range(K):
        scores_output[k].sort(key=lambda x: x[1], reverse=True)
        pvals_output[k].sort(key=lambda x: x[1])
    return scores_output, pvals_output



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_unweighted_1_vs_rest_t_test(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=1.0,
        int calc_pvals=True,
        int use_nonzeros=False):
    """
    Returns a 1 vs rest t-test for every gene and cluster.

    Output is two dicts of {cluster : [list of (gene, ratio)]}, {cluster : [list of (gene, p-val)]}, where
    the lists are sorted by descending ratio/ascending p-val.

    Mode: could be either 't', 'mann-whitney', or 'wilcoxon'
    """
    cdef double[:,:] cluster_means
    cdef double[:,:] rest_cluster_means
    cdef double[:,:] base_means
    cdef double[:,:] rest_base_means
    cdef double[:,:] cluster_variances
    cdef double[:,:] cg_nonzeros
    cdef double[:,:] rest_cg_nonzeros
    cdef long[:] cluster_cell_counts
    cdef np.ndarray[double, ndim=1] log_data = np.log2(data + 1.0)
    cluster_means, rest_cluster_means, cluster_cell_counts, cg_nonzeros, rest_cg_nonzeros  = csc_1_vs_rest_cluster_means(
            log_data, indices, indptr, labels, cells, genes)
    base_means, rest_base_means, cluster_cell_counts, _, _  = csc_1_vs_rest_cluster_means(
            data, indices, indptr, labels, cells, genes)
    # calculate variance... var = E[X]^2 - E[X^2]
    cdef long g, k
    cdef np.ndarray[double, ndim=1] log_data_sq = np.power(log_data, 2)
    cdef double[:,:] cluster_sq_means
    cluster_sq_means, rest_cluster_sq_means, _, _, _  = csc_1_vs_rest_cluster_means(
            log_data_sq, indices, indptr, labels, cells, genes)
    labels_set = set(labels)
    cdef long K = len(labels_set)
    cdef double[:,:] scores = np.zeros((genes, K))
    cdef double[:,:] pvals = np.zeros((genes, K))
    cdef double mean_k, mean_k2, var_k, var_k2, score
    cdef double[:,:] cluster_vars = np.zeros((genes, K))
    cdef double[:,:] rest_cluster_vars = np.zeros((genes, K))
    for g in range(genes):
        for k in range(K):
            cluster_vars[g, k] = cluster_sq_means[g, k] - cluster_means[g, k]**2
            rest_cluster_vars[g, k] = rest_cluster_sq_means[g, k] - rest_cluster_means[g, k]**2
    for g in range(genes):
        for k in range(K):
            mean_k = cluster_means[g, k]
            var_k = cluster_vars[g, k]
            # sum up all the "rest" clusters
            mean_k2 = rest_cluster_means[g, k]
            var_k2 = rest_cluster_vars[g, k]
            score = mean_k - mean_k2
            scores[g, k] = (base_means[g, k] + eps)/(rest_base_means[g, k] + eps)
            if calc_pvals:
                if (var_k == 0 and var_k2 == 0) or cluster_cell_counts[k] == 0:
                    pvals[g, k] = 1
                else:
                    pvals[g, k] = t_test(mean_k, mean_k2, var_k,
                            var_k2,
                            cg_nonzeros[g, k],
                            rest_cg_nonzeros[g, k])
    scores_output = {}
    pvals_output = {}
    for k in range(K):
        scores_output[k] = []
        pvals_output[k] = []
    for g in range(genes):
        for k in range(K):
            scores_output[k].append((g, scores[g, k]))
            pvals_output[k].append((g, pvals[g, k]))
    # this section sorts all the c-scores and pvals.
    for k in range(K):
        scores_output[k].sort(key=lambda x: x[1], reverse=True)
        pvals_output[k].sort(key=lambda x: x[1])
    return scores_output, pvals_output



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
            c_scores[k].append((g, min_cs))
            c_pvals[k].append((g, best_pval))
    for k in range(K):
        k_c_scores = np.array([x[1] for x in c_scores[k]])
        indices = k_c_scores.argsort()[::-1]
        c_scores[k] = [c_scores[k][i] for i in indices]
        k_p_vals = np.array([x[1] for x in c_pvals[k]])
        p_indices = k_p_vals.argsort()
        c_pvals[k] = [c_pvals[k][i] for i in p_indices]
    return c_scores, c_pvals

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def t_test_separation_scores(np.ndarray[double, ndim=3] scores,
        np.ndarray[double, ndim=3] pvals, double eps=1e-100):
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
    cdef int[:,:] best_genes = np.zeros((K, K), dtype=np.int32)
    cdef int best_gene
    for k1 in range(K):
        for k2 in range(K):
            if k2 == k1:
                continue
            best_pval = 1
            best_gene = 0
            for g in range(genes):
                if pvals[k1, k2, g] < best_pval:
                    best_pval = pvals[k1, k2, g]
                    best_gene = g
            separation_scores[k1, k2] = -log10(best_pval + eps)
            best_genes[k1, k2] = best_gene
    return np.array(separation_scores), np.array(best_genes)

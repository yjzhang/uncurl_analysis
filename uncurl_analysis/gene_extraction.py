from __future__ import print_function
import sys

import numpy as np
from scipy import sparse

from uncurl_analysis.sparse_gene_extraction import csc_c_scores, csc_weighted_c_scores, csc_weighted_t_test, csc_unweighted_t_test, t_test_c_scores, t_test_separation_scores, csc_unweighted_1_vs_rest_t_test, csc_unweighted_1_vs_rest_rank_sum_test

# TODO: efficient sparse implementation of find_overexpressed_genes?
def find_overexpressed_genes(data, labels, eps=0):
    """
    Returns a dict of label : list of (gene id, score) pairs for all genes,
    sorted by descending score.

    Args:
        data (array): dense or sparse array of shape genes x cells
        labels (array): 1d array of ints

    Returns:
        scores - dict of {label : [(gene_id, score) sorted by descending score]}
    """
    genes = data.shape[0]
    cells = data.shape[1]
    if eps==0:
        eps = 10.0/cells
    labels_set = set(labels)
    if sparse.issparse(data):
        # need to convert labels to an integer array
        labels_array = np.zeros(len(labels), dtype=int)
        labels_map = {}
        for i, l in enumerate(sorted(list(set(labels)))):
            labels_map[l] = i
        for i, l in enumerate(labels):
            labels_array[i] = labels_map[l]
        data_csc = sparse.csc_matrix(data)
        scores = csc_c_scores(
                data_csc.data,
                data_csc.indices,
                data_csc.indptr,
                labels_array,
                cells,
                genes,
                eps)
        inverse_labels_map = {k:i for i, k in labels_map.items()}
        new_scores = {}
        for k, s in scores.items():
            new_scores[inverse_labels_map[k]] = s
        scores = new_scores
    else:
        scores = {}
        for k in labels_set:
            scores[k] = []
        for g in range(genes):
            data_g = data[g,:]
            for k in labels_set:
                cells_c = data_g[labels==k]
                #cells_not_c = data_g[labels!=k]
                other_labels = labels_set.difference(set([k]))
                cluster_gene_mean = cells_c.mean() + eps
                max_other_mean = max(data_g[labels==x].mean() for x in other_labels)
                score = cluster_gene_mean/(max_other_mean+eps)
                scores[k].append((g, score))
    for k in labels_set:
        scores[k].sort(key=lambda x: x[1], reverse=True)
    return scores

def find_overexpressed_genes_weighted(data, w, eps=0):
    data_csc = sparse.csc_matrix(data)
    genes, cells = data.shape
    scores = csc_weighted_c_scores(
            data_csc.data,
            data_csc.indices,
            data_csc.indptr,
            w,
            cells,
            genes,
            eps)
    return scores

def pairwise_t(data, w_or_labels, eps=1.0, calc_pvals=True):
    """
    Computes pairwise t-test between all pairs of clusters and all genes.

    If calc_pvals is False, the pvals will be all 0.

    Returns:
        ratios, pvals - two arrays of shape (k, k, genes), where k is the number of clusters.
    """
    data_csc = sparse.csc_matrix(data)
    genes, cells = data.shape
    if len(w_or_labels.shape) == 2:
        scores, pvals = csc_weighted_t_test(
                data_csc.data,
                data_csc.indices,
                data_csc.indptr,
                w_or_labels,
                cells,
                genes,
                calc_pvals)
    else:
        labels_array = np.zeros(len(w_or_labels), dtype=int)
        labels_map = {}
        for i, l in enumerate(sorted(list(set(w_or_labels)))):
            labels_map[l] = i
        for i, l in enumerate(w_or_labels):
            labels_array[i] = labels_map[l]
        scores, pvals = csc_unweighted_t_test(
                data_csc.data,
                data_csc.indices,
                data_csc.indptr,
                labels_array,
                cells,
                genes,
                eps,
                calc_pvals)
        # TODO: re-map keys? or would that mess up the c-score calculation?
    return scores, pvals

def one_vs_rest_t(data, labels, eps=1.0, calc_pvals=True, test='t'):
    """
    Computes 1-vs-rest t-test for all clusters and genes.

    If calc_pvals is False, the pvals will be all 0.
    """
    data_csc = sparse.csc_matrix(data)
    genes, cells = data.shape
    labels_array = np.zeros(len(labels), dtype=int)
    # map from label names to indices
    labels_map = {}
    for i, l in enumerate(sorted(list(set(labels)))):
        labels_map[l] = i
    for i, l in enumerate(labels):
        labels_array[i] = labels_map[l]
    if test == 't':
        test_func = csc_unweighted_1_vs_rest_t_test
    elif test == 'u':
        test_func = csc_unweighted_1_vs_rest_rank_sum_test
    scores, pvals = test_func(
            data_csc.data,
            data_csc.indices,
            data_csc.indptr,
            labels_array,
            cells,
            genes,
            eps,
            calc_pvals)
    # map back to original label set?
    new_scores = {}
    new_pvals = {}
    for i, l in labels_map.items():
        new_scores[i] = scores[l]
        new_pvals[i] = pvals[l]
    return new_scores, new_pvals


def c_scores_from_t(scores, pvals):
    """
    Converts pairwise t-test results to c-scores.

    Args:
        scores (array): shape (k, k, genes)
        pvals (array): shape (k, k, genes)

    Returns:
        c_scores (dict): {k : [(gene, score)...]}
        c_pvals (dict): {k : [(gene, pval)...]}
    """
    c_scores, c_pvals = t_test_c_scores(scores, pvals)
    return c_scores, c_pvals

def separation_scores_from_t(scores, pvals):
    """
    Converts pairwise t-test results to separation scores.

    Args:
        scores (array): shape (k, k, genes)
        pvals (array): shape (k, k, genes)

    Returns:
        separation_scores (array): shape (k, k)
        best_genes (array): shape (k, k) - indicates the gene associated with
        each score.
    """
    separation_scores, best_genes = t_test_separation_scores(scores, pvals)
    return separation_scores, best_genes

def find_overexpressed_genes_m(m, eps=0):
    """
    Calculates the c-score for all genes/cell types, using the M matrix
    returned by uncurl.
    """
    genes = m.shape[0]
    K = m.shape[1]
    if eps==0:
        eps = 1e-4
    scores = {}
    for k in range(K):
        scores[k] = []
    for g in range(genes):
        for k in range(K):
            cluster_gene_mean = m[g,k] + eps
            max_other_mean = max(m[g,k2] for k2 in range(K) if k2!=k)
            score = cluster_gene_mean/(max_other_mean+eps)
            scores[k].append((g, score))
    for k in range(K):
        scores[k].sort(key=lambda x: x[1], reverse=True)
    return scores

def generate_permutations(data, k, real_clusters=None, n_perms=100):
    """
    Generates c-score distributions for each gene by randomly assigning cells
    to clusters and then calculating c-scores.

    Args:
        data (array): dense or sparse matrix of shape genes x cells
        k (int): number of clusters
        real_clusters (array, optional): real clusters to permute. If None, clusters will be randomly assigned based on k. Default: None
        n_perms (int): number of permutations to use

    Returns:
        A dict of {gene_id : [sorted (ascending) list of c-scores]}
    """
    gene_c_scores = {g:[] for g in range(data.shape[0])}
    # TODO permutations should be of the same shape as the input data (same number of cells?)
    for perm in range(n_perms):
        clusters_set = range(k)
        clusters = np.random.randint(0, k, data.shape[1])
        if real_clusters is not None:
            clusters_set = set(real_clusters)
            clusters = real_clusters.copy()
            np.random.shuffle(clusters)
        c_scores = find_overexpressed_genes(data, clusters)
        # for every gene, there should be at least one cluster for which
        # it has a c-score of at least 1.0.
        for c in clusters_set:
            cluster_scores = c_scores[c]
            for gene_id, cs in cluster_scores:
                if cs > 1.0:
                    gene_c_scores[gene_id].append(cs)
                else:
                    break
    for g in range(data.shape[0]):
        gene_c_scores[g].sort()
    return gene_c_scores

def calculate_permutation_pval(v, scores):
    """
    Given a sorted list of scores and a value, this calculates the value's position in the list of scores, and returns 1 - location(v)/len(scores).
    """
    # could binary search this but it's probably not worth it
    position = -1
    for i, s in enumerate(scores):
        if s >= v:
            break
        position = i
    return 1.0 - float(position+1)/len(scores)

def c_scores_to_pvals(scores, permutations):
    """
    Converts a dict of c-scores (output of find_overexpressed_genes) to a dict of
    p-values using the output of the permutation test.

    Args:
        scores: output of find_overexpressed_genes
        permutations: output of generate_permutations

    Returns:
        dict of k : list of (gene_id, pval) sorted by ascending pval
    """
    pvals = {}
    for k, sc in scores.items():
        pvals[k] = []
        for gene_id, c_score in sc:
            perms = permutations[gene_id]
            pval = 1.0
            if len(perms) > 1:
                pval = calculate_permutation_pval(c_score, perms)
            pvals[k].append((gene_id, pval))
        # this is a stable sort so it should preserve the c-score ordering?
        pvals[k].sort(key=lambda x: x[1])
    return pvals



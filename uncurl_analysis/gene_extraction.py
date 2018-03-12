from __future__ import print_function

import numpy as np
from scipy import sparse

from uncurl_analysis.sparse_gene_extraction import csc_overexpressed_genes

# TODO: efficient sparse implementation of find_overexpressed_genes?
def find_overexpressed_genes(data, labels, eps=0):
    """
    Returns a dict of label : list of (gene id, score) pairs for all genes,
    sorted by descending score.
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
        scores = csc_overexpressed_genes(
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

def generate_permutations(data, k, n_perms=100):
    """
    Generates c-score distributions for each gene by randomly assigning cells
    to clusters and then calculating c-scores.

    Returns:
        A dict of {gene_id : [sorted (ascending) list of c-scores]}
    """
    gene_c_scores = {g:[] for g in range(data.shape[0])}
    for perm in range(n_perms):
        clusters = np.random.randint(0, k, data.shape[1])
        c_scores = find_overexpressed_genes(data, clusters)
        # for every gene, there should be at least one cluster for which
        # it has a c-score of at least 1.0.
        for c in range(k):
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
    # could binary search this
    for i, s in enumerate(scores):
        if s >= v:
            break
    return 1.0 - float(i)/len(scores)

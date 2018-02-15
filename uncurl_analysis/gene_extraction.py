from __future__ import print_function

import numpy as np
from scipy import sparse

def find_overexpresed_genes(data, labels, eps=0):
    """
    Returns a dict of label : list of (gene id, score) pairs for all genes,
    sorted by descending score.
    """
    genes = data.shape[0]
    cells = data.shape[1]
    if eps==0:
        eps = 10.0/cells
    labels_set = set(labels)
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
        scores[k].sort(key=lambda x: x[1])
    return scores

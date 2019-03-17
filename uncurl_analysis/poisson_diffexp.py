# UNCURL-based differential expression
# assuming that M includes all genes... then we can do a Poisson test on MW?
# or some sort of weighted test using W?
# which Poisson test should we use? let's use the Log-Wald test bc of
# https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-5-S3-S1

import numpy as np
import scipy.stats
import scipy.special
from uncurl_analysis.poisson_tests import log_wald_poisson_test, uncurl_poisson_test_1_vs_rest

def uncurl_poisson_test_pairwise(m, w, mode='counts'):
    """
    Pairwise Poisson tests between all clusters.

    Returns:
        all_pvs, all_ratios: two arrays of shape (genes, k, k) indicating the
        p-values between two clusters.
    """
    # TODO
    clusters = w.argmax(0)
    genes = m.shape[0]
    cells = w.shape[1]
    all_pvs = []
    all_ratios = []
    cluster_cell_counts = None
    # TODO: how to use this mode...?
    if mode == 'counts':
        cell_counts = np.zeros(cells)
        cluster_cell_counts = np.zeros(w.shape[0])
        for i in range(cells):
            cell_counts[i] = np.sum(m.dot(w[:,i]))
        for k in range(w.shape[0]):
            cluster_cell_counts[k] = cell_counts[clusters==k].sum()
    for g in range(genes):
        pvs, ratios = uncurl_poisson_gene_pairwise(m, w, clusters, g, cluster_cell_counts)
        all_pvs.append(pvs)
        all_ratios.append(ratios)
    return np.array(all_pvs), np.array(all_ratios)

def uncurl_poisson_gene_pairwise(m, w, clusters, gene_index, cell_counts=None):
    """
    Calculates gene expression ratio and p-val for one gene and all pairs of clusters
    using the Log-Wald test.

    Args:
        m (genes x k array)
        w (k x cells array)
        clusters (array of cluster indices for all cells)
        gene_index (int): gene to test
        cell_counts (array of dim clusters): total cell counts for each cluster

    Returns:
        pvs, ratios: two arrays of shape (k, k)
    """
    n_clusters = w.shape[0]
    gene_matrix = np.dot(m[gene_index, :], w)
    cluster_pvs = np.zeros((n_clusters, n_clusters))
    cluster_ratios = np.zeros((n_clusters, n_clusters))
    all_cluster_cells = []
    for k in range(n_clusters):
        all_cluster_cells.append((clusters == k))
    for k1 in range(n_clusters):
        cluster_cells = all_cluster_cells[k1]
        in_cluster = gene_matrix[cluster_cells]
        if len(in_cluster) == 0:
            pass
        for k2 in range(n_clusters):
            cluster_cells_2 = all_cluster_cells[k2]
            in_cluster_2 = gene_matrix[cluster_cells_2]
            if cell_counts is not None:
                counts1 = cell_counts[k1]
                counts2 = cell_counts[k2]
                pv, ratio = log_wald_poisson_test(in_cluster, in_cluster_2, counts1, counts2)
            else:
                pv, ratio = log_wald_poisson_test(in_cluster, in_cluster_2)
            cluster_pvs[k1, k2] = pv
            cluster_ratios[k1, k2] = ratio
    return cluster_pvs, cluster_ratios

def c_scores_from_pv_ratios(all_pvs, all_ratios):
    """
    Computes c-scores based on the results of the pairwise tests.

    For each gene, we want to find the pair of clusters c1, c2
    such that all_ratios[g, c1, c2] > 1
    and all_ratios[g, c1, c2] < all_ratios[g, c1, c_i] for all i

    All genes only have one cluster that has a c-score > 1.

    Returns:
        c_clusters, c_scores, c_pvals:
    """
    genes = all_pvs.shape[0]
    clusters = all_pvs.shape[1]
    c_scores = {k : [] for k in range(clusters)}
    for g in range(genes):
        best_cluster = 0
        lowest_ratio = 1e10
        best_pval = 1.0
        for k1 in range(clusters):
            lowest_ratio = 1e10
            best_k2 = None
            for k2 in range(clusters):
                if k2 == k1:
                    continue
                ratio = all_ratios[g, k1, k2]
                if ratio < lowest_ratio:
                    lowest_ratio = ratio
                    best_k2 = k2
            if lowest_ratio > 1.0:
                lowest_ratio = ratio
                best_cluster = k1
                best_pval = all_pvs[g, k1, best_k2]
                break
        c_scores[best_cluster].append((g, lowest_ratio, best_pval))
    # TODO: reshape this...
    # create three k x g arrays: cluster_genes, cluster_ratios, and cluster_pvals.
    # cluster_genes has the top c-score gene indices for each cluster, and -1 for all indices that don't have a top gene.
    # cluster_ratios and cluster_pvals have the ratios/pvals for each cluster/gene.
    cluster_genes = np.zeros((clusters, genes)) - 1
    cluster_ratios = np.zeros((clusters, genes)) - 1
    cluster_pvals = np.zeros((clusters, genes)) - 1
    for k in range(clusters):
        scores = c_scores[k]
        scores.sort()
    return c_clusters, c_scores, c_pvals

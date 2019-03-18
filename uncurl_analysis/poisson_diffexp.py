# UNCURL-based differential expression
# assuming that M includes all genes... then we can do a Poisson test on MW?
# or some sort of weighted test using W?
# which Poisson test should we use? let's use the Log-Wald test bc of
# https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-5-S3-S1

import numpy as np
import scipy.stats
import scipy.special
from uncurl_analysis.poisson_tests import log_wald_poisson_test, uncurl_poisson_test_1_vs_rest, uncurl_poisson_test_pairwise

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

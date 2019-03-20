# UNCURL-based differential expression
# assuming that M includes all genes... then we can do a Poisson test on MW?
# or some sort of weighted test using W?
# which Poisson test should we use? let's use the Log-Wald test bc of
# https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-5-S3-S1

import numpy as np
from uncurl_analysis.poisson_tests import log_wald_poisson_test, uncurl_test_1_vs_rest, uncurl_test_pairwise

# TODO: poisson test with pre-defined group?
def poisson_test_known_groups(data, groups, test_mode='pairwise',
        test='poisson',
        mode='counts', m=None, w=None):
    """
    Given a list of known groups, one for each cell: this runs uncurl-based Poisson diffexp
    on.

    Args:
        data (dense or sparse array): shape is (genes, cells)
        groups (1d array or list): int or string for each cell, indicating groups
        test_mode (str): 'pairwise' or '1_vs_rest'. Default: 'pairwise'
        test (str): 'poisson' or 't' (TODO)
        mode (str): 'counts' or 'cells'

    Returns:
        all_pvs, all_ratios, clusters_to_groups
        all_pvs and all_ratios are arrays. If mode is 'pairwise', arrays are of shape (genes, k, k).
        Otherwise, arrays are of shape (genes, k).
        clusters_to_groups is a dict of cluster id to group name.
    """
    import uncurl
    groups_list = sorted(list(set(groups)))
    groups_to_clusters = {x: i for i, x in enumerate(groups_list)}
    clusters_to_groups = {i: x for i, x in enumerate(groups_list)}
    clusters = np.array([groups_to_clusters[i] for i in groups])
    # initialize uncurl...
    # should we select a gene subset to do uncurl??? nah...
    if m is None and w is None:
        m, w, ll = uncurl.poisson_estimate_state(data, len(groups_list), init_weights=clusters,
                max_assign_weight=1.0, run_w_first=False, max_iters=10, inner_max_iters=100)
    # now we can use poisson test
    if test_mode == 'pairwise':
        all_pvs, all_ratios = uncurl_test_pairwise(m, w, mode=mode, test=test, clusters=clusters)
        return all_pvs, all_ratios, clusters_to_groups
    else:
        all_pvs, all_ratios = uncurl_test_1_vs_rest(m, w, mode=mode, test=test, clusters=clusters)
        return all_pvs, all_ratios, clusters_to_groups


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
        # sort by ratio
        c_scores[best_cluster].sort(key=lambda x: x[1], reverse=True)
        for i, sc in enumerate(c_scores[best_cluster]):
            g, lowest_ratio, best_pval = sc
            cluster_genes[k, i] = g
            cluster_ratios[k, i] = lowest_ratio
            cluster_pvals[k, i] = best_pval
    return c_scores

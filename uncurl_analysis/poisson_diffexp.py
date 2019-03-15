# UNCURL-based differential expression
# assuming that M includes all genes... then we can do a Poisson test on MW?
# or some sort of weighted test using W?
# which Poisson test should we use? let's use the Log-Wald test bc of
# https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-5-S3-S1

import numpy as np
import scipy.stats
import scipy.special

def log_wald_poisson_test(data1, data2, counts1=0, counts2=0):
    """
    Given two vectors, this calculates the log-Wald test between data1 and
    data2, with data2 being the null model. Essentially this tests for
    overexpression in data1 compared to data2.

    Source: https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.1949

    Formula: W3 = [ln(X1/X0) - ln(d)]*sqrt(1/X0 + 1/X1)
    Where X1 and X0 are the counts observed in the two conditions,
    and d = t1/t0, the times taken for the two conditions.

    Here, t1 and t0 are either the total number of cells, or the total
    read counts for those cells. I'm not sure which is better...

    The p-value is calculated by assuming that W3 is distributed by a
    standard normal distribution, so W3 is essentially a z-score.

    Args:
        data1 (np array)
        data2 (np array)
        counts1 (number): total counts for condition 1. Default: cell count
        counts2 (number): total counts for condition 2. Default: cell count

    """
    if counts1 == 0:
        counts1 = len(data1)
    if counts2 == 0:
        counts2 = len(data2)
    X1 = data1.sum()
    X0 = data2.sum()
    if X1 == 0 and X0 == 0:
        return 0.5, 1.0
    # add a 'pseudocount' of 1
    X1 += 1.0
    X0 += 1.0
    counts1 += 1
    counts2 += 1
    d = float(counts1)/float(counts2)
    W3 = (np.log(X1/X0) - np.log(d))*np.sqrt(1.0/X0 + 1.0/X1)
    if np.isnan(W3):
        return 1.0, 0.0
    # normal CDF
    # ndtr is much more computationally efficient than norm.cdf
    # on one test case, using norm.cdf took ~13 seconds
    # while using ndtr took ~4 seconds
    pv = 1 - scipy.special.ndtr(W3)
    #pv = 1 - scipy.stats.norm.cdf(W3)
    ratio = X1/(X0*d)
    return pv, ratio

def uncurl_poisson_test_1_vs_rest(m, w, mode='counts'):
    """
    Calculates 1-vs-rest ratios and p-values for all genes.

    mode can be 'cells' or 'counts'. In 'cells', the observation time in the
    Poisson test is the number of cells. In 'counts', the observation time
    is the total reads in the cells.

    Returns two arrays: all_pvs and all_ratios, of shape (genes, clusters).
    """
    # TODO: make this more efficient...
    clusters = w.argmax(0)
    genes = m.shape[0]
    cells = w.shape[1]
    all_pvs = []
    all_ratios = []
    cluster_cell_counts = None
    if mode == 'counts':
        cell_counts = np.zeros(cells)
        cluster_cell_counts = np.zeros(w.shape[0])
        for i in range(cells):
            cell_counts[i] = np.sum(m.dot(w[:,i]))
        for k in range(w.shape[0]):
            cluster_cell_counts[k] = cell_counts[clusters==k].sum()
    for g in range(genes):
        cluster_pvs, cluster_ratios = uncurl_poisson_gene_1_vs_rest(m, w, clusters, g, cell_counts=cluster_cell_counts)
        all_pvs.append(cluster_pvs)
        all_ratios.append(cluster_ratios)
    return np.array(all_pvs), np.array(all_ratios)


def uncurl_poisson_gene_1_vs_rest(m, w, clusters, gene_index, cell_counts=None):
    """
    Calculates 1-vs-rest ratio and p-val for one gene and all clusters
    using the Log-Wald test.

    Args:
        m (genes x k array)
        w (k x cells array)
        clusters (array of cluster indices for all cells)
        gene_index (int): gene to test
        cell_counts (array of dim clusters): total cell counts for each cluster
    """
    n_clusters = w.shape[0]
    gene_matrix = np.dot(m[gene_index, :], w)
    cluster_pvs = []
    cluster_ratios = []
    for k in range(n_clusters):
        cluster_cells = (clusters == k)
        in_cluster = gene_matrix[cluster_cells]
        if len(in_cluster) == 0:
            cluster_pvs.append(0.0)
            cluster_ratios.append(0.0)
        else:
            not_in_cluster = gene_matrix[~cluster_cells]
            if cell_counts is not None:
                counts1 = cell_counts[k]
                counts2 = cell_counts[:k].sum() + cell_counts[k+1:].sum()
                pv, ratio = log_wald_poisson_test(in_cluster, not_in_cluster, counts1, counts2)
            else:
                pv, ratio = log_wald_poisson_test(in_cluster, not_in_cluster)
            cluster_pvs.append(pv)
            cluster_ratios.append(ratio)
    return cluster_pvs, cluster_ratios


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
        c_scores, c_pvals: two arrays of shape (g, k)
    """
    genes = all_pvs.shape[0]
    clusters = all_pvs.shape[1]
    c_scores = []
    c_pvals = []
    for g in range(genes):
        for k1 in range(clusters):
            for k2 in range(clusters):
                pass

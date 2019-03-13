# UNCURL-based differential expression
# assuming that M includes all genes... then we can do a Poisson test on MW?
# or some sort of weighted test using W?
# which Poisson test should we use? let's use the Log-Wald test bc of
# https://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-5-S3-S1

import numpy as np
import scipy.stats

def log_wald_poisson_test(data1, data2, counts1=0, counts2=0):
    """
    Given two vectors, this calculates the log-Wald test between data1 and
    data2, with data2 being the null model.

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
    d = float(counts1)/float(counts2)
    W3 = (np.log(X1/X0) - np.log(d))*np.sqrt(1.0/X0 + 1.0/X1)
    pv = 1 - scipy.stats.norm.cdf(W3)
    ratio = X1/(X0*d)
    return pv, ratio

def uncurl_poisson_test_1_vs_rest(m, w):
    """
    Calculates 1-vs-rest ratios and p-values for all genes.
    """
    # TODO
    clusters = w.argmax(0)

def uncurl_poisson_gene_1_vs_rest(m, w, clusters, gene_index):
    """
    Calculates 1-vs-rest ratio and p-val for one gene and all clusters
    using the Log-Wald test.
    """
    n_clusters = w.shape[0]
    gene_matrix = np.dot(m[gene_index, :], w)
    for k in range(n_clusters):
        pass

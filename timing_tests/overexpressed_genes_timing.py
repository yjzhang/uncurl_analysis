import time

from uncurl_analysis.gene_extraction import pairwise_t, c_scores_from_t, separation_scores_from_t

from scipy import sparse
from scipy.io import loadmat
import numpy as np

if __name__ == '__main__':

    # TODO: generate synthetic datasets for testing,
    # where top cluster-specific genes are known
    dat = loadmat('data/10x_pooled_400.mat')
    # take 5000 genes arbitrarily?
    data = sparse.csc_matrix(dat['data'])
    labs = dat['labels'].flatten()

    t0 = time.time()
    t_test_scores, t_test_p_vals = pairwise_t(data, labs)
    print('pairwise_t time: ' + str(time.time() - t0))
    K = len(set(labs))
    genes, cells = data.shape
    t = time.time()
    t0 = time.time()
    c_scores, c_pvals = c_scores_from_t(t_test_scores, t_test_p_vals)
    print('c_scores_from_t time: ' + str(time.time() - t0))
    for k in range(K):
        pvals = np.array([x[1] for x in c_pvals[k]])
        cscores = np.array([x[1] for x in c_scores[k]])
    # test separation score
    separation_scores, best_genes = separation_scores_from_t(t_test_scores, t_test_p_vals)

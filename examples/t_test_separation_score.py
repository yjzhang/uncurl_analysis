# t-test, separation score
from __future__ import print_function
import time

import line_profiler

import numpy as np
import scipy.io
from scipy import sparse
import uncurl

from uncurl_analysis import gene_extraction
from uncurl_analysis.sparse_gene_extraction import csc_unweighted_t_test

# step 1: load data
dat = scipy.io.loadmat('data/10x_pooled_400.mat')
data = sparse.csc_matrix(dat['data'])
labs = dat['labels'].flatten()
genes, cells = data.shape

labels_array = np.zeros(len(labs), dtype=int)
labels_map = {}
for i, l in enumerate(sorted(list(set(labs)))):
    labels_map[l] = i
for i, l in enumerate(labs):
    labels_array[i] = labels_map[l]


# profile csc_unweighted_t_test
func = csc_unweighted_t_test
profile = line_profiler.LineProfiler(func)
profile.runcall(func,
        data.data,
        data.indices,
        data.indptr,
        labels_array,
        cells,
        genes)
profile.print_stats()

# test weighted t test
genes = uncurl.max_variance_genes(data, 5, 0.2)
data_subset = data[genes,:]
M, W, ll = uncurl.run_state_estimation(data_subset, 8, max_iters=20, inner_max_iters=50)
t_scores, t_pvals = gene_extraction.pairwise_t(data, W)

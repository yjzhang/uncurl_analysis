
import time

import numpy as np
from uncurl_analysis import poisson_diffexp
import scipy.io 
import uncurl 

mat = scipy.io.loadmat('data/10x_pooled_400.mat')
data = mat['data']
# do uncurl, followed by update_m 
selected_genes = uncurl.max_variance_genes(data)
data_subset = data[selected_genes, :]
m, w, ll = uncurl.run_state_estimation(data_subset, 8, max_iters=20, inner_max_iters=50)
m = uncurl.update_m(data, m, w, selected_genes)

t0 = time.time()
all_pvs, all_ratios = poisson_diffexp.uncurl_poisson_test_1_vs_rest(m, w, mode='counts')
print('diffexp time: ', time.time() - t0)

t0 = time.time()
all_pvs_2, all_ratios_2 = poisson_diffexp.uncurl_poisson_test_pairwise(m, w, mode='counts')
print('pairwise diffexp time: ', time.time() - t0)

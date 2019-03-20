
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

# test on simulated data
# plotting mw
import matplotlib.pyplot as plt
mw = m.dot(w)
clusters = w.argmax(0)
data_array = data.toarray()
plt.figure(dpi=150)
plt.scatter(data_array[:, clusters==0].mean(1), data_array[:, clusters==0].var(1), label='raw data')
plt.scatter(mw[:, clusters==0].mean(1), mw[:, clusters==0].var(1), label='uncurl')
plt.xlabel('mean')
plt.ylabel('variance')
plt.title('Variance vs Mean for all genes over all cells in cluster 0')
plt.legend()
plt.savefig('examples/uncurl_mean_vs_variance_cluster_0.png')

plt.figure(dpi=150)
plt.scatter(data_array.mean(1), data_array.var(1), label='raw data')
plt.scatter(mw.mean(1), mw.var(1), label='uncurl')
plt.xlabel('mean')
plt.ylabel('variance')
plt.title('Variance vs Mean for all genes over all cells')
plt.legend()
plt.savefig('examples/uncurl_mean_vs_variance_all_cells.png')



top_genes = mw.mean(1).argsort()[::-1]

plt.figure(dpi=150)
plt.hist(mw[top_genes[0], clusters==0], normed=True, bins=20, alpha=0.5, label='cluster 0')
plt.hist(mw[top_genes[0], :], normed=True, bins=20, alpha=0.5, label='all cells')
plt.title('Distribution of MW for gene {0}'.format(top_genes[0]))
plt.xlabel('Processed expression')
plt.legend()
plt.savefig('examples/uncurl_gene_0_distribution.png')

plt.figure(dpi=150)
plt.hist(data_array[top_genes[0], clusters==0], normed=True, bins=20, alpha=0.5, label='cluster 0')
plt.hist(data_array[top_genes[0], :], normed=True, bins=20, alpha=0.5, label='all cells')
plt.title('Distribution of raw read counts for gene {0}'.format(top_genes[0]))
plt.xlabel('Raw expression')
plt.savefig('examples/raw_gene_0_distribution.png')

plt.hist(mw[top_genes[100], clusters==0], normed=True, bins=20, alpha=0.5)
plt.hist(mw[top_genes[100], :], normed=True, bins=20, alpha=0.5)
plt.show()

plt.hist(mw[top_genes[1], clusters==0], bins=20)
plt.hist(mw[top_genes[2], clusters==0], bins=20)
# select a random gene...

# we have two clusters and two genes. Gene 0 is upregulated in cluster 1, gene 1 is upregulated in cluster 0.
#from uncurl import simulation
#data, clusters = simulation.generate_poisson_data(np.array([[1.0, 100], [100, 1.0]]), 100)

# the correct response: pvs[0,1,0] is significant and pvs[1,0,1] is significant
#pvs, ratios, _ = poisson_diffexp.poisson_test_known_groups(data, clusters, test_mode='1_vs_rest')



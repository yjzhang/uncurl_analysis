# testing various clustering methods
import numpy as np
import scipy.io
from uncurl_analysis import clustering_methods
from  sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import uncurl


# 1. load data
data = scipy.io.loadmat('data/10x_pooled_400.mat')

data_csc = data['data']
labels = data['labels'].flatten()
gene_names = data['gene_names']

# 2. gene selection
genes = uncurl.max_variance_genes(data_csc)
data_subset = data_csc[genes,:]
gene_names_subset = gene_names[genes]

# 3. run uncurl
m, w, ll = uncurl.run_state_estimation(data_subset, 8)
print('nmi basic: ' + str(nmi(labels, w.argmax(0))))


# 4. run clustering
for metric in ['euclidean', 'cosine']:
    for n_neighbors in [10, 15, 20]:
        print('n_neighbors: ', n_neighbors, ' metric: ', metric)
        w_graph = clustering_methods.create_graph(w.T, n_neighbors=n_neighbors, metric=metric)
        clusters = clustering_methods.run_leiden(w_graph)
        print('nmi leiden: ' + str(nmi(labels, clusters)))
        clusters_louvain = clustering_methods.run_louvain(w_graph)
        print('nmi louvain: ' + str(nmi(labels, clusters_louvain)))

# 5. try running clustering w/o uncurl
clustering_result = clustering_methods.baseline_cluster(data_subset)
# TODO: figure out cuts
print('nmi leiden baseline: ' + str(nmi(labels, clustering_result.membership)))


# try other dataset (Zeisel 2015)

import scipy.sparse
# 1. load data
data = scipy.io.loadmat('data/GSE60361_dat.mat')
labels = data['ActLabs'].flatten()
data_csc = scipy.sparse.csc_matrix(data['Dat'])
# 2. gene selection
genes = uncurl.max_variance_genes(data_csc)
data_subset = data_csc[genes,:]
# 3. run uncurl
m, w, ll = uncurl.run_state_estimation(data_subset, 7)
print('nmi basic: ' + str(nmi(labels, w.argmax(0))))

# 4. run clustering
for metric in ['euclidean', 'cosine']:
    for n_neighbors in [10, 15, 20]:
        print('n_neighbors: ', n_neighbors, ' metric: ', metric)
        w_graph = clustering_methods.create_graph(w.T, n_neighbors=n_neighbors, metric=metric)
        clusters = clustering_methods.run_leiden(w_graph)
        print('nmi leiden: ' + str(nmi(labels, clusters)))
        clusters_louvain = clustering_methods.run_louvain(w_graph)
        print('nmi louvain: ' + str(nmi(labels, clusters_louvain)))

# 5. try running clustering w/o uncurl
clustering_result = clustering_methods.baseline_cluster(data_subset)
# TODO: figure out cuts
print('nmi leiden baseline: ' + str(nmi(labels, clustering_result.membership)))



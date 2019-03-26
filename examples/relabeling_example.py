# re-initializing uncurl 

# splitting/merging clusters

import numpy as np
import scipy.io
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

# 4. basic cluster analysis

# run mds
m_mds = uncurl.dim_reduce(m, w, 2) # 8 x 2 array

# 5. building a distance matrix between clusters, find closest pair

# create distance matrix
# find the min distance between two cluster pairs
distance_matrix = np.zeros((8,8))
min_distance_pair = (0,0)
min_distance = 1e10
for i in range(8):
    for j in range(8):
        distance_matrix[i,j] = uncurl.sparse_utils.poisson_dist(m[:,i],
                m[:,j])
        if i != j and distance_matrix[i,j] < min_distance:
            min_distance = distance_matrix[i,j]
            min_distance_pair = (i,j)
            min_distance = distance_matrix[i,j]

# 6. run split_cluster - split the largest cluster
from collections import Counter
from uncurl_analysis import relabeling

clusters = w.argmax(0)
cluster_counts = Counter(clusters)
top_cluster, top_count = cluster_counts.most_common()[0]

m_split, w_split = relabeling.split_cluster(data_subset, m, w, top_cluster)
print('nmi after splitting the largest cluster: ' + str(nmi(labels, w_split.argmax(0))))

# 7. merge the min distance pair
m_merge, w_merge = relabeling.merge_clusters(data_subset, m, w,
        min_distance_pair)

print('nmi after merging the closest pairs: ' + str(nmi(labels, w_merge.argmax(0))))

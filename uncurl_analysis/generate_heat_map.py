# TODO: generate heat map of highest expressed genes in each cluster???

# figure 1:
# 1. partition genes into the clusters they're overexpressed in (cluster with test statistic > 1).
# 2. generate a heat map of gene expresson by cell, where columns are cells sorted by cluster, and rows are genes sorted by the test statistic for each cluster, where the genes with test statistic > 1 for each cluster are partitioned.
# 3. generate heat map for both before and after pre-processing. Row-normalized.


# figure 2:
# generate a new sorted list of genes per cluster by test statistic, after pre-processing
# do a gene expression plot
# 

import os

import numpy as np
import uncurl

# try using seaborn
import matplotlib.pyplot as plt
import seaborn as sns

from h5_10x_analysis import get_matrix_from_h5

genes_per_cluster = 10

# load gene names, test statistic
gene_diffexp_dir = 'diffexp_k10_means_eps_1e-3/'
#gene_diffexp_dir = 'diffexp_tsne_k10_means_eps_1e-3/'
#gene_diffexp_dir = 'diffexp_magic_tsne_k10_means_eps_1e-3/'
k = 10
gene_names = []
test_statistics = []
# cluster_genes contains the gene names assigned to each cluster, sorted by test statistic.
cluster_genes = []
for c in range(k):
    test_statistic_file = gene_diffexp_dir + 'poisson_test_statistic_sorted_c{0}.txt'.format(c)
    gene_names_file = gene_diffexp_dir + 'gene_names_c{0}.txt'.format(c)
    gene_names.append(np.loadtxt(gene_names_file, dtype=str))
    test_statistics.append(np.loadtxt(test_statistic_file))
    high_test_statistics = (test_statistics[-1]>1)
    #cluster_genes.append(gene_names[-1][high_test_statistics])
    cluster_genes.append(gene_names[-1][:genes_per_cluster])

# load cells, mw 
data_file = '1M_neurons_neuron20k.h5'
genome = "mm10"
gene_bc_matrix = get_matrix_from_h5(data_file, genome)

gene_names = gene_bc_matrix.gene_names

X1 = gene_bc_matrix.matrix
X1 = X1.tocsc()

frac = 1.0
genes_subset = uncurl.max_variance_genes(X1, 5, frac)
X1 = X1[genes_subset,:]
gene_names_subset = gene_names[genes_subset]

# load mw
k = 10
n_genes = len(genes_subset)
m = np.loadtxt('mw/m_20000_cells_{1}_genes_k{0}.txt'.format(k, n_genes))
w = np.loadtxt('mw/w_20000_cells_{1}_genes_k{0}.txt'.format(k, n_genes))

# match gene names to rows in data matrix
gene_names_to_indices = {n:i for i,n in enumerate(gene_names_subset)}
# map gene names in cluster_genes to indices
gene_indices_ordered = []
for c in range(k):
    gene_indices = [gene_names_to_indices[n] for n in cluster_genes[c]]
    gene_indices_ordered.append(gene_indices)
gene_indices = reduce(lambda x,y: x+y, gene_indices_ordered)
gene_names_clusters = gene_names_subset[gene_indices]

cumulative_gene_counts = reduce(lambda x,y: x + [x[-1] + len(y)], gene_indices_ordered, [0])

# sort cells by id into cluster
# TODO: get cluster labels from tsne
#cluster_labels = np.loadtxt('LogData_TSVD_50_TSNE_k10_19763_genes_cluster_labels.txt').astype(int)
#cluster_labels = np.loadtxt('magic_k10_4476_genes_cluster_labels.txt').astype(int)
cluster_labels = w.argmax(0)
cell_indices_ordered = [[] for i in range(k)]
#cells = w.shape[1]
cells = X1.shape[1]
for i in range(cells):
    cell_indices_ordered[cluster_labels[i]].append(i)
# sort cell_indices within each cluster by descending W[cluster]
for c in range(k):
    w_subset = w[c, w.argmax(0)==c]
    w_sorted_indices = w_subset.argsort()[::-1]
    cell_subset = np.array(cell_indices_ordered[c])
    cell_indices_ordered[c] = list(cell_subset[w_sorted_indices])
cell_indices = reduce(lambda x,y: x+y, cell_indices_ordered)

cumulative_cell_counts = reduce(lambda x,y: x + [x[-1] + len(y)], cell_indices_ordered, [0])

# re-order data matrix
X1_full = X1.toarray().astype(float)
X1_reordered = X1_full[np.ix_(gene_indices, cell_indices)]

# normalize by row
X1_reordered = X1_reordered/X1_reordered.max(1, keepdims=True)

# plot heatmap using seaborn
fig = plt.figure(figsize=(30,20))
ax = sns.heatmap(X1_reordered, cmap='inferno_r', yticklabels=gene_names_clusters)
ax.set_xlabel('Cells', fontsize=30)
ax.set_ylabel('Genes', fontsize=30)

# add lines
ax.hlines(cumulative_gene_counts, *ax.get_xlim())
ax.vlines(cumulative_cell_counts, *ax.get_ylim())

#plt.savefig('baseline_cluster_heatmap_eps_1e-3_cell_sorted_gene_labels.png')
plt.savefig('baseline_cluster_heatmap_eps_1e-3.png')

# plot heatmap for mw
plt.clf()

mw = m.dot(w)
mw_reordered = mw[np.ix_(gene_indices, cell_indices)]
mw_reordered = mw_reordered/mw_reordered.max(1, keepdims=True)

fig = plt.figure(figsize=(30,20))
ax = sns.heatmap(mw_reordered, cmap='inferno_r', yticklabels=gene_names_clusters)
ax.hlines(cumulative_gene_counts, *ax.get_xlim())
ax.vlines(cumulative_cell_counts, *ax.get_ylim())
ax.set_xlabel('Cells', fontsize=30)
ax.set_ylabel('Genes', fontsize=30)

plt.savefig('mw_cluster_heatmap_eps_1e-3.png')

# plot heatmap for magic
plt.clf()
magic_data = np.loadtxt('magic_20k_cells_4476_genes.txt')
magic_reordered = magic_data[np.ix_(gene_indices, cell_indices)]
magic_reordered = magic_reordered/magic_reordered.max(1, keepdims=True)

fig = plt.figure(figsize=(30,20))
ax = sns.heatmap(magic_reordered, cmap='inferno_r', yticklabels=gene_names_clusters)
ax.hlines(cumulative_gene_counts, *ax.get_xlim())
ax.vlines(cumulative_cell_counts, *ax.get_ylim())
ax.set_xlabel('Cells', fontsize=30)
ax.set_ylabel('Genes', fontsize=30)

plt.savefig('magic_cluster_heatmap_eps_1e-3.png')

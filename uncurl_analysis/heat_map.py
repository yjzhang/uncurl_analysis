# TODO: generate heat map of highest expressed genes in each cluster???

# figure 1:
# 1. partition genes into the clusters they're overexpressed in (cluster with test statistic > 1).
# 2. generate a heat map of gene expresson by cell, where columns are cells sorted by cluster, and rows are genes sorted by the test statistic for each cluster, where the genes with test statistic > 1 for each cluster are partitioned.
# 3. generate heat map for both before and after pre-processing. Row-normalized.


# figure 2:
# generate a new sorted list of genes per cluster by test statistic, after pre-processing
# do a gene expression plot
# 


import numpy as np

# try using seaborn
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heat_map(data, W, top_genes, genes_per_cluster=10, gene_names=None, output_filename=None, figsize=(30,20), **plot_args):
    """
    Args:
        data: genes x cells array, dense or sparse
        W (array): output of uncurl - shape is (k, cells)
        top_genes (dict): map from cluster labels to lists of (gene_id, score) sorted by descending score
        genes_per_cluster (int): number of top genes to display per cluster. default: 10
        gene_names: array or dict that maps gene ids to gene names. If None, just uses gene indices as gene names.
        output_filename (string): file to write to. Default: None
        figsize (tuple): default: (30, 20)
        **plot_args: kwargs to pass to seaborn
    """
    cluster_labels = W.argmax(0)
    k = W.shape[0]
    cells = data.shape[1]
    cell_indices_ordered = [[] for i in range(k)]
    gene_indices_ordered = [[] for i in range(k)]
    for i in range(cells):
        cell_indices_ordered[cluster_labels[i]].append(i)
    # sort cell_indices within each cluster by descending W[cluster]
    for c in range(k):
        w_subset = W[c, W.argmax(0)==c]
        w_sorted_indices = w_subset.argsort()[::-1]
        cell_subset = np.array(cell_indices_ordered[c])
        cell_indices_ordered[c] = list(cell_subset[w_sorted_indices])
        gene_indices_ordered[c] = [x[0] for x in top_genes[c][:10]]
    cell_indices = reduce(lambda x,y: x+y, cell_indices_ordered)
    gene_indices = reduce(lambda x,y: x+y, gene_indices_ordered)

    cumulative_cell_counts = reduce(lambda x,y: x + [x[-1] + len(y)], cell_indices_ordered, [0])
    cumulative_gene_counts = reduce(lambda x,y: x + [x[-1] + len(y)], gene_indices_ordered, [0])
    gene_names_clusters = gene_indices
    if gene_names is not None:
        gene_names_clusters = [gene_names[i] for i in gene_indices]

    # re-order data matrix
    #X1_full = X1.toarray().astype(float)
    X1_reordered = data[np.ix_(gene_indices, cell_indices)]

    # normalize by row
    X1_reordered = X1_reordered/X1_reordered.max(1, keepdims=True)

    # plot heatmap using seaborn
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(X1_reordered, cmap='inferno_r', yticklabels=gene_names_clusters)
    ax.set_xlabel('Cells', fontsize=30)
    ax.set_ylabel('Genes', fontsize=30)

    # add lines
    ax.hlines(cumulative_gene_counts, *ax.get_xlim())
    ax.vlines(cumulative_cell_counts, *ax.get_ylim())

    #plt.savefig('baseline_cluster_heatmap_eps_1e-3_cell_sorted_gene_labels.png')
    if output_filename is not None:
        plt.savefig(output_filename)
    return fig

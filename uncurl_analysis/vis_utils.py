
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import sparse

tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def plot_gene_markers(tsne, markers, gene_names, data, filename, save_individually=False, figsize=None, **cluster_params):
    genes_in_markers = []
    for gene in markers:
        if gene not in gene_names:
            continue
        else:
            genes_in_markers.append(gene)
    n_rows = len(genes_in_markers)
    if figsize is None and not save_individually:
        figsize = (12, n_rows*8)
    if not save_individually:
        plt.figure(figsize=figsize)
    for i, gene in enumerate(genes_in_markers):
        if save_individually:
            plt.figure(figsize=figsize)
        else:
            plt.subplot(n_rows, 1, i+1)
        gene_id = np.argwhere(gene_names==gene)[0,0]
        print(gene_id)
        expression = data[gene_id,:].astype(float)
        if sparse.issparse(data):
            expression = expression.toarray()
        exp_norm = expression.flatten()
        exp_norm = exp_norm/exp_norm.max()
        rgba_colors = np.zeros((data.shape[1], 4))
        rgba_colors[:,0] = 1.0
        rgba_colors[:,3] = exp_norm
        plt.scatter(tsne[0,:], tsne[1,:], color=rgba_colors, **cluster_params)
        plt.title(gene)
        if save_individually:
            filename_parts = filename.split('.')
            filename_parts[0] += '_' + gene
            plt.savefig('.'.join(filename_parts))
            plt.close()
    if not save_individually:
        plt.savefig(filename)

def plot_gene_markers_mw(tsne, markers, gene_names, data, m, w, filename, save_individually=False, figsize=None, **cluster_params):
    genes_in_markers = []
    for gene in markers:
        if gene not in gene_names:
            continue
        else:
            genes_in_markers.append(gene)
    print(genes_in_markers)
    n_rows = len(genes_in_markers)
    if figsize is None and not save_individually:
        figsize = (20, n_rows*8)
    if not save_individually:
        plt.figure(figsize=figsize)
    for i, gene in enumerate(genes_in_markers):
        if save_individually:
            plt.figure(figsize=figsize)
        gene_id = np.argwhere(gene_names==gene)[0,0]
        print(gene_id)
        expression_1 = data[gene_id,:].astype(float)
        if sparse.issparse(data):
            expression_1 = expression_1.toarray()
        expression_2 = m[gene_id,:].dot(w)
        rgba_colors = np.zeros((data.shape[1], 4))
        rgba_colors[:,0] = 1.0
        rgba_colors[:,3] = expression_1/expression_1.max()
        if save_individually:
            plt.subplot(1, 2, 1)
        else:
            plt.subplot(n_rows, 2, 2*i+1)
        plt.scatter(tsne[0,:], tsne[1,:], color=rgba_colors, **cluster_params)
        plt.title(gene)
        rgba_colors[:,3] = expression_2/expression_2.max()
        if save_individually:
            plt.subplot(1, 2, 2)
        else:
            plt.subplot(n_rows, 2, 2*i+2)
        plt.scatter(tsne[0,:], tsne[1,:], color=rgba_colors, **cluster_params)
        plt.title(gene)
        if save_individually:
            filename_parts = filename.split('.')
            filename_parts[0] += '_' + gene
            plt.savefig('.'.join(filename_parts))
            plt.close()
    if not save_individually:
        plt.savefig(filename)

def plot_gene_markers_avg(tsne, markers, gene_names, data, filename, figsize=None, cluster_color_id=0, **cluster_params):
    """
    Creates a plot with the average expression of all the listed genes (expression of each gene is normalized).
    """
    genes_in_markers = []
    for gene in markers:
        if gene not in gene_names:
            continue
        else:
            genes_in_markers.append(gene)
    plt.figure(figsize=figsize)
    plt.clf()
    exp_means = np.zeros(data.shape[1])
    color_rgb = colors.ColorConverter.to_rgba(tab_colors[cluster_color_id])
    # TODO: increase contrast somehow
    print(tab_colors[cluster_color_id])
    print(color_rgb)
    #color_rgb = list(color_rgb)
    #for i in range(3):
    #    color_rgb[i]*=2
    #    if color_rgb[i] > 1:
    #        color_rgb[i] = 1
    #color_rgb = tuple(color_rgb)
    for i, gene in enumerate(genes_in_markers):
        gene_id = np.argwhere(gene_names==gene)[0,0]
        print(gene_id)
        expression = data[gene_id,:].astype(float)
        if sparse.issparse(data):
            expression = expression.toarray().flatten()
        exp_norm = expression
        # make colors look more "binary"?
        exp_norm[exp_norm>exp_norm.max()/2] = exp_norm.max()
        exp_norm = exp_norm/exp_norm.max()
        exp_norm[exp_norm>1] = 1.0
        exp_means += exp_norm
    rgba_colors = np.zeros((data.shape[1], 4))
    rgba_colors[:,0:3] = color_rgb[:3]
    rgba_colors[:,3] = exp_means/exp_means.max()
    print(rgba_colors.shape)
    plt.scatter(tsne[0,:], tsne[1,:], color=rgba_colors, **cluster_params)
    plt.savefig(filename)



# support for uncurl relabeling

# merge, split, delete clusters

# relabel cell types

import numpy as np

import uncurl
from uncurl import state_estimation

from . import entropy

def relabel(data, m_old, w_old, cell_ids, cell_labels, **uncurl_params):
    """
    Re-runs UNCURL on the dataset, after re-initializing W

    Args:
        data (array): genes x cells
        m_old (array): genes x k
        w_old (array): k x cells
        cell_ids (array): 1d array of cell ids
        cell_labels (array): 1d array of new cell labels
        **uncurl_params: optional kwargs to pass to uncurl

    Returns: M_new, W_new
    """
    k = m_old.shape[1]
    init_weights = w_old.argmax(0)
    for c in cell_ids:
        true_label = cell_labels[c]
        init_weights[c] = true_label
    m_new, w_new, ll_new = uncurl.run_state_estimation(data,
            clusters=k,
            #init_means=m_old,
            init_weights=init_weights,
            **uncurl_params)
    return m_new, w_new

def split_cluster(data, m_old, w_old, cluster_to_split, **uncurl_params):
    """
    Splits a given cluster, returning the results of an uncurl re-initialization.

    Args:
        data (array): genes x cells
        m_old (array): genes x k
        w_old (array): k x cells
        cluster_to_split (int): cluster id to split on
        **uncurl_params: optional kwargs to pass to uncurl

    Returns: M_new, W_new
    """
    # TODO
    k = m_old.shape[1]
    k += 1
    labels = w_old.argmax(0)
    cell_subset = (labels==cluster_to_split)
    # or (sorted_labels[1,:]==cluster_to_split)
    # TODO: initialization??? run km++ or poisson cluster on the cell subset?
    data_subset = data[:,cell_subset]
    new_m, new_w = state_estimation.initialize_means_weights(data_subset, 2,
            max_assign_weight=0.75)
    m_init = np.hstack([m_old[:, :cluster_to_split],
                       new_m,
                       m_old[:, cluster_to_split+1:]])
    # extend new_w to encompass all cells
    new_w_2 = np.zeros((2, w_old.shape[1]))
    new_w_2[0,:] = w_old[cluster_to_split]/2
    new_w_2[1,:] = w_old[cluster_to_split]/2
    new_w_2[:,cell_subset] = new_w
    w_init = np.vstack([w_old[:cluster_to_split, :],
                       new_w_2,
                       w_old[cluster_to_split+1:, :]])
    w_init = w_init/w_init.sum(0)
    m_new, w_new, ll_new = uncurl.run_state_estimation(data,
            clusters=k,
            init_means=m_init,
            init_weights=w_init,
            **uncurl_params)
    return m_new, w_new

def merge_clusters(data, m_old, w_old, clusters_to_merge,
        rerun_uncurl=True, **uncurl_params):
    """
    Merges a given list of clusters, returning the results of an uncurl re-initialization.

    Merging is done by averaging m_old over the clusters to be merged,
    and summing over w_old (and re-normalizing).

    Args:
        data (array): genes x cells
        m_old (array): genes x k
        w_old (array): k x cells
        clusters_to_merge (list): list of cluster ids to merge
        rerun_uncurl (boolean): if True, re-runs uncurl with a new initialization.
            If false, this just returns the merged cluster matrix.
        **uncurl_params: optional kwargs to pass to uncurl

    Returns: M_new, W_new
    """
    k = m_old.shape[1] - 1
    m_init_new_col = np.zeros(m_old.shape[0])
    w_init_new_row = np.zeros(w_old.shape[1])
    clusters_to_remove = np.array([True for i in range(k+1)])
    clusters_to_remove[list(clusters_to_merge)] = False
    m_init = m_old[:,clusters_to_remove]
    w_init = w_old[clusters_to_remove,:]
    for c in clusters_to_merge:
        m_init_new_col += m_old[:,c]
        w_init_new_row += w_old[c,:]
    m_init_new_col = m_init_new_col/len(clusters_to_merge)
    m_init_new_col = m_init_new_col.reshape((m_old.shape[0],1))
    w_init_new_row = w_init_new_row.reshape((1, w_old.shape[1]))
    m_init = np.hstack([m_init[:, 0:clusters_to_merge[0]],
                        m_init_new_col,
                        m_init[:, clusters_to_merge[0]:]])
    w_init = np.vstack([w_init[0:clusters_to_merge[0], :],
                        w_init_new_row,
                        w_init[clusters_to_merge[0]:, :]])
    w_init = w_init/w_init.sum(0)
    m_new, w_new, ll_new = uncurl.run_state_estimation(data,
            clusters=k,
            init_means=m_init,
            init_weights=w_init,
            **uncurl_params)
    return m_new, w_new


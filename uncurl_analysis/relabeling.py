# support for uncurl relabeling

# merge, split, delete clusters

# relabel cell types

import numpy as np

import uncurl

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
    k = m_old.shape[1]
    k += 1
    new_m_col = m_old[:,cluster_to_split]
    new_m_col += new_m_col*np.random.random(m_old.shape[1])
    new_m_col = new_m_col.reshape((m_old.shape[0],1))
    m_old = np.hstack([m_old[:,:cluster_to_split],
                       new_m_col,
                       m_old[:,cluster_to_split:]])
    new_w_row = w_old[cluster_to_split,:]
    w_old = np.vstack([w_old[:cluster_to_split,:],
                       new_w_row,
                       w_old[cluster_to_split:,:]])
    m_new, w_new, ll_new = uncurl.run_state_estimation(data,
            clusters=k,
            init_means=m_old,
            init_weights=w_old,
            **uncurl_params)
    return m_new, w_new

def split_cluster_cells(data, m_old, w_old, cluster_to_split, **uncurl_params):
    """
    Splits a given cluster, returning the results of an uncurl re-initialization.

    Unlike split_cluster, this only runs uncurl on the subset of cells that are assigned to the given cluster.

    Args:
        data (array): genes x cells
        m_old (array): genes x k
        w_old (array): k x cells
        cluster_to_split (int): cluster id to split on
        **uncurl_params: optional kwargs to pass to uncurl

    Returns: M_new, W_new
    """
    k = m_old.shape[1]
    k += 1
    labels = w_old.argmax(0)
    sorted_labels = w_old.argsort(0)
    cell_subset = (sorted_labels[0,:]==cluster_to_split) or (sorted_labels[1,:]==cluster_to_split)
    #cell_subset = (labels==cluster_to_split)
    cell_others = ~cell_subset
    data_subset = data[cell_subset, :]
    m_sub, w_sub, ll_new = uncurl.run_state_estimation(data_subset,
            clusters=2,
            **uncurl_params)
    m_new = np.zeros((m_old.shape[0], m_old.shape[1]+1))
    w_new = np.zeros((w_old.shape[0]+1, w_old.shape[1]))
    m_new[:,:cluster_to_split] = m_old[:,:cluster_to_split]
    m_new[:,cluster_to_split:cluster_to_split+2] = m_sub
    m_new[:,cluster_to_split+1:] = m_old[:,cluster_to_split+1:]
    w_new[:cluster_to_split,:] = w_old[:cluster_to_split,:]
    w_new[cluster_to_split:cluster_to_split+2, cell_subset] = w_sub/w_old[:, cell_subset]
    w_new[cluster_to_split+1:,:] = w_old[cluster_to_split+1:,:]
    return m_new, w_new

def merge_clusters(data, m_old, w_old, clusters_to_merge, **uncurl_params):
    """
    Merges a given list of clusters, returning the results of an uncurl re-initialization.

    Args:
        data (array): genes x cells
        m_old (array): genes x k
        w_old (array): k x cells
        clusters_to_merge (list): list of cluster ids to merge
        **uncurl_params: optional kwargs to pass to uncurl

    Returns: M_new, W_new
    """
    k = m_old.shape[1]
    m_new, w_new, ll_new = uncurl.run_state_estimation(data,
            clusters=k,
            init_means=m_old,
            init_weights=w_old,
            **uncurl_params)
    return m_new, w_new

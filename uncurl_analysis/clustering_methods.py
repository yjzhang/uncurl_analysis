# TODO: interfaces to different clustering methods - ex. Louvain, Leiden,...

import igraph as ig
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import umap

def create_graph(data, n_neighbors=20, metric='euclidean', **params):
    """
    Data is of shape cells x d, where d is dimensionality-reduced.

    Returns an igraph graph
    """
    knn_indices, knn_dists, tree = umap.umap_.nearest_neighbors(data, n_neighbors=n_neighbors, metric=metric,
            metric_kwds={}, random_state=np.random.RandomState(), angular=False)
    # build a graph...
    g = ig.Graph()
    g.add_vertices(data.shape[0])
    # preprocess edges
    edges = []
    edge_weights = []
    for i in range(data.shape[0]):
        nns = knn_indices[i, 1:]
        weight = 1.0/knn_dists[i, 1:]
        edges += [(i, n) for n in nns]
        edge_weights += list(weight)
    # add edges
    g.add_edges(edges)
    g.es['weight'] = edge_weights
    return g

def run_leiden(g, **params):
    """
    runs the leiden partitioning algorithm on a given graph.
    """
    import leidenalg
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    print(part)
    return part.membership

def run_louvain(g, **params):
    """
    runs the leiden partitioning algorithm on a given graph.
    """
    part = g.community_multilevel(weights='weight')
    return part.membership

def baseline_cluster(data, **params):
    """
    Args:
    """
    # TODO: combine dim-red + clustering

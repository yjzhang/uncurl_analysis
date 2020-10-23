# TODO: interfaces to different clustering methods - ex. Louvain, Leiden,...

import igraph as ig
import numpy as np
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
        weight = 1.0/(knn_dists[i, 1:] + 1e-10)
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
    return part.membership

def run_louvain(g, **params):
    """
    runs the leiden partitioning algorithm on a given graph.
    """
    g = g.simplify()
    part = g.community_multilevel()
    return part.membership

def baseline_cluster(data, **params):
    """
    Runs clustering on raw single-cell data.
    Args:
        data (array of shape (genes, cells))
        n_neighbors (int)
        metric (str): 'euclidean', 'cosine', etc.
    """
    # combine dim-red + clustering
    from uncurl.preprocessing import log1p
    import leidenalg
    tsvd = TruncatedSVD(20)
    data_transformed = tsvd.fit_transform(log1p(data).T)
    graph = create_graph(data_transformed, **params)
    part = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
    return part.membership

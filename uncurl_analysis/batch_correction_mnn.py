# TODO: try to do batch effect correction with mutual nearest neighbors
# ref: https://github.com/MarioniLab/FurtherMNN2018
# We also want it to work on sparse matrices.

import nmslib
from sklearn.utils.extmath import randomized_svd
from uncurl.preprocessing import log1p, cell_normalize


def identify_mnns(data1, data2, k=20, n_components=20):
    """
    Identify mutual nearest neighbors in the two datasets.

    Args:
        data1, data2: sparse (csc) matrices of shape genes x cells
        k (int): number of nearest neighbors
        n_components (int): number of components for tsvd

    Returns:
        a list of lists, indicating MNNs every cell in data1
    """
    # 1. compute tsvd on sparse matrices
    U, Sigma, VT = randomized_svd(log1p(cell_normalize(data1)).T,
                      n_components)
    data1_reduced = U*Sigma
    U, Sigma, VT = randomized_svd(log1p(cell_normalize(data2)).T,
                      n_components)
    data2_reduced = U*Sigma
    # 2. build index for euclidean nearest neighbors using nmslib
    index1 = nmslib.init(method='hnsw', space='l2')
    index1.addDataPointBatch(data1_reduced)
    index2 = nmslib.init(method='hnsw', space='l2')
    index2.addDataPointBatch(data2_reduced)
    neighbors1 = []
    neighbors2 = []
    for i in range(data1.shape[1]):
        points = index2.knnQueryBatch(data1_reduced[i,:], k=k)
        points = set([n[0][0] for n in points])
        neighbors1.append(points)
    for i in range(data2.shape[1]):
        points = index1.knnQueryBatch(data2_reduced[i,:], k=k)
        points = set([n[0][0] for n in points])
        neighbors2.append(points)
    mnns = [[] for i in range(data1.shape[1])]
    for i in range(data1.shape[1]):
        for p in neighbors1[i]:
            if i in neighbors2[p]:
                mnns[i].append(p)
    return mnns


def compute_correction_vectors(data1, data2, mnns):
    """
    """

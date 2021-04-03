# TODO: try to do batch effect correction with mutual nearest neighbors
# ref: https://github.com/MarioniLab/FurtherMNN2018
# We also want it to work on sparse matrices.

import nmslib
import scipy.sparse
from sklearn.utils.extmath import randomized_svd
from sklearn.neighbors import KDTree
from uncurl.preprocessing import log1p, cell_normalize


def identify_mnns(data1, data2, k=20, n_components=20, metric='cosine'):
    """
    Identify mutual nearest neighbors in the two datasets.

    Args:
        data1, data2: sparse (csc) matrices of shape genes x cells
        k (int): number of nearest neighbors
        n_components (int): number of components for tsvd
        metric (str): cosine or euclidean

    Returns:
        a list of lists, indicating MNNs every cell in data1
    """
    # 1. compute tsvd on sparse matrices
    data_combined = scipy.sparse.hstack([data1, data2])
    cell_indices = [1]*data1.shape[1] + [2]*data2.shape[1]
    U, Sigma, VT = randomized_svd(log1p(cell_normalize(data_combined)).T,
                      n_components)
    data_reduced = U*Sigma
    # 2. build index for euclidean nearest neighbors using nmslib
    # space can be cosinesiml or l2
    metric_to_space = {'cosine': 'cosinesimil', 'euclidean': 'l2'}
    index1 = nmslib.init(method='hnsw', space=metric_to_space[metric])
    index1.addDataPointBatch(data_reduced)
    neighbors1 = []
    neighbors2 = []
    for i in range(data1.shape[1]):
        points = index2.knnQueryBatch(data_reduced[i,:], k=k)
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
    Given two datasets (as sparse matrices), this projects data2 onto data1 using the provided list of mutual nearest neighbors.
    """
    # TODO


def batch_correct_n_datasets(datasets):
    """
    Given a list of datasets (as sparse matrices of shape genes x cells), this runs batch effect correction to create a unified dataset.
    """
    # TODO

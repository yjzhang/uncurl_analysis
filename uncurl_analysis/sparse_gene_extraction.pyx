cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log2

from scipy import sparse
from scipy.special import xlogy

ctypedef fused int2:
    short
    int
    long
    long long

ctypedef fused DTYPE_t:
    float
    double

ctypedef fused numeric:
    short
    unsigned short
    int
    unsigned int
    long
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def csc_overexpressed_genes(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        np.ndarray[long, ndim=1] labels,
        Py_ssize_t cells,
        Py_ssize_t genes,
        double eps=0):
    """
    Find overexpressed genes given a csc matrix...
    """
    cdef numeric[:] data_ = data
    cdef int2[:] indices_ = indices
    cdef int2[:] indptr_ = indptr
    if eps==0:
        eps = 10.0/cells
    cdef int2 g, c, start_ind, end_ind, i2
    cdef int2 k
    scores = {}
    labels_set = set(labels)
    cdef int2 K = len(labels_set)
    cdef double[:,:] cluster_means = np.zeros((genes, K))
    for k in labels_set:
        scores[k] = []
    cdef double[:] cluster_cell_counts = np.zeros(K)
    for c in range(cells):
        k = labels[c]
        cluster_cell_counts[k] += 1
        start_ind = indptr_[c]
        end_ind = indptr_[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices_[i2]
            cluster_means[g, k] += data_[i2]
    for g in range(genes):
        for k in range(K):
            cluster_means[g, k] = cluster_means[g, k]/cluster_cell_counts[k]
    cdef double max_k, max_k2
    for g in range(genes):
        for k in range(K):
            max_k = cluster_means[g, k] + eps
            max_k2 = 0
            for k2 in range(K):
                if k2 != k:
                    if cluster_means[g,k2] > max_k2:
                        max_k2 = cluster_means[g,k2]
            scores[k].append((g, max_k/(max_k2 + eps)))
    return scores

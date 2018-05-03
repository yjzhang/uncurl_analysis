cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log

from scipy import sparse
from scipy.special import xlogy

ctypedef fused int2:
    int
    long
    long long

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
def csc_log_prob_poisson_no_norm(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        # bulk data
        np.ndarray[double, ndim=1] bulk_data,
        double eps=1e-10):
    """
    Finds unnormalized Poisson log-likelihood, where data is a 1d CSC matrix
    and bulk_data is a 1d numpy array.
    """
    cdef int2 i, gene_id;
    cdef Py_ssize_t data_length = data.shape[0]
    cdef double ll = 0;
    cdef numeric[:] data_ = data
    cdef double[:] bulk = bulk_data
    for i in range(data_length):
        gene_id = indices[i]
        ll += data_[i]*log(bulk[gene_id] + eps) - bulk[gene_id]
    return ll

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def log_prob_poisson_no_norm(np.ndarray[numeric, ndim=1] data,
        # bulk data
        np.ndarray[double, ndim=1] bulk_data,
        double eps=1e-10):
    """
    Finds unnormalized Poisson log-likelihood, where data is a 1d numpy array
    and bulk_data is a 1d numpy array. Assumes gene indices are the same in
    both datasets.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t data_length = data.shape[0]
    cdef double ll = 0;
    cdef numeric[:] data_ = data
    cdef double[:] bulk = bulk_data
    for i in range(data_length):
        if data_[i] == 0:
            ll -= bulk[i]
        else:
            ll += data_[i]*log(bulk[i] + eps) - bulk[i]
    return ll

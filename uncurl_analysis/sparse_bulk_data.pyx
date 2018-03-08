cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log

from scipy import sparse
from scipy.special import xlogy

ctypedef fused int2:
    short
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
        np.ndarray[numeric, ndim=1] bulk_data,
        double eps=1e-10):
    """
    Finds unnormalized Poisson log-likelihood, where data is a 1d CSC matrix
    and bulk_data is a 1d numpy array.
    """
    cdef int2 i, gene_id;
    cdef Py_ssize_t data_length = data.shape[0]
    cdef double ll = 0;
    cdef numeric[:] bulk = bulk_data
    for i in range(data_length):
        gene_id = indices[i]
        ll += data[i]*log(bulk[gene_id] + eps) - bulk[gene_id]
    return ll

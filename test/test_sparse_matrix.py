import os
import unittest
from unittest import TestCase

from uncurl_analysis import sparse_matrix_h5

from scipy import sparse
import scipy.io
import numpy as np

class SparseMatrixStorageTest(TestCase):

    def setUp(self):
        pass

    def testStoreRetrieve(self):
        data_mat = scipy.io.loadmat('data/10x_pooled_400.mat')
        data = sparse.csc_matrix(data_mat['data'])
        sparse_matrix_h5.store_matrix(data, '/tmp/test.h5')
        row = sparse_matrix_h5.load_row('/tmp/test.h5', 100)
        self.assertTrue((data[100,:].toarray() == row).all())
        row = sparse_matrix_h5.load_row('/tmp/test.h5', 0)
        self.assertTrue((data[0,:].toarray() == row).all())
        row = sparse_matrix_h5.load_row('/tmp/test.h5', 15000)
        self.assertTrue((data[15000,:].toarray() == row).all())
        retrieved_matrix = sparse_matrix_h5.load_matrix('/tmp/test.h5')
        retrieved_matrix = sparse.csc_matrix(retrieved_matrix)
        self.assertTrue((retrieved_matrix != data).nnz == 0)

if __name__ == '__main__':
    unittest.main()


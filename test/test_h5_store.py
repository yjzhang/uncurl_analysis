import os
import unittest
from unittest import TestCase

from uncurl_analysis import dense_matrix_h5

import numpy as np

class H5StorageTest(TestCase):

    def setUp(self):
        pass

    def test_array_create(self):
        data = np.random.random((20, 100))
        h5_file = 'data.h5'
        try:
            os.remove(h5_file)
        except:
            pass
        dense_matrix_h5.store_array(h5_file, data)
        array = dense_matrix_h5.H5Array(h5_file)
        self.assertTrue((array[0,:] == data[0,:]).all())
        self.assertTrue((array.toarray() == data).all())

    def test_dict_create(self):
        data = {}
        for i in range(10):
            data[i] = np.random.random(20)
        h5_file = 'data_array.h5'
        try:
            os.remove(h5_file)
        except:
            pass
        dense_matrix_h5.store_dict(h5_file, data)
        d = dense_matrix_h5.H5Dict(h5_file)
        for i in range(10):
            self.assertTrue((d[i] == data[i]).all())
        self.assertEqual(len(d), 10)
        for k, v in d.items():
            self.assertTrue((data[int(k)] == v).all())
        d['10'] = data[0]
        self.assertTrue((data[0] == d[10]).all())



if __name__ == '__main__':
    unittest.main()


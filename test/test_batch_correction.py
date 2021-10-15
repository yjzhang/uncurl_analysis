import unittest

import numpy as np
from scipy.io import loadmat
from scipy import sparse

import random

from uncurl_analysis.batch_correction import batch_correct_mnn

class BatchCorrectionTest(unittest.TestCase):

    def setUp(self):
        # load data, which are separated into multiple batches.
        dat = loadmat('data/10x_pooled_400.mat')
        self.data = sparse.csc_matrix(dat['data'])
        self.data_dense = self.data.toarray()
        self.labs = dat['labels'].flatten()

    def testBatchCorrect1(self):
        indices = list(range(self.data.shape[1]))
        # 1. separate data into batches, randomly
        random.shuffle(indices)
        batch1 = indices[:200]
        batch2 = indices[200:]
        data1 = self.data[:, batch1]
        data2 = self.data[:, batch2]
        # 2. test batch correction
        print(data1.shape)
        print(data2.shape)
        data_corrected = batch_correct_mnn([data1, data2])
        self.assertTrue(data_corrected.shape[1] == self.data.shape[1])
        self.assertTrue(data_corrected.shape[0] == self.data.shape[0])


if __name__ == '__main__':
    unittest.main()

import os
from unittest import TestCase

from uncurl_analysis import sc_analysis

from scipy import sparse
import scipy.io
from scipy.io import loadmat
import numpy as np

class SCAnalysisTest(TestCase):

    def setUp(self):
        dat = loadmat('data/10x_pooled_400.mat')
        self.data_dir = '/tmp/uncurl_analysis/test'
        try:
            os.makedirs(self.data_dir)
        except:
            pass
        self.data = sparse.csc_matrix(dat['data'])
        scipy.io.mmwrite(os.path.join(self.data_dir, 'data.mtx'), self.data)

    def testLoadFromFolder(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx')
        self.assertEqual(sca.clusters, 8)
        self.assertEqual(sca.data_dir, self.data_dir)
        self.assertEqual(sca.data_f, os.path.join(self.data_dir, 'data.mtx'))

    def testRunUncurl(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                max_iters=20,
                inner_max_iters=50)
        sca.run_uncurl()
        self.assertTrue(sca.has_w)
        self.assertTrue(sca.has_m)
        self.assertTrue(sca.w.shape[0] == 8)
        self.assertTrue(sca.w.shape[1] == self.data.shape[1])


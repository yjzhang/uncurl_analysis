from unittest import TestCase

from uncurl_analysis.bulk_data import bulk_lookup

from scipy import sparse
from scipy.io import loadmat
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

class BulkLookupTest(TestCase):

    def setUp(self):
        dat = loadmat('data/10x_pooled_400.mat')
        self.data = sparse.csc_matrix(dat['data'])
        self.data_dense = self.data.toarray()
        self.labs = dat['labels'].flatten()
        self.bulk_means = {}
        for x in set(self.labs):
            means = np.array(self.data[:,self.labs==x].mean(1)).flatten()
            self.bulk_means[x] = means/means.sum()

    def testPoissonLookup(self):
        data_dense = self.data_dense
        labels = []
        for i in range(data_dense.shape[1]):
            cell = data_dense[:,i]
            scores = bulk_lookup(self.bulk_means, cell)
            labels.append(scores[0][0])
        nmi_val = nmi(self.labs, labels)
        self.assertTrue(nmi_val > 0.99)

    def testCosineLookup(self):
        data_dense = self.data_dense
        labels = []
        for i in range(data_dense.shape[1]):
            cell = data_dense[:,i]
            scores = bulk_lookup(self.bulk_means, cell, method='cosine')
            labels.append(scores[0][0])
        nmi_val = nmi(self.labs, labels)
        self.assertTrue(nmi_val > 0.85)

    def testRankCorrLookup(self):
        data_dense = self.data_dense
        labels = []
        for i in range(data_dense.shape[1]):
            cell = data_dense[:,i]
            scores = bulk_lookup(self.bulk_means, cell, method='rank_corr')
            labels.append(scores[0][0])
        nmi_val = nmi(self.labs, labels)
        self.assertTrue(nmi_val > 0.95)

    def testCorrLookup(self):
        data_dense = self.data_dense
        labels = []
        for i in range(data_dense.shape[1]):
            cell = data_dense[:,i]
            scores = bulk_lookup(self.bulk_means, cell, method='corr')
            labels.append(scores[0][0])
        nmi_val = nmi(self.labs, labels)
        self.assertTrue(nmi_val > 0.85)

import numpy as np
import scipy.io
import scipy.sparse
from  sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import uncurl

from uncurl_analysis import clustering_methods

import unittest

class ClusteringMethodsTest(unittest.TestCase):

    def setUp(self):
        dat = scipy.io.loadmat('data/10x_pooled_400.mat')
        self.data = scipy.sparse.csc_matrix(dat['data'])
        self.labels = dat['labels'].flatten()
        # 2. gene selection
        genes = uncurl.max_variance_genes(self.data)
        self.data_subset = self.data[genes,:]


    def test_leiden(self):
        m, w, ll = uncurl.run_state_estimation(self.data_subset, 8, max_iters=20, inner_max_iters=50)
        print('nmi basic: ' + str(nmi(self.labels, w.argmax(0))))
        g = clustering_methods.create_graph(w.T, metric='cosine')
        leiden_clustering = clustering_methods.run_leiden(g)
        self.assertTrue(nmi(self.labels, leiden_clustering) >= 0.7)
        louvain_clustering = clustering_methods.run_louvain(g)
        self.assertTrue(nmi(self.labels, louvain_clustering) >= 0.7)


if __name__ == '__main__':
    unittest.main()

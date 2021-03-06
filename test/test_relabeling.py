# re-initializing uncurl 

# splitting/merging clusters

from unittest import TestCase

import numpy as np
import scipy.io
from  sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import uncurl

from collections import Counter
from uncurl_analysis import relabeling


class RelabelingTest(TestCase):

    def setUp(self):
        data = scipy.io.loadmat('data/10x_pooled_400.mat')

        data_csc = data['data']
        self.labels = data['labels'].flatten()
        #gene_names = data['gene_names']

        # 2. gene selection
        genes = uncurl.max_variance_genes(data_csc)
        self.data_subset = data_csc[genes,:]
        #gene_names_subset = gene_names[genes]

        # 3. run uncurl
        m, w, ll = uncurl.run_state_estimation(self.data_subset, 8, max_iters=20, inner_max_iters=50)
        print('nmi basic: ' + str(nmi(self.labels, w.argmax(0))))
        self.m = m
        self.w = w

    def test_split(self):
        # 5. building a distance matrix between clusters, find closest pair

        # 6. run split_cluster - split the largest cluster
        m = self.m
        w = self.w
        data_subset = self.data_subset
        labels = self.labels
        clusters = w.argmax(0)
        cluster_counts = Counter(clusters)
        top_cluster, top_count = cluster_counts.most_common()[0]

        m_split, w_split = relabeling.split_cluster(data_subset, m, w, top_cluster, max_iters=20, inner_max_iters=50)
        nmi_base = nmi(labels, w.argmax(0))
        nmi_split = nmi(labels, w_split.argmax(0))
        print('nmi after splitting the largest cluster: ' + str(nmi_split))
        self.assertTrue(nmi_split >= nmi_base - 0.02)
        self.assertEqual(w_split.shape[0], w.shape[0] + 1)

    def test_merge(self):
        # create distance matrix
        # find the min distance between two cluster pairs
        m = self.m
        w = self.w
        data_subset = self.data_subset
        clusters_to_merge = [0,1,2]
        # merge the min distance pair
        m_merge, w_merge = relabeling.merge_clusters(data_subset, m, w,
                clusters_to_merge, max_iters=20, inner_max_iters=50)
        nmi_base = nmi(self.labels, w.argmax(0))
        nmi_merge = nmi(self.labels, w_merge.argmax(0))
        print('nmi after merging the closest pairs: ' + str(nmi_merge))
        self.assertTrue(nmi_merge >= nmi_base - 0.3)
        self.assertEqual(w_merge.shape[0], w.shape[0] -2)

    def test_new(self):
        """
        Tests creating a new cluster from a selection of cells
        """
        data_subset = self.data_subset
        m = self.m
        w = self.w
        selected_cells = list(range(375, w.shape[1]))
        m_new, w_new = relabeling.new_cluster(data_subset, m, w,
                selected_cells, max_iters=20, inner_max_iters=50)
        nmi_base = nmi(self.labels, w.argmax(0))
        nmi_new = nmi(self.labels, w_new.argmax(0))
        self.assertTrue(w_new.shape[0] == 9)
        print('nmi after creating a new cluster: ' + str(nmi_new))
        self.assertTrue(nmi_new >= nmi_base - 0.1)
        self.assertEqual(w_new.shape[0], w.shape[0] + 1)
        self.assertTrue(sum((w_new.argmax(0)[selected_cells] == 8)) >= len(selected_cells)/2)

from collections import Counter
import os
from unittest import TestCase
import shutil


from uncurl_analysis import sc_analysis

from scipy import sparse
import scipy.io
from scipy.io import loadmat
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import numpy as np

class SCAnalysisTest(TestCase):

    def setUp(self):
        dat = loadmat('data/10x_pooled_400.mat')
        self.labs = dat['labels'].flatten()
        self.data_dir = '/tmp/uncurl_analysis/test'
        try:
            os.makedirs(self.data_dir)
        except:
            shutil.rmtree(self.data_dir)
            os.makedirs(self.data_dir)
        self.data = sparse.csc_matrix(dat['data'])
        # take subset of 5000 genes
        self.data = self.data[5000:10000,:]
        scipy.io.mmwrite(os.path.join(self.data_dir, 'data.mtx'), self.data)
        shutil.copy('data/10x_pooled_400_gene_names.tsv', os.path.join(self.data_dir, 'gene_names.txt'))

    def test_load_from_folder(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx')
        self.assertEqual(sca.clusters, 8)
        self.assertEqual(sca.data_dir, self.data_dir)
        self.assertEqual(sca.data_f, os.path.join(self.data_dir, 'data.mtx'))

    def test_run_uncurl(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                frac=0.5,
                data_filename='data.mtx',
                max_iters=20,
                inner_max_iters=50)
        sca.run_uncurl()
        self.assertTrue(sca.has_w)
        self.assertTrue(sca.has_m)
        self.assertTrue(sca.w.shape[0] == 8)
        self.assertTrue(sca.w.shape[1] == self.data.shape[1])
        self.assertTrue(os.path.exists(sca.w_f))
        self.assertTrue(os.path.exists(sca.m_f))
        print(nmi(sca.labels, self.labs))
        self.assertTrue(nmi(sca.labels, self.labs) > 0.65)

    def test_dim_red_sample(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                cell_frac=0.2,
                max_iters=20,
                inner_max_iters=20)
        mds_means = sca.mds_means
        self.assertEqual(mds_means.shape[0], 2)
        self.assertEqual(mds_means.shape[1], 8)
        dr = sca.dim_red
        self.assertEqual(dr.shape[0], 2)
        self.assertEqual(dr.shape[1], int(0.2*sca.data.shape[1]))

    def test_dim_red_2(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                dim_red_option='UMAP',
                baseline_dim_red='UMAP',
                cell_frac=0.2,
                max_iters=20,
                inner_max_iters=20)
        mds_means = sca.mds_means
        self.assertEqual(mds_means.shape[0], 2)
        self.assertEqual(mds_means.shape[1], 8)
        dr = sca.dim_red
        self.assertEqual(dr.shape[0], 2)
        self.assertEqual(dr.shape[1], int(0.2*sca.data.shape[1]))
        dr_baseline = sca.baseline_vis
        self.assertEqual(dr_baseline.shape[0], 2)
        self.assertEqual(dr_baseline.shape[1], int(0.2*sca.data.shape[1]))


    def test_run_full_analysis(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                baseline_dim_red='tsvd',
                dim_red_option='MDS',
                normalize=True,
                min_reads=10,
                max_reads=4000,
                cell_frac=0.5,
                max_iters=20,
                inner_max_iters=20)
        print(sca.data.shape)
        print(sca.cell_subset.shape)
        print(sca.cell_subset)
        print(sca.data_subset.shape)
        self.assertEqual(sca.cell_subset.shape[0], 400)
        self.assertTrue(sca.data_subset.shape[1] > 200)
        sca.run_full_analysis()
        self.assertTrue(sca.has_dim_red)
        self.assertTrue(sca.has_pvals)
        self.assertTrue(sca.has_top_genes_1_vs_rest)
        self.assertTrue(sca.has_top_genes)
        self.assertTrue(sca.has_baseline_vis)
        top_genes = sca.top_genes
        self.assertEqual(len(top_genes), 8)
        self.assertEqual(len(top_genes[0]), sca.data.shape[0])
        top_genes_1_vs_rest = sca.top_genes_1_vs_rest
        self.assertEqual(len(top_genes_1_vs_rest), 8)
        self.assertEqual(len(top_genes_1_vs_rest[0]), sca.data.shape[0])
        self.assertEqual(sca.dim_red.shape[0], 2)

    def test_json(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                baseline_dim_red='tsvd',
                cell_frac=0.5,
                max_iters=20,
                inner_max_iters=20)
        sca.run_full_analysis()
        sca.save_json_reset()
        sca = sca.load_params_from_folder()
        self.assertEqual(sca.clusters, 8)
        self.assertEqual(sca.baseline_dim_red, 'tsvd')
        self.assertEqual(sca.uncurl_kwargs['max_iters'], 20)
        self.assertTrue(sca.has_dim_red)
        self.assertTrue(sca.has_w)
        self.assertTrue(sca.has_m)
        self.assertEqual(sca.cell_subset.shape[0], 400)

    def test_split_cluster(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                baseline_dim_red='tsvd',
                dim_red_option='MDS',
                cell_frac=1.0,
                max_iters=20,
                inner_max_iters=10)
        sca.run_full_analysis()
        # split two clusters....
        clusters = sca.labels
        cluster_counts = Counter(clusters)
        top_cluster, top_count = cluster_counts.most_common()[0]
        print(cluster_counts)
        print(top_cluster, top_count)
        sca.recluster('split', [top_cluster])
        sca.run_post_analysis()
        self.assertEqual(sca.clusters, 9)
        self.assertEqual(sca.w_sampled.shape[0], 9)

    def test_gene_names(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                baseline_dim_red='tsvd',
                dim_red_option='MDS',
                cell_frac=1.0,
                max_iters=20,
                inner_max_iters=10)
        gene_info = sca.data_sampled_gene('AGRN')
        self.assertEqual(gene_info.shape[0], sca.data_sampled_all_genes.shape[1])

    def test_add_color_track(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                baseline_dim_red='tsvd',
                dim_red_option='MDS',
                cell_frac=1.0,
                max_iters=20,
                inner_max_iters=10)
        sca.add_color_track('true_labels', self.labs, is_discrete=True)
        true_labels, _ = sca.get_color_track('true_labels')
        self.assertTrue(nmi(true_labels, self.labs) > 0.99)
        top_genes = sca.calculate_diffexp('true_labels')
        self.assertEqual(len(top_genes), 8)
        sca.add_color_track('true_labels_2', self.labs, is_discrete=False)
        true_labels_2, _ = sca.get_color_track('true_labels_2')
        self.assertTrue((true_labels_2.astype(int) == self.labs).all())



    """
    def test_merge_cluster(self):
        sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx',
                baseline_dim_red='tsvd',
                cell_frac=0.99,
                max_iters=20,
                inner_max_iters=20)
        sca.run_full_analysis()
        # merge two clusters....
        distance_matrix = np.zeros((8,8))
        min_distance_pair = (0,0)
        min_distance = 1e10
        for i in range(8):
            for j in range(8):
                distance_matrix[i,j] = uncurl.sparse_utils.poisson_dist(m[:,i], m[:,j])
                if i != j and distance_matrix[i,j] < min_distance:
                    min_distance = distance_matrix[i,j]
                    min_distance_pair = (i,j)
        m_merge, w_merge = relabeling.merge_clusters(data_subset, m, w,
                        min_distance_pair)
        clusters = sca.w.argmax(0)
        cluster_counts = Counter(clusters)
        top_cluster, top_count = cluster_counts.most_common()[0]
        sca.recluster('split', [top_cluster])
        sca.run_post_analysis()
        self.assertEqual(sca.clusters, 9)
        self.assertEqual(self.w_sampled.shape[0], 9)
    """

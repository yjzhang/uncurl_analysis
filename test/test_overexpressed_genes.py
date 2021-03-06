from unittest import TestCase

from uncurl_analysis.gene_extraction import find_overexpressed_genes, generate_permutations, calculate_permutation_pval, c_scores_to_pvals, pairwise_t, c_scores_from_t, separation_scores_from_t, one_vs_rest_t

from scipy import sparse
from scipy.io import loadmat
import numpy as np

class OverexpressedGenesTest(TestCase):

    def setUp(self):
        # TODO: generate synthetic datasets for testing,
        # where top cluster-specific genes are known
        dat = loadmat('data/10x_pooled_400.mat')
        # take 5000 genes arbitrarily?
        self.data = sparse.csc_matrix(dat['data'])[:5000,:]
        self.labs = dat['labels'].flatten()

    def testDense(self):
        data_dense = self.data.toarray()
        scores = find_overexpressed_genes(data_dense, self.labs)

    def testSparse(self):
        data_dense = self.data.toarray()
        scores = find_overexpressed_genes(data_dense, self.labs)
        scores_sparse = find_overexpressed_genes(self.data, self.labs)
        for k in set(self.labs):
            s1 = np.array([x[1] for x in scores[k]])
            s2 = np.array([x[1] for x in scores_sparse[k]])
            s1_genes = [x[0] for x in scores[k]]
            s2_genes = [x[0] for x in scores_sparse[k]]
            self.assertTrue(np.sqrt(((s1-s2)**2).sum()) < 1e-4)
            self.assertEqual(s1_genes, s2_genes)

    def testPermutationTest(self):
        scores_sparse = find_overexpressed_genes(self.data, self.labs)
        perms = generate_permutations(self.data, len(set(self.labs)),
                n_perms=20)
        pvals = c_scores_to_pvals(scores_sparse, perms)
        for k in set(self.labs):
            scores_k = scores_sparse[k]
            for gene_id, score in scores_k:
                if score <= 1.0:
                    break
                baseline = perms[gene_id]
                pval = calculate_permutation_pval(score, baseline)
                self.assertTrue(pval <= 1.0 and pval >= 0.0)

    def testTTest(self):
        # pairwise t-test, c-score
        t_test_scores, t_test_p_vals = pairwise_t(self.data, self.labs)
        self.assertEqual(t_test_scores.shape, t_test_p_vals.shape)
        K = len(set(self.labs))
        genes, cells = self.data.shape
        self.assertEqual(t_test_scores.shape, (K, K, genes))
        c_scores, c_pvals = c_scores_from_t(t_test_scores, t_test_p_vals)
        for k in range(K):
            print(k)
            print(c_scores[k][:10])
            print(c_pvals[k][:10])
            pvals = np.array([x[1] for x in c_pvals[k]])
            cscores = np.array([x[1] for x in c_scores[k]])
            self.assertTrue(c_scores[k][0][1] >= 1)
            #self.assertTrue(c_pvals[k][0][1] <= 0.05)
            self.assertTrue((pvals >= 0).all())
            self.assertTrue((pvals <= 1).all())
            self.assertTrue((cscores >= 0).all())
        # test separation score
        separation_scores, best_genes = separation_scores_from_t(t_test_scores, t_test_p_vals)
        self.assertEqual(separation_scores.shape, (K, K))
        self.assertTrue((separation_scores >= 0).all())

    def testTTest_normalize_fdr(self):
        # pairwise t-test, c-score
        t_test_scores, t_test_p_vals = pairwise_t(self.data, self.labs, normalize=True, use_fdr=True)
        self.assertEqual(t_test_scores.shape, t_test_p_vals.shape)
        K = len(set(self.labs))
        genes, cells = self.data.shape
        self.assertEqual(t_test_scores.shape, (K, K, genes))
        c_scores, c_pvals = c_scores_from_t(t_test_scores, t_test_p_vals)
        for k in range(K):
            print(k)
            print(c_scores[k][:10])
            print(c_pvals[k][:10])
            pvals = np.array([x[1] for x in c_pvals[k]])
            cscores = np.array([x[1] for x in c_scores[k]])
            self.assertTrue(c_scores[k][0][1] >= 1)
            #self.assertTrue(c_pvals[k][0][1] <= 0.05)
            self.assertTrue((pvals >= 0).all())
            self.assertTrue((pvals <= 1).all())
            self.assertTrue((cscores >= 0).all())

    def test_1_v_rest_TTest(self):
        t_test_scores, t_test_p_vals = one_vs_rest_t(self.data, self.labs)
        self.assertEqual(len(t_test_scores), len(t_test_p_vals))
        K = len(set(self.labs))
        genes, cells = self.data.shape
        self.assertEqual(len(t_test_scores), K)
        for k in set(self.labs):
            print(k)
            print(t_test_scores[k][:10])
            print(t_test_p_vals[k][:10])
            pvals = np.array([x[1] for x in t_test_p_vals[k]])
            cscores = np.array([x[1] for x in t_test_scores[k]])
            self.assertTrue(t_test_scores[k][0][1] >= 1)
            self.assertTrue(t_test_p_vals[k][0][1] <= 0.05)
            self.assertTrue((pvals >= 0).all())
            self.assertTrue((pvals <= 1).all())
            self.assertTrue((cscores >= 0).all())
        t_test_scores, _ = one_vs_rest_t(self.data, self.labs, calc_pvals=False)
        for k in set(self.labs):
            self.assertTrue(t_test_scores[k][0][1] >= 1)

    def test_1_v_rest_TTest_normalize_fdr(self):
        t_test_scores, t_test_p_vals = one_vs_rest_t(self.data, self.labs, normalize=True, use_fdr=True)
        self.assertEqual(len(t_test_scores), len(t_test_p_vals))
        K = len(set(self.labs))
        genes, cells = self.data.shape
        self.assertEqual(len(t_test_scores), K)
        for k in set(self.labs):
            print(k)
            print(t_test_scores[k][:10])
            print(t_test_p_vals[k][:10])
            pvals = np.array([x[1] for x in t_test_p_vals[k]])
            cscores = np.array([x[1] for x in t_test_scores[k]])
            self.assertTrue(t_test_scores[k][0][1] >= 1)
            self.assertTrue(t_test_p_vals[k][0][1] <= 0.05)
            self.assertTrue((pvals >= 0).all())
            self.assertTrue((pvals <= 1).all())
            self.assertTrue((cscores >= 0).all())
        for k in set(self.labs):
            self.assertTrue(t_test_scores[k][0][1] >= 1)

    def test_simulated_data(self):
        from uncurl import simulation
        data, clusters = simulation.generate_poisson_data(np.array([[1.0, 10.0], [10.0, 1.0], [0.5, 0.5]]), 100)
        data_csc = sparse.csc_matrix(data)
        ratios, pvals = pairwise_t(data_csc, clusters, eps=1e-8)
        print(ratios)
        print(data[1, clusters==0].mean()/data[1, clusters==1].mean())
        print(pvals)
        self.assertTrue(np.abs(ratios[0,1,1] - data[1, clusters==0].mean()/data[1, clusters==1].mean()) < 1.0)
        self.assertTrue(pvals[0,1,1] < 0.01)
        self.assertTrue(pvals[1,0,0] < 0.01)
        self.assertTrue(pvals[0,1,2] > 0.01)

    def test_simulated_data_normalize_fdr(self):
        from uncurl import simulation
        data, clusters = simulation.generate_poisson_data(np.array([[1.0, 10.0], [10.0, 1.0], [0.5, 0.5]]), 100)
        data_csc = sparse.csc_matrix(data)
        ratios, pvals = pairwise_t(data_csc, clusters, eps=1e-8, normalize=False, use_fdr=True)
        print(ratios)
        print(data[1, clusters==0].mean()/data[1, clusters==1].mean())
        print(pvals)
        self.assertTrue(np.abs(ratios[0,1,1] - data[1, clusters==0].mean()/data[1, clusters==1].mean()) < 1.0)
        self.assertTrue(pvals[0,1,1] < 0.01)
        self.assertTrue(pvals[1,0,0] < 0.01)
        self.assertTrue(pvals[0,1,2] > 0.01)

    def test_1_v_rest_simulated_data(self):
        from uncurl import simulation
        data, clusters = simulation.generate_poisson_data(np.array([[0.5, 4.0, 3.0], [10.0, 1.0, 1.0], [0.5, 0.5, 0.5]]), 100)
        data_csc = sparse.csc_matrix(data)
        ratios, pvals = one_vs_rest_t(data_csc, clusters, eps=1e-8, test='u')
        print(ratios)
        print(data[1, clusters==0].mean()/data[1, clusters==1].mean())
        print(pvals)
        self.assertTrue(ratios[0][0][0] == 1)
        self.assertTrue(ratios[1][0][0] == 0)
        self.assertTrue(pvals[0][0][0] == 1)
        self.assertTrue(pvals[0][0][1] < 0.05)
        self.assertTrue(pvals[0][2][1] > 0.01)
        self.assertTrue(pvals[1][0][0] == 0)
        self.assertTrue(pvals[1][0][1] < 0.05)

    def test_1_v_rest_simulated_data_normalize_fdr(self):
        from uncurl import simulation
        data, clusters = simulation.generate_poisson_data(np.array([[0.5, 4.0, 3.0], [10.0, 1.0, 1.0], [0.5, 0.5, 0.5]]), 100)
        data_csc = sparse.csc_matrix(data)
        ratios, pvals = one_vs_rest_t(data_csc, clusters, eps=1e-8, test='t', normalize=False, use_fdr=True)
        print(ratios)
        print(data[1, clusters==0].mean()/data[1, clusters==1].mean())
        print(pvals)
        self.assertTrue(ratios[0][0][0] == 1)
        self.assertTrue(ratios[1][0][0] == 0)
        self.assertTrue(pvals[0][0][0] == 1)
        self.assertTrue(pvals[0][0][1] < 0.05)
        self.assertTrue(pvals[0][2][1] > 0.01)
        self.assertTrue(pvals[1][0][0] < 0.001)
        self.assertTrue(pvals[1][0][1] < 0.05)

if __name__ == '__main__':
    import unittest
    unittest.main()

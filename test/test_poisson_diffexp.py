import unittest

import numpy as np
import scipy.io
import uncurl

from uncurl_analysis import poisson_diffexp

class DiffexpTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_log_wald(self):
        data1 = np.array([0.1,0.1,0.1,0.1,0.1])
        data2 = np.array([1,1.2,3,1.1])
        pv, ratio = poisson_diffexp.log_wald_poisson_test(data2, data1)
        print('pv: ', pv, ' ratio: ', ratio)
        self.assertTrue(pv < 0.05)

    def test_real_data_1_vs_rest(self):
        mat = scipy.io.loadmat('data/10x_pooled_400.mat')
        data = mat['data']
        # do uncurl, followed by update_m
        selected_genes = uncurl.max_variance_genes(data)
        data_subset = data[selected_genes, :]
        m, w, ll = uncurl.run_state_estimation(data_subset, 8, max_iters=20, inner_max_iters=50)
        m = uncurl.update_m(data, m, w, selected_genes)
        # TODO: how should the p-values be tested?
        all_pvs, all_ratios = poisson_diffexp.uncurl_poisson_test_1_vs_rest(m, w)
        all_pvs = np.array(all_pvs)
        all_ratios = np.array(all_ratios)
        self.assertTrue((all_pvs < 0.05).sum() > 100)
        self.assertTrue((all_ratios > 10).sum() > 100)
        self.assertEqual(all_pvs.shape, (data.shape[0], 8))
        all_pvs, all_ratios = poisson_diffexp.uncurl_poisson_test_1_vs_rest(m, w, mode='counts')
        all_pvs = np.array(all_pvs)
        all_ratios = np.array(all_ratios)
        self.assertEqual(all_pvs.shape, (data.shape[0], 8))
        self.assertTrue((all_pvs < 0.01).sum() > 100)
        self.assertTrue((all_pvs < 0.01).sum() < data.shape[0])
        self.assertTrue((all_ratios > 10).sum() > 100)

    def test_real_data_pairwise(self):
        mat = scipy.io.loadmat('data/10x_pooled_400.mat')
        data = mat['data']
        # do uncurl, followed by update_m
        selected_genes = uncurl.max_variance_genes(data)
        data_subset = data[selected_genes, :]
        m, w, ll = uncurl.run_state_estimation(data_subset, 8, max_iters=20, inner_max_iters=50)
        m = uncurl.update_m(data, m, w, selected_genes)
        # test pairwise
        all_pvs, all_ratios = poisson_diffexp.uncurl_poisson_test_pairwise(m, w, mode='counts')
        self.assertEqual(all_pvs.shape, (data.shape[0], 8, 8))
        self.assertEqual(all_ratios.shape, (data.shape[0], 8, 8))
        self.assertTrue((all_pvs < 0.001).sum() < data.shape[0])
        self.assertTrue((all_pvs < 0.01).sum() > 100)

    def test_simulated_data(self):
        from uncurl import simulation
        data, clusters = simulation.generate_poisson_data(np.array([[1.0, 100], [100, 1.0]]), 100)
        pvs, ratios, _ = poisson_diffexp.poisson_test_known_groups(data, clusters, test_mode='1_vs_rest')
        self.assertTrue(pvs[0, 1] < 0.001)
        self.assertFalse(pvs[0, 0] < 0.05)
        self.assertTrue(pvs[1, 0] < 0.001)
        self.assertFalse(pvs[1, 1] < 0.05)
        data, clusters = simulation.generate_poisson_data(np.array([[1.0, 5], [5, 1.0], [5, 5]]), 100)
        pvs, ratios, _ = poisson_diffexp.poisson_test_known_groups(data, clusters, test_mode='1_vs_rest')
        print(pvs)
        print(ratios)
        self.assertTrue(pvs[0, 1] < 0.05)
        self.assertTrue(pvs[1, 0] < 0.05)
        self.assertFalse(pvs[2, 1] < 0.05)
        self.assertFalse(pvs[2, 0] < 0.05)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from uncurl_analysis import poisson_diffexp

class DiffexpTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_log_wald(self):
        data1 = np.array([0.1,0.1,0.1,0.1,0.1])
        data2 = np.array([1,1.2,3,1.1])
        pv, ratio = poisson_diffexp.log_wald_poisson_test(data2, data1)
        print('pv: ', pv, ' ratio: ', ratio)
        self.assertTrue(pv < 0.1)

    def test_real_data(self):
        import scipy.io
        import uncurl
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
        self.assertEqual(all_pvs.shape, (data.shape[0], 8))
        self.assertTrue((all_pvs < 0.05).sum() > 100)
        self.assertTrue((all_pvs < 0.05).sum() < data.shape[1])
        self.assertTrue((all_ratios > 10).sum() > 100)
        all_pvs, all_ratios = poisson_diffexp.uncurl_poisson_test_1_vs_rest(m, w, mode='counts')
        all_pvs = np.array(all_pvs)
        all_ratios = np.array(all_ratios)

if __name__ == '__main__':
    unittest.main()

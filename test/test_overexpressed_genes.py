from unittest import TestCase

from uncurl_analysis.gene_extraction import find_overexpressed_genes, find_overexpressed_genes_m, generate_permutations, calculate_permutation_pval, c_scores_to_pvals

from scipy import sparse
from scipy.io import loadmat
import numpy as np

class OverexpressedGenesTest(TestCase):

    def setUp(self):
        # TODO: generate synthetic datasets for testing,
        # where top cluster-specific genes are known
        dat = loadmat('data/10x_pooled_400.mat')
        self.data = sparse.csc_matrix(dat['data'])
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
            print(k)
            for gene_id, score in scores_k:
                if score <= 1.0:
                    break
                baseline = perms[gene_id]
                pval = calculate_permutation_pval(score, baseline)
                print(gene_id, pval)
                self.assertTrue(pval <= 1.0 and pval >= 0.0)

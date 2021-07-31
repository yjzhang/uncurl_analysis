import os
import shutil
import unittest

import numpy as np
from scipy import sparse
import scipy.io

from uncurl_analysis import merge_datasets

class MergeTest(unittest.TestCase):

    def setUp(self):
        # TODO: copy/merge datasets
        dat = scipy.io.loadmat('data/10x_pooled_400.mat')
        self.data_dir = '/tmp/uncurl_analysis/test'
        try:
            shutil.rmtree(self.data_dir)
            os.makedirs(self.data_dir)
        except:
            os.makedirs(self.data_dir)
        self.data = sparse.csc_matrix(dat['data'])
        data_1 = self.data[:, :200]
        data_2 = self.data[:, 200:]
        scipy.io.mmwrite(os.path.join(self.data_dir, 'data_1.mtx'), data_1)
        scipy.io.mmwrite(os.path.join(self.data_dir, 'data_2.mtx'), data_2)
        data_2_trimmed = data_2[:-10, :]
        scipy.io.mmwrite(os.path.join(self.data_dir, 'data_3.mtx'), data_2_trimmed)
        genes = np.loadtxt('data/10x_pooled_400_gene_names.tsv', dtype=str)
        self.genes = genes
        genes_trimmed = genes[:-10]
        print(data_2_trimmed.shape, genes_trimmed.shape)
        np.savetxt(os.path.join(self.data_dir, 'gene_names_3.txt'), genes_trimmed, fmt='%s')
        shutil.copy('data/10x_pooled_400_gene_names.tsv', os.path.join(self.data_dir, 'gene_names_1.txt'))
        shutil.copy('data/10x_pooled_400_gene_names.tsv', os.path.join(self.data_dir, 'gene_names_2.txt'))
        self.data_paths_1 = [os.path.join(self.data_dir, 'data_1.mtx'), os.path.join(self.data_dir, 'data_2.mtx')]
        self.data_paths_2 = [os.path.join(self.data_dir, 'data_1.mtx'), os.path.join(self.data_dir, 'data_3.mtx')]
        self.gene_paths_1 = [os.path.join(self.data_dir, 'gene_names_1.txt'), os.path.join(self.data_dir, 'gene_names_2.txt')]
        self.gene_paths_2 = [os.path.join(self.data_dir, 'gene_names_1.txt'), os.path.join(self.data_dir, 'gene_names_3.txt')]

    def test_merge_data_1(self):
        # merge datasets with the same genes
        data_path, genes_path = merge_datasets.merge_files(self.data_paths_1, self.gene_paths_1, ['0', '1'], self.data_dir)
        self.assertTrue(data_path.endswith('.mtx.gz'))
        data = scipy.io.mmread(data_path)
        genes = np.loadtxt(genes_path, dtype=str)
        print(data.shape)
        self.assertEqual(data.shape[1], 400)
        self.assertEqual(data.shape[0], genes.shape[0])
        self.assertEqual(data.shape[0], 19848)
        self.assertTrue((self.data - data).__abs__().sum() < 0.00001)
        self.assertEqual(set(self.genes), set(genes))

    def test_merge_data_2(self):
        # merge datasets with the same genes
        data_path, genes_path = merge_datasets.merge_files(self.data_paths_2, self.gene_paths_2, ['0', '1'], self.data_dir)
        data = scipy.io.mmread(data_path)
        genes = np.loadtxt(genes_path, dtype=str)
        print(data.shape)
        self.assertEqual(data.shape[1], 400)
        self.assertEqual(data.shape[0], genes.shape[0])
        #data_test = []
        #gene_to_index = {gene : i for i, gene in enumerate(self.genes)}
        #gene_to_index = {}
        #for i, g in enumerate(self.genes):
        #    if g in gene_to_index:
        #        gene_to_index[g].append(i)
        #    else:
        #        gene_to_index[g] = [i]
        #for gene in genes:
        #    data_test.append(self.data[gene_to_index[gene], :])
        #self.data = sparse.vstack(data_test)
        self.assertEqual(set(self.genes), set(genes))
        #print((self.data - data).__abs__())
        #self.assertTrue((self.data - data).__abs__().sum() < 0.00001)

    def test_batch_correction(self):
        # 1. load all datasets
        pass

if __name__ == '__main__':
    unittest.main()

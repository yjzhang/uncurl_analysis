# TODO: run uncurl hierarchically

# since k=8 here, we can do a 2x2x2 split...

import heapq

import os
import sys
import time

import uncurl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy.io
from scipy import sparse


def m_ndcg(l1, l2, l3):
    """
    Calculates the discounted cumulative gain of list l1 that is shuffled
    to l2 and l3

    Based on Kuang and Park (2013).

    Args:
        l1, l2, l3 are np arrays of integers - sorted descending indices.
    """
    n_genes = l1.shape[0]
    genes = range(n_genes)
    # these arrays contain the ranking of each gene.
    # maps of gene_id : ranking
    l1_1 = np.zeros(n_genes)
    l2_1 = np.zeros(n_genes)
    l3_1 = np.zeros(n_genes)
    for i in genes:
        l1_1[l1[i]] = i
        l2_1[l2[i]] = i
        l3_1[l3[i]] = i
    # discount_factor and gain map genes to the respective factors.
    discount_factor = np.zeros(n_genes)
    gain = np.zeros(n_genes)
    for g in genes:
        discount_factor[g] = np.log(n_genes - max(l2_1[g], l3_1[g]) + 2)
        gain[g] = np.log(n_genes - l1_1[g] + 2)/discount_factor[g]
    print(discount_factor)
    print(gain)
    mDCG_2 = gain[l2[0]]
    mDCG_3 = gain[l3[0]]
    # g is gene,
    for i, g in enumerate(l2[1:]):
        mDCG_2 += gain[g]/np.log2(i+2)
    for i, g in enumerate(l3[1:]):
        mDCG_3 += gain[g]/np.log2(i+2)
    print('mDCG left: {0}, mDCG right: {1}'.format(mDCG_2, mDCG_3))
    gain_sorted_indices = np.argsort(gain)[::-1]
    gain_sorted = gain[gain_sorted_indices]
    mIDCG = gain_sorted[0]
    for i in range(1, n_genes):
        mIDCG += gain_sorted[i]/np.log2(i+1)
    print('mIDCG: {0}'.format(mIDCG))
    mNDCG_2 = mDCG_2/mIDCG
    mNDCG_3 = mDCG_3/mIDCG
    score = mNDCG_2*mNDCG_3
    return score

class UncurlNode(object):
    """
    Class representing a node used in hierarchical uncurl
    """

    def __init__(self, data, M, W, left=None, right=None):
        self.data = data
        self.M = M
        self.W = W
        self.left = left
        self.right = right

def hierarchical_uncurl(data, largek, max_depth, **uncurl_params):
    """
    Recursive matrix factorization using uncurl with k=2

    Based on Kuang and Park (2013)

    Args:
        data: CSC sparse matrix, shape=(genes, cells)
        largek: int, number of cell types
        max_depth: ???
        **uncurl_params: params for use with uncurl
    """
    # allocate M and W
    M = np.zeros((data.shape[0], largek))
    W = np.zeros((largek, data.shape[1]))
    # partition
    current_cell_partitions = []
    # list of current leaf nodes??? what should they contain?
    current_partitions = []
    data_subsets = [data]
    while len(current_partitions) < largek:
        data_ = data_subsets.pop()
        m, w = uncurl.poisson_estimate_state(data_, 2, **uncurl_params)
        labels = w.argmax(0)
        data_subset0 = data_[labels==0]
        data_subset1 = data_[labels==1]
    return M, W

def run_partition(data, smallk, largek, method, max_depth):
    """
    Very simple recursive partitioning-based state estimation system.

    Args:
        data
        smallk (int): k for each individual clustering
        largek (int): k for the whole global clustering
    """
    # what if some cell subsets have zero gene expression values?
    # we reduce the gene subset and then re-position m
    print('run partition: data shape={0}, smallk={1}, largek={2}'.format(data.shape, smallk, largek))
    genes = uncurl.max_variance_genes(data, nbins=1, frac=1.0)
    results, ll = method.run(data[genes,:])
    w = results[0]
    m_ = results[1]
    m = np.zeros((data.shape[0], smallk))
    m[genes,:] = m_
    clusters_0 = w.argmax(0)
    if max_depth == 0:
        print('return at depth 0')
        return m, w
    m_new = np.zeros((m.shape[0], largek))
    w_new = np.zeros((largek, w.shape[1]))
    # the size of each sub-cluster
    n_k = largek/smallk
    for i in range(smallk):
        # TODO: how to deal with uncertain (high entropy) cells?
        # soft-cluster n percentile of the cells with the highest entropy
        # (include them in both subsets),
        # after returning, use the sub-cluster with lower entropy.
        data_c0 = data[:,clusters_0==i]
        m_s1, w_s1 = run_partition(data_c0, smallk, largek/2, method, max_depth-1)
        print(m_s1.shape)
        print(w_s1.shape)
        # place the sub-results for m and w back into the big one
        k_range = range(i*n_k, (i+1)*n_k)
        m_new[:,k_range] = m_s1
        w_new[np.ix_(k_range, clusters_0==i)] = w_s1
    return m_new, w_new


if __name__ == '__main__':
    l1 = np.array(range(100))
    l2 = l1[::-1]
    l3 = np.copy(l1)
    np.random.shuffle(l3)
    print(m_ndcg(l1,l2,l3))
    print(m_ndcg(l1,l1,l1))
    print(m_ndcg(l1,l2,l1))
    print(m_ndcg(l1,l2,l2))
    print(m_ndcg(l1,l3,l3))
    print(m_ndcg(l1,l1,l3))
    exit(0)
    X1 = scipy.io.mmread('data_8000_cells.mtx')
    X1 = X1.tocsc()
    true_labels1 = np.loadtxt('labels_8000_cells.txt').astype(int).flatten()

    k = 8
    frac = 0.2
    genes = uncurl.max_variance_genes(X1, nbins=5, frac=frac)
    data_subset = X1[genes,:]
    n_genes = data_subset.shape[0]

    # TODO: run uncurl

    se_mw = uncurl.experiment_runner.PoissonSE(clusters=k, return_m=True)
    se_mw_2 = uncurl.experiment_runner.PoissonSE(clusters=2, return_m=True)

    # layer 1:
    t0 = time.time()
    print('starting recursive uncurl')
    m, w = run_partition(data_subset, 2, 8, se_mw_2, 2)
    print('time elapsed: {0}'.format(time.time() - t0))
    print('nmi: {0}'.format(nmi(w.argmax(0), true_labels1)))

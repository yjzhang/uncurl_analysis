import time

from uncurl_analysis.bulk_data import bulk_lookup

from scipy import sparse
from scipy.io import loadmat
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

if __name__ == '__main__':

    dat = loadmat('data/10x_pooled_400.mat')
    data = sparse.csc_matrix(dat['data'])
    data_dense = data.toarray()
    labs = dat['labels'].flatten()
    bulk_means = {}
    for x in set(labs):
        means = np.array(data[:,labs==x].mean(1)).flatten()
        bulk_means[x] = means/means.sum()

    t0 = time.time()
    labels = []
    for i in range(data_dense.shape[1]):
        cell = data_dense[:,i]
        scores = bulk_lookup(bulk_means, cell)
        labels.append(scores[0][0])
    nmi_val = nmi(labs, labels)
    print('poisson dense lookup: time: ' + str(time.time() - t0))

    t0 = time.time()
    labels = []
    for i in range(data.shape[1]):
        cell = data[:,i]
        scores = bulk_lookup(bulk_means, cell)
        labels.append(scores[0][0])
    nmi_val = nmi(labs, labels)
    print('poisson sparse lookup: time: ' + str(time.time() - t0))


    t0 = time.time()
    labels = []
    for i in range(data_dense.shape[1]):
        cell = data_dense[:,i]
        scores = bulk_lookup(bulk_means, cell, method='cosine')
        labels.append(scores[0][0])
    nmi_val = nmi(labs, labels)
    print('cosine lookup: time: ' + str(time.time() - t0))

    t0 = time.time()
    labels = []
    for i in range(data_dense.shape[1]):
        cell = data_dense[:,i]
        scores = bulk_lookup(bulk_means, cell, method='rank_corr')
        labels.append(scores[0][0])
    nmi_val = nmi(labs, labels)
    print('rank_corr lookup: time: ' + str(time.time() - t0))

    t0 = time.time()
    labels = []
    for i in range(data_dense.shape[1]):
        cell = data_dense[:,i]
        scores = bulk_lookup(bulk_means, cell, method='corr')
        labels.append(scores[0][0])
    nmi_val = nmi(labs, labels)
    print('corr lookup: time: ' + str(time.time() - t0))

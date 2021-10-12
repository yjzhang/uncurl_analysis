import numpy as np 
from scipy.io import loadmat
from scipy import sparse 

import random 

import mnnpy

dat = loadmat('data/10x_pooled_400.mat')
data = sparse.csc_matrix(dat['data'])
labs = dat['labels'].flatten()

indices = list(range(data.shape[1])) 
random.shuffle(indices) 
batch1 = indices[:200]
batch2 = indices[200:] 
data1 = data[:, batch1]
data2 = data[:, batch2]
data1_dense = data1.T.toarray()
data2_dense = data2.T.toarray()
var_index = list(range(data1.shape[0]))
data_corrected = mnnpy.mnn_correct(data1.T, data2.T, var_index=var_index)

data_corrected_dense = mnnpy.mnn_correct(data1_dense, data2_dense, var_index=var_index)


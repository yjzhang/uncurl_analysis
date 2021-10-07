import numpy as np 
from scipy.io import loadmat
from scipy import sparse 

import random 

dat = loadmat('data/10x_pooled_400.mat')
data = sparse.csc_matrix(dat['data'])
labs = dat['labels'].flatten()

indices = list(range(data.shape[1])) 
random.shuffle(indices) 
batch1 = indices[:200]
batch2 = indices[200:] 
data1 = data[:, batch1]
data2 = data[:, batch2]
data_corrected = batch_correct_mnn([data1, data2]) 


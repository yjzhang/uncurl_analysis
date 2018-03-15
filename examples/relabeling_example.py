# re-initializing uncurl 

# splitting/merging clusters

import numpy as np
import scipy.io
import uncurl

data = scipy.io.loadmat('data/10x_pooled_400.mat')

data_csc = data['data']
labels = data['labels']
gene_names = data['gene_names']

# TODO

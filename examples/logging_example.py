import os
import shutil

from uncurl_analysis import sc_analysis

import scipy
from scipy import sparse
from scipy.io import loadmat

dat = loadmat('data/10x_pooled_400.mat')
data_dir = '/tmp/uncurl_analysis/test'
try:
    shutil.rmtree(data_dir)
    os.makedirs(data_dir)
except:
    os.makedirs(data_dir)
data = sparse.csc_matrix(dat['data'])
# take subset of max variance genes
scipy.io.mmwrite(os.path.join(data_dir, 'data.mtx'), data)
shutil.copy('data/10x_pooled_400_gene_names.tsv', os.path.join(data_dir, 'gene_names.txt'))

sca = sc_analysis.SCAnalysis(data_dir,
        frac=0.2,
        clusters=8,
        data_filename='data.mtx',
        baseline_dim_red='tsvd',
        dim_red_option='MDS',
        cell_frac=1.0,
        max_iters=20,
        inner_max_iters=10)


sca.run_full_analysis()
original_labels = sca.labels.copy()
print(original_labels)
original_w = sca.w.copy()
print(original_w)
# split two clusters....
clusters = sca.labels
sca.recluster('merge', [0, 1], write_log_entry=True)
sca.run_post_analysis()
sca.recluster('split', [0], write_log_entry=True)
sca.run_post_analysis()
# TODO: check history
log = sca.log
print(log)
entry = log[0]
entry2 = log[1]
# try to re-load?
sca.restore_prev(entry[1])
print(original_labels)
print(sca.labels)


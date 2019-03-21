# differential expression on Islam et al. 2011 data
import pickle

import pandas as pd

import scipy.io

from uncurl_analysis.gene_extraction import find_overexpressed_genes, pairwise_t, c_scores_from_t

from uncurl.preprocessing import log1p

# step 1: load data
data = scipy.io.loadmat('data/SCDE_k2_sup.mat')
data_csc = data['dat']
labels = data['Lab'].flatten()

# note: labels are 1 and 2 - which corresponds to ES and which corresponds to MEF?

table = pd.read_table('data/GSE29087_L139_expression_tab.txt')
table_data = table.iloc[7:, 7:]
# load gene names
table_gene_names = table.iloc[7:,0]

assert(len(table_gene_names)==data_csc.shape[0])

# do a t-test
import time
t0 = time.time()
#data_csc.data = data_csc.data**2
scores, pvals = pairwise_t(data_csc, labels, eps=0.1)

# TODO: calculate implied ratios from scores vs actual ratios
implied_ratios = 2**scores[1,0,:]
actual_ratios = (data_csc[:,labels==2].mean(1) + 0.1)/(data_csc[:,labels==1].mean(1) + 0.1)
import numpy as np
actual_ratios = np.array(actual_ratios).flatten()

# calculate p-values
c_scores, c_pvals = c_scores_from_t(scores, pvals)
print('c score time: ', time.time() - t0)

# step 3: map gene ids to gene names
new_pvals = {k:[] for k in c_pvals.keys()}
for k, p in c_pvals.items():
    for gene_id, pv in p:
        new_pvals[k].append((table_gene_names.iloc[gene_id], pv))

# save new_pvals
with open('scde_c_score_pvals.pkl', 'wb') as f:
    pickle.dump(new_pvals, f)

with open('scde_c_scores.pkl', 'wb') as f:
    pickle.dump(c_scores, f)

# use rdata to load benchmark from scde
import rpy2.robjects as robjects

robjects.r['load']("data/benchmark.RData")
genes = robjects.r['gs']
top_1000_genes = list(genes)
plist = robjects.r['plist']
scde_genes = list(plist[0][1])
deseq_genes = list(plist[1][8])
cuffdiff2_genes = list(plist[2][0])

top_cs_genes = [x[0] for x in new_pvals[1] if x[1]<=0.05]


cs_genes_intersection = len(set(top_cs_genes).intersection(top_1000_genes))
print(cs_genes_intersection)
scde_genes_intersection = len(set(scde_genes).intersection(top_1000_genes))
print(scde_genes_intersection)
deseq_genes_intersection = len(set(deseq_genes).intersection(top_1000_genes))
print(deseq_genes_intersection)
cuffdiff2_genes_intersection = len(set(cuffdiff2_genes).intersection(top_1000_genes))
print(cuffdiff2_genes_intersection)

from sklearn.metrics import roc_curve, auc
# TODO: plot ROC curve for the permutation c-score

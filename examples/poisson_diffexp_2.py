
import time

import numpy as np
import pandas as pd
from uncurl_analysis import poisson_diffexp
import scipy.io 
import uncurl 

# step 1: load data
data = scipy.io.loadmat('data/SCDE_k2_sup.mat')
data_csc = data['dat']
data_csc = uncurl.preprocessing.cell_normalize(data_csc)
labels = data['Lab'].flatten()

# note: labels are 1 and 2 - which corresponds to ES and which corresponds to MEF?

table = pd.read_table('data/GSE29087_L139_expression_tab.txt')
table_data = table.iloc[7:, 7:]
# load gene names
table_gene_names = table.iloc[7:,0]

assert(len(table_gene_names)==data_csc.shape[0])

t0 = time.time()
all_pvs, all_ratios, clusters_to_groups = poisson_diffexp.poisson_test_known_groups(data_csc, labels, test_mode='1_vs_rest')
print('diffexp time: ', time.time() - t0)

gene_names_index = {x:i for i, x in enumerate(table_gene_names)}

# step 3: map gene ids to gene names
top_genes = all_pvs.argsort(0)
top_ratios = all_ratios.argsort(0)[::-1]
new_pvals = {k: [] for k in clusters_to_groups.values()}
for k in range(all_pvs.shape[1]):
    for gene_id, pv in zip(top_genes[:,k], all_pvs[:,k][top_genes[:,k]]):
        new_pvals[clusters_to_groups[k]].append((table_gene_names.iloc[gene_id], pv))

top_ratio_genes = {k : [(table_gene_names.iloc[i], all_ratios[i, k]) for i in top_ratios[:,k]] for k in range(all_pvs.shape[1])}

# use rdata to load benchmark from scde
import rpy2.robjects as robjects

robjects.r['load']("data/benchmark.RData")
genes = robjects.r['gs']
# top_1000_genes is the 'ground truth' list of genes.
top_1000_genes = list(genes)
plist = robjects.r['plist']
scde_genes = list(plist[0][1])
deseq_genes = list(plist[1][8])
cuffdiff2_genes = list(plist[2][0])

top_cs_genes = [x[0] for x in new_pvals[2] if x[1]<=0.05]
top_ratio_genes_subset = [x[0] for x in top_ratio_genes[1]][:1000]


cs_genes_intersection = len(set(top_cs_genes).intersection(top_1000_genes))
print(cs_genes_intersection)
scde_genes_intersection = len(set(scde_genes).intersection(top_1000_genes))
print(scde_genes_intersection)
deseq_genes_intersection = len(set(deseq_genes).intersection(top_1000_genes))
print(deseq_genes_intersection)
cuffdiff2_genes_intersection = len(set(cuffdiff2_genes).intersection(top_1000_genes))
print(cuffdiff2_genes_intersection)



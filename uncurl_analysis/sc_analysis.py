import json
import os
import time

import numpy as np
import scipy.io
from scipy import sparse
import uncurl
#from uncurl.sparse_utils import symmetric_kld

from . import gene_extraction, relabeling, sparse_matrix_h5, dense_matrix_h5, custom_cell_selection
from .entropy import entropy

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS

import simplex_sample

DIM_RED_OPTIONS = ['mds', 'tsne', 'tsvd', 'pca', 'umap']

CLUSTERING_METHODS = ['argmax', 'louvain', 'leiden', 'leiden_baseline']

class SimpleEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


class SCAnalysis(object):
    """
    This class represents an ongoing single-cell RNA-Seq analysis.
    """
    # TODO: re-design this class to have more of a plugin-like architecture
    # have objects representing each of the analysis things that can be added
    # to the SCAnalysis object at run-time

    def __init__(self, data_dir,
            data_filename='data.mtx',
            clusters=10,
            data_is_sparse=True,
            normalize=False,
            min_reads=0,
            max_reads=1e10,
            frac=0.2,
            cell_frac=1.0,
            dim_red_option='tsne',
            baseline_dim_red=None,
            clustering_method='argmax',
            one_vs_all_test='t',
            use_fdr=False,
            min_unique_genes=0,
            max_unique_genes=1e10,
            max_mt_frac=1.0,
            **uncurl_kwargs):
        """
        Args:
            data_dir (str): directory where data is stored
        """
        # TODO: if sc_analysis.json already exists, load it.
        # note: each field contains file names, and whether or not
        # the analysis is complete.
        self.data_dir = data_dir
        # params is a dict of parameters...
        self.params = {}
        self.params['clusters'] = int(clusters)
        self.params['min_reads'] = int(min_reads)
        self.params['max_reads'] = int(max_reads)
        self.params['normalize'] = bool(normalize)
        self.params['is_sparse'] = data_is_sparse
        self.params['genes_frac'] = float(frac)
        self.params['cell_frac'] = float(cell_frac)
        self.params['dim_red_option'] = dim_red_option.lower()
        self.params['baseline_dim_red'] = baseline_dim_red
        self.params['clustering_method'] = clustering_method.lower()
        self.params['one_vs_all_test'] = one_vs_all_test
        self.params['use_fdr'] = bool(use_fdr)
        self.params['min_unique_genes'] = int(min_unique_genes)
        self.params['max_unique_genes'] = int(max_unique_genes)
        self.params['max_mt_frac'] = float(max_mt_frac)

        self.uncurl_kwargs = uncurl_kwargs

        self.data_f = os.path.join(data_dir, data_filename)
        self._data = None
        self._data_subset = None
        self._data_normalized = None

        if not os.path.exists(self.data_f):
            df1 = os.path.join(data_dir, 'data.mtx')
            df2 = os.path.join(data_dir, 'data.mtx.gz')
            df3 = os.path.join(data_dir, 'data.txt')
            df4 = os.path.join(data_dir, 'data.txt.gz')
            if os.path.exists(df1):
                self.data_f = df1
            elif os.path.exists(df2):
                self.data_f = df2
            elif os.path.exists(df3):
                self.data_f = df3
            elif os.path.exists(df4):
                self.data_f = df4
        print(self.data_f)

        self.data_sampled_all_genes_f = os.path.join(data_dir,
                'data_sampled_all_genes.h5')
        self.has_data_sampled_all_genes = os.path.exists(self.data_sampled_all_genes_f)
        self._data_sampled_all_genes = None

        self.read_counts_f = os.path.join(data_dir, 'read_counts.npy')
        self.has_read_counts = os.path.exists(self.read_counts_f)
        self._read_counts = None

        self.init_f = os.path.join(data_dir, 'init.txt')
        self.has_init = os.path.exists(self.init_f)
        self._init = None

        self.gene_names_f = os.path.join(data_dir, 'gene_names.txt')
        if not os.path.exists(self.gene_names_f):
            gf1 = os.path.join(data_dir, 'genes.csv')
            if os.path.exists(gf1):
                self.gene_names_f = gf1
        self._gene_names = None

        self.gene_subset_f = os.path.join(data_dir, 'gene_subset.txt')
        self.has_gene_subset = os.path.exists(self.gene_subset_f)
        self._gene_subset = None


        self.w_f = os.path.join(data_dir, 'w.txt')
        self.has_w = os.path.exists(self.w_f)
        self._w = None

        self.w_sampled_f = os.path.join(data_dir, 'w_sampled.txt')
        self.has_w_sampled = os.path.exists(self.w_sampled_f)
        self._w_sampled = None

        self.labels_f = os.path.join(data_dir, 'labels.txt')
        self.has_labels = os.path.exists(self.labels_f)
        self._labels = None

        self.cluster_means_f = os.path.join(data_dir, 'cluster_means.txt')
        self.has_cluster_means = os.path.exists(self.cluster_means_f)
        self._cluster_means = None

        self.cluster_names_f = os.path.join(data_dir, 'cluster_names.json')
        self.has_cluster_names = os.path.exists(self.cluster_names_f)
        self._cluster_names = None

        # m from running uncurl
        self.m_f = os.path.join(data_dir, 'm.txt')
        self.has_m = os.path.exists(self.m_f)
        self._m = None

        # m with all genes
        self.m_full_f = os.path.join(data_dir, 'm_full.txt')
        self.has_m_full = os.path.exists(self.m_full_f)
        self._m_full = None

        # m from sampled genes
        self.m_sampled_f = os.path.join(data_dir, 'm_sampled.txt')
        self.has_m_sampled = os.path.exists(self.m_sampled_f)
        self._m_sampled = None

        self.cell_subset_f = os.path.join(data_dir, 'cells_subset.txt')
        self.has_cell_subset = os.path.exists(self.cell_subset_f)
        self._cell_subset = None

        self.cell_sample_f = os.path.join(data_dir, 'cell_sample.txt')
        self.has_cell_sample = os.path.exists(self.cell_sample_f)
        self._cell_sample = None

        self.baseline_vis_f = os.path.join(data_dir, 'baseline_vis.txt')
        self.has_baseline_vis = os.path.exists(self.baseline_vis_f)
        self._baseline_vis = None

        self.dim_red_f = os.path.join(data_dir, 'mds_data.txt')
        self.has_dim_red = os.path.exists(self.dim_red_f)
        self._dim_red = None

        self.mds_means_f = os.path.join(data_dir, 'mds_means.txt')
        self.has_mds_means = os.path.exists(self.mds_means_f)
        self._mds_means = None

        self.top_genes_f = os.path.join(data_dir, 'top_genes.txt')
        self.has_top_genes = os.path.exists(self.top_genes_f)
        self._top_genes = None

        self.pvals_f = os.path.join(data_dir, 'gene_pvals.txt')
        self.has_pvals = os.path.exists(self.pvals_f)
        self._pvals = None

        self.top_genes_1_vs_rest_f = os.path.join(data_dir, 'top_genes_1_vs_rest.txt')
        self.has_top_genes_1_vs_rest = os.path.exists(self.top_genes_1_vs_rest_f)
        self._top_genes_1_vs_rest = None
        self.pvals_1_vs_rest_f = os.path.join(data_dir, 'gene_pvals_1_vs_rest.txt')
        self.has_pvals_1_vs_rest = os.path.exists(self.pvals_1_vs_rest_f)
        self._pvals_1_vs_rest = None

        self.t_scores_f = os.path.join(data_dir, 't_scores.h5')
        self.has_t_scores = os.path.exists(self.t_scores_f)
        self._t_scores = None
        self.t_pvals_f = os.path.join(data_dir, 't_pvals.h5')
        self.has_t_pvals = os.path.exists(self.t_pvals_f)
        self._t_pvals = None
        self.separation_scores_f = os.path.join(data_dir, 'separation_scores.txt')
        self.has_separation_scores = os.path.exists(self.separation_scores_f)
        self._separation_scores = None
        self.separation_genes_f = os.path.join(data_dir, 'separation_genes.txt')
        self.has_separation_genes = os.path.exists(self.separation_genes_f)
        self._separation_genes = None

        self.entropy_f = os.path.join(data_dir, 'entropy.txt')
        self.has_entropy = os.path.exists(self.entropy_f)
        self._entropy = None

        # externally loaded color tracks
        self.color_tracks_f = os.path.join(data_dir, 'color_tracks.json')
        # dict of color tracks to (is_discrete, filename)
        self._color_tracks = None

        self.color_tracks_cache = {}

        # custom cell selections
        self.custom_selections_f = os.path.join(data_dir, 'custom_selections.json')
        self._custom_selections = None

        # dimensionality reduction of genes
        self.gene_dim_red_f = os.path.join(data_dir, 'gene_dim_red.txt')
        self.has_gene_dim_red = os.path.exists(self.gene_dim_red_f)
        self._gene_dim_red = None

        self.baseline_gene_dim_red_f = os.path.join(data_dir, 'baseline_gene_dim_red.txt')
        self.has_baseline_gene_dim_red = os.path.exists(self.baseline_gene_dim_red_f)
        self._baseline_gene_dim_red = None

        self.gene_clusters_f = os.path.join(data_dir, 'gene_clusters.txt')
        self.has_gene_clusters = os.path.exists(self.gene_clusters_f)
        self._gene_clusters = None

        # TODO: action log
        self.log_f = os.path.join(data_dir, 'log.txt')
        self.has_log = os.path.exists(self.log_f)
        self._log = None

        # dict of output_name : running time
        self.profiling = {}

        self.json_f = os.path.join(data_dir, 'sc_analysis.json')

    
    @property
    def log(self):
        """
        Log is a list of tuples of strings. We only log operations that change the dataset - merge, split, recluster, creation of new colormap, change of colormap, etc.
        Format: (operation, *params)
        """
        self.log_f = str(self.log_f)
        self._log = []
        if not os.path.exists(self.log_f):
            pass
        else:
            with open(self.log_f) as f:
                data = f.readlines()
            for l in data:
                l = [x.strip() for x in l.split('\t')]
                l[3] = bool(l[3])
                self._log.append(l)
        return self._log

    def write_log_entry(self, action, save_m_w=True):
        """
        Params:
            action (str): just a string describing the action. Could be anything.
            save_m_w (bool):

        Log format: tab-separated lines -
        <action>\t<id>\t<time>

        Action format:
        <Merge, split, delete, upload> <change
        """
        import datetime
        import uuid
        id = str(uuid.uuid4())
        dt = datetime.datetime.now()
        if save_m_w:
            w_filename = self.w_sampled_f + '_' + id
            m_filename = self.m_sampled_f + '_' + id
            np.savetxt(w_filename, self.w_sampled)
            np.savetxt(m_filename, self.m_sampled)
        entry = '{0}\t{1}\t{2}\t{3}\n'.format(action, id, str(dt), save_m_w)
        self.log.append((action, id, str(dt), save_m_w))
        with open(self.log_f, 'a') as f:
            f.write(entry)

    def restore_prev(self, action_id):
        """
        Restores a previous log entry.
        """
        import shutil
        log_ids = [x[1] for x in self.log]
        if action_id not in log_ids:
            print('Error: action_id not found')
            return 'Error: action_id not found'
        ind = log_ids.index(action_id)
        entry = self.log[ind]
        if not entry[3]:
            print('Error: restore not available for id')
            return 'Error: restore not available for id'
        w_filename = self.w_sampled_f + '_' + action_id
        m_filename = self.m_sampled_f + '_' + action_id
        self._w_sampled = None
        self._m_sampled = None
        shutil.copy(w_filename, self.w_sampled_f)
        shutil.copy(m_filename, self.m_sampled_f)
        self.run_post_analysis()
        return 1

    @property
    def data(self):
        """
        Data - either a sparse csc matrix or a dense numpy array, of shape
        (genes, cells)
        """
        if self._data is None:
            # fffff this is a python3/python2 thing
            self.data_f = str(self.data_f)
            print('loading data:', self.data_f)
            try:
                if self.params['is_sparse'] or self.data_f.endswith('.mtx') or self.data_f.endswith('.mtx.gz'):
                    self._data = scipy.io.mmread(self.data_f)
                    self._data = sparse.csc_matrix(self._data)
                else:
                    self._data = np.loadtxt(self.data_f)
                return self._data
            except Exception as e:
                print('Could not load data:', e)
                return None
        else:
            return self._data

    @property
    def init(self):
        """
        Initialization matrix...
        """
        if self._init is None:
            if not self.has_init:
                return None
            else:
                self._init = np.loadtxt(self.init_f)
                return self._init
        else:
            return self._init

    @property
    def gene_subset(self):
        """
        Gene subset (from max_variance_genes)
        """
        if self._gene_subset is None:
            if not self.has_gene_subset:
                t = time.time()
                data = self.data_normalized
                if 'genes_frac' not in self.params or not isinstance(self.params['genes_frac'], float):
                    self.params['genes_frac'] = 0.2
                gene_subset = uncurl.max_variance_genes(data, nbins=5,
                        frac=self.params['genes_frac'])
                gene_subset = np.array(gene_subset)
                np.savetxt(self.gene_subset_f, gene_subset, fmt='%d')
                self.has_gene_subset = True
                self.profiling['gene_subset'] = time.time() - t
            else:
                gene_subset = np.loadtxt(self.gene_subset_f, dtype=int)
            self._gene_subset = gene_subset
        return self._gene_subset

    @property
    def read_counts(self):
        """
        Total read counts for each cell
        """
        if self._read_counts is None:
            if not self.has_read_counts:
                self._read_counts = np.array(self.data.sum(0)).flatten()
                np.save(self.read_counts_f, self._read_counts)
                self.has_read_counts = True
            else:
                self._read_counts = np.load(self.read_counts_f)
        return self._read_counts

    @property
    def cell_subset(self):
        """
        Cell subset (array of booleans, based on read count filtering)
        """
        if self._cell_subset is None:
            if not self.has_cell_subset:
                print('determining cell_subset')
                t = time.time()
                read_counts = self.read_counts
                print('min_reads:', self.params['min_reads'], 'max_reads:', self.params['max_reads'])
                self._cell_subset = (read_counts >= self.params['min_reads']) & (read_counts <= self.params['max_reads'])
                if sparse.issparse(self.data):
                    unique_genes = self.data.getnnz(0)
                else:
                    unique_genes = np.count_nonzero(self.data, 0)
                print('min_unique_genes:', self.params['min_unique_genes'], 'max_unique_genes:', self.params['max_unique_genes'])
                self._cell_subset = self._cell_subset & (unique_genes >= self.params['min_unique_genes']) & (unique_genes <= self.params['max_unique_genes'])
                print('max_mt_frac:', self.params['max_mt_frac'])
                # filter by mt genes
                if self.params['max_mt_frac'] < 1:
                    mt_genes = map(lambda x: x.startswith('Mt-') or x.startswith('MT-') or x.startswith('mt-'), self.gene_names)
                    mt_genes = np.array(list(mt_genes))
                    if len(mt_genes) > 0:
                        mt_gene_counts = np.array(self.data[mt_genes, :].sum(0)).flatten()
                        mt_gene_frac = mt_gene_counts/read_counts
                        self._cell_subset = self._cell_subset & (mt_gene_frac <= self.params['max_mt_frac'])
                np.savetxt(self.cell_subset_f, self._cell_subset, fmt='%d')
                self.has_cell_subset = True
                self.profiling['cell_subset'] = time.time() - t
            else:
                self._cell_subset = np.loadtxt(self.cell_subset_f, dtype=bool)
        return self._cell_subset

    @property
    def data_normalized(self):
        """
        Data before gene filters, but after cell subset. Read count-normalized.
        """
        if self._data_normalized is None:
            if self.params['normalize']:
                data = self.data[:, self.cell_subset]
                self._data_normalized = uncurl.preprocessing.cell_normalize(data)
            else:
                self._data_normalized = self.data
        return self._data_normalized

    @property
    def data_subset(self):
        """
        Data after passed through the gene/cell filters
        """
        if self._data_subset is None:
            data = self.data_normalized
            self._data_subset = data[self.gene_subset, :]
        return self._data_subset


    @property
    def gene_names(self):
        """
        Array of gene names
        """
        if self._gene_names is None:
            if self.gene_names_f.endswith('genes.csv'):
                try:
                    import pandas as pd
                    # this captures the split-seq output format
                    if self.gene_names_f.endswith('genes.csv'):
                        gene_names = pd.read_csv(self.gene_names_f)
                    if 'gene_name' in gene_names.columns:
                        self._gene_names = gene_names.gene_name.values
                    elif 'gene_names' in gene_names.columns:
                        self._gene_names = gene_names.gene_name.values
                    else:
                        self._gene_names = np.loadtxt(self.gene_names_f, dtype=str)
                        # default gene names
                        if len(self._gene_names) <= 1:
                            self._gene_names = np.array(['gene_{0}'.format(i) for i in range(self.data.shape[0])])
                    return self._gene_names
                except:
                    try:
                        self._gene_names = np.loadtxt(self.gene_names_f, dtype=str)
                    except:
                        self._gene_names = np.array(['gene_{0}'.format(i) for i in range(self.data.shape[0])])
                    return self._gene_names
            else:
                try:
                    self._gene_names = np.loadtxt(self.gene_names_f, dtype=str)
                    if len(self._gene_names) <= 1:
                        self._gene_names = np.array(['gene_{0}'.format(i) for i in range(self.data.shape[0])])
                    return self._gene_names
                except:
                    # default gene names
                    self._gene_names = np.array(['gene_{0}'.format(i) for i in range(self.data.shape[0])])
                    return self._gene_names
        else:
            return self._gene_names

    def run_uncurl(self):
        """
        Runs uncurl on self.data_subset.
        """
        t = time.time()
        init = None
        if self.has_init:
            init = self.init
            # run qualNorm
            init = uncurl.qualNorm(self.data_subset, init)
        m, w, ll = uncurl.run_state_estimation(self.data_subset,
                clusters=self.params['clusters'],
                init_means=init,
                **self.uncurl_kwargs)
        np.savetxt(self.w_f, w)
        np.savetxt(self.m_f, m)
        self._m = m
        self._w = w
        self.has_w = True
        self.has_m = True
        self.profiling['uncurl'] = time.time() - t

    @property
    def m(self):
        if self._m is None:
            if self.has_m:
                m = np.loadtxt(self.m_f)
                self._m = m
            else:
                self.run_uncurl()
        return self._m

    @property
    def w(self):
        if self._w is None:
            if self.has_w:
                w = np.loadtxt(self.w_f)
                self._w = w
            else:
                self.run_uncurl()
        return self._w

    @property
    def w_sampled(self):
        if not self.has_w_sampled:
            w_sampled = self.w[:, self.cell_sample]
            return w_sampled
        else:
            if self._w_sampled is None:
                self._w_sampled = np.loadtxt(self.w_sampled_f)
            return self._w_sampled

    @property
    def m_sampled(self):
        if not self.has_m_sampled:
            return self.m
        else:
            if self._m_sampled is None:
                self._m_sampled = np.loadtxt(self.m_sampled_f)
            return self._m_sampled

    @property
    def m_full(self):
        if self._m_full is None:
            if self.has_m_full:
                m_full = np.loadtxt(self.m_full_f)
                self._m_full = m_full
            else:
                print('calculating m_full')
                m = self.m
                w = self.w
                # data contains cell subset, but not gene subset
                # TODO: this doesn't work for non-poisson distributions? Or does it?
                data = self.data[:, self.cell_subset]
                selected_genes = self.gene_subset
                self._m_full = uncurl.update_m(data, m, w, selected_genes,
                        **self.uncurl_kwargs)
                self.has_m_full = True
                np.savetxt(self.m_full_f, self._m_full)
        return self._m_full

    @property
    def labels(self):
        if not self.has_labels:
            if 'clustering_method' in self.params and self.params['clustering_method'] == 'louvain':
                from .clustering_methods import create_graph, run_louvain
                graph = create_graph(self.w_sampled.T, n_neighbors=20, metric='cosine')
                labels = run_louvain(graph)
                self._labels = np.array(labels)
            elif 'clustering_method' in self.params and self.params['clustering_method'] == 'leiden':
                from .clustering_methods import create_graph, run_leiden
                graph = create_graph(self.w_sampled.T, n_neighbors=20, metric='cosine')
                labels = run_leiden(graph)
                self._labels = np.array(labels)
            elif 'clustering_method' in self.params and self.params['clustering_method'] == 'leiden_baseline':
                from .clustering_methods import baseline_cluster
                labels = baseline_cluster(self.data_sampled)
                self._labels = np.array(labels)
            else:
                self._labels = self.w_sampled.argmax(0)
            np.savetxt(self.labels_f, self._labels, fmt='%d')
            self.has_labels = True
        else:
            if self._labels is None:
                self._labels = np.loadtxt(self.labels_f, dtype=int)
        return self._labels

    @property
    def cluster_means(self):
        """
        Means of the cells in each cluster over all genes.

        This is a 2d dense array of shape (genes, k).
        """
        if self._cluster_means is None:
            if self.has_cluster_means:
                self._cluster_means = np.loadtxt(self.cluster_means_f)
            else:
                data = self.data_sampled_all_genes
                means = np.zeros((data.shape[0], self.labels.max()+1))
                for lab in set(self.labels):
                    if sparse.issparse(data):
                        means[:, lab] = np.array(data[:, self.labels==lab].mean(1)).flatten()
                    else:
                        means[:, lab] = data[:, self.labels==lab].mean(1)
                np.savetxt(self.cluster_means_f, means)
                self._cluster_means = means
        return self._cluster_means

    @property
    def mds_means(self):
        """
        MDS of the post-uncurl cluster means
        """
        if self._mds_means is None:
            if self.has_mds_means:
                self._mds_means = np.loadtxt(self.mds_means_f)
            else:
                k = self.w_sampled.shape[0]
                self._mds_means = np.zeros((2, k))
                for i in range(k):
                    self._mds_means[:,i] = self.dim_red[:, self.labels==i].mean(1)
                np.savetxt(self.mds_means_f, self._mds_means)
                self.has_mds_means = True
        return self._mds_means

    @property
    def cell_sample(self):
        """
        Cell sample (after applying data subset) - based on uniform
        simplex sampling on W.
        """
        if self._cell_sample is None and self.params['cell_frac'] >= 1 and not os.path.exists(self.cell_sample_f):
            self._cell_sample = np.arange(sum(self.cell_subset))
        else:
            if self._cell_sample is None:
                if self.has_cell_sample:
                    self._cell_sample = np.loadtxt(self.cell_sample_f, dtype=int)
                else:
                    # TODO: balance between random sample and simplex sample
                    k, cells = self.w.shape
                    n_samples = int(cells*self.params['cell_frac'])
                    n_simplex_samples = int(n_samples/2)
                    n_rand_samples = n_samples - n_simplex_samples
                    samples = simplex_sample.sample(k, n_simplex_samples)
                    indices = simplex_sample.data_sample(self.w, samples,
                            replace=False)
                    unsampled_indices = [x for x in range(cells) if x not in set(indices)]
                    import random
                    indices_random = random.sample(unsampled_indices, n_rand_samples)
                    full_indices = np.concatenate([indices, indices_random])
                    full_indices.sort()
                    indices = full_indices
                    np.savetxt(self.cell_sample_f, indices, fmt='%d')
                    self.has_cell_sample = True
                    self._cell_sample = indices
        return self._cell_sample

    @property
    def data_sampled(self):
        """
        Data after passed through the gene/cell filters, and sampled.
        """
        data_subset = self.data_subset
        cell_sample = self.cell_sample
        return data_subset[:, cell_sample]

    @property
    def data_sampled_all_genes(self):
        """
        Data after passed through the gene/cell filters, and sampled. That is, displayed cells.
        """
        if self._data_sampled_all_genes is None:
            if not self.has_data_sampled_all_genes:
                self._data_sampled_all_genes = self.data[:, self.cell_subset][:, self.cell_sample]
                sparse_matrix_h5.store_matrix(self._data_sampled_all_genes,
                        self.data_sampled_all_genes_f)
            else:
                self._data_sampled_all_genes = sparse_matrix_h5.load_matrix(
                        self.data_sampled_all_genes_f)
                self._data_sampled_all_genes = sparse.csc_matrix(self._data_sampled_all_genes)
        return self._data_sampled_all_genes


    @property
    def baseline_vis(self):
        """
        baseline_vis is a non-uncurl-based 2D dimensionality reduction.
        shape: (2, n)
        """
        if self._baseline_vis is None:
            if self.has_baseline_vis:
                self._baseline_vis = np.loadtxt(self.baseline_vis_f)
            else:
                t = time.time()
                if self.params['baseline_dim_red'] is None or not isinstance(self.params['baseline_dim_red'], str):
                    baseline_dim_red = self.params['dim_red_option'].lower()
                else:
                    baseline_dim_red = self.params['baseline_dim_red'].lower()
                if baseline_dim_red == 'none':
                    return self.dim_red
                else:
                    # TODO: should we normalize data?
                    data_sampled = self.data_sampled
                    if self.params['normalize']:
                        from uncurl.preprocessing import cell_normalize
                        data_sampled = cell_normalize(data_sampled)
                    tsvd = TruncatedSVD(50)
                    data_log_norm = uncurl.preprocessing.log1p(data_sampled)
                    if baseline_dim_red == 'tsne':
                        data_tsvd = tsvd.fit_transform(data_log_norm.T)
                        tsne = TSNE(2)
                        data_dim_red = tsne.fit_transform(data_tsvd)
                    elif baseline_dim_red == 'tsvd' or baseline_dim_red == 'pca':
                        tsvd2 = TruncatedSVD(2)
                        data_dim_red = tsvd2.fit_transform(data_log_norm.T)
                    elif baseline_dim_red == 'mds':
                        data_tsvd = tsvd.fit_transform(data_log_norm.T)
                        mds = MDS(2)
                        data_dim_red = mds.fit_transform(data_tsvd)
                    elif baseline_dim_red == 'umap':
                        from umap import UMAP
                        um = UMAP()
                        data_tsvd = tsvd.fit_transform(data_log_norm.T)
                        data_dim_red = um.fit_transform(data_tsvd)
                    else:
                        raise Exception('dimensionality reduction name {0} is unknown'.format(baseline_dim_red))
                    self._baseline_vis = data_dim_red.T
                    np.savetxt(self.baseline_vis_f, self._baseline_vis)
                    self.has_baseline_vis = True
                self.profiling['baseline_vis'] = time.time() - t
        return self._baseline_vis

    @property
    def dim_red(self):
        """
        Uncurl-based dimensionality reduction
        """
        if self._dim_red is None:
            if self.has_dim_red:
                self._dim_red = np.loadtxt(self.dim_red_f)
            else:
                t = time.time()
                self.params['dim_red_option'] = self.params['dim_red_option'].lower()
                w = self.w_sampled
                if self.params['dim_red_option'] == 'mds':
                    self._dim_red = uncurl.mds(self.m_sampled, w, 2)
                elif self.params['dim_red_option'] == 'tsne':
                    # TODO: do we actually want to use the symmetric kld metric?
                    # it kills performance without doing much...
                    # metric=symmetric_kld
                    tsne = TSNE(2)
                    self._dim_red = tsne.fit_transform(w.T).T
                elif self.params['dim_red_option'] == 'pca':
                    pca = PCA(2)
                    self._dim_red = pca.fit_transform(w.T).T
                elif self.params['dim_red_option'] == 'tsvd':
                    tsvd = TruncatedSVD(2)
                    self._dim_red = tsvd.fit_transform(w.T).T
                elif self.params['dim_red_option'] == 'umap':
                    from umap import UMAP
                    um = UMAP(metric='cosine')
                    self._dim_red = um.fit_transform(w.T).T
                np.savetxt(self.dim_red_f, self._dim_red)
                self.has_dim_red = True
                self.profiling['dim_red'] = time.time() - t
        return self._dim_red

    @property
    def gene_dim_red(self):
        """dimensionality-reduced view of the genes """
        if self._gene_dim_red is None:
            if self.has_gene_dim_red:
                self._gene_dim_red = np.loadtxt(self.gene_dim_red_f)
            else:
                t = time.time()
                self.params['dim_red_option'] = self.params['dim_red_option'].lower()
                m = self.m_full
                # lmao this is overloaded in a way that makes zero sense
                m = m/(m.sum(1, keepdims=True) + 1e-8)
                if self.params['dim_red_option'] == 'umap' or self.params['dim_red_option'] == 'tsne':
                    from umap import UMAP
                    um = UMAP(metric='cosine')
                    self._gene_dim_red = um.fit_transform(m).T
                #elif self.params['dim_red_option'] == 'tsne':
                #    tsne = TSNE(2)
                #    self._gene_dim_red = tsne.fit_transform(m).T
                elif self.params['dim_red_option'] == 'tsvd' or self.params['dim_red_option'] == 'pca' or self.params['dim_red_option'] == 'mds':
                    tsvd = TruncatedSVD(2)
                    self._gene_dim_red = tsvd.fit_transform(m).T
                np.savetxt(self.gene_dim_red_f, self._gene_dim_red)
                self.has_gene_dim_red = True
                self.profiling['gene_dim_red'] = time.time() - t
        return self._gene_dim_red

    @property
    def gene_clusters(self):
        """Array of ints, of length genes, representing the cluster label of each gene."""
        if self._gene_clusters is None:
            if self.has_gene_clusters:
                self._gene_clusters = np.loadtxt(self.gene_clusters_f)
            else:
                m = self.m_full
                data = m/(m.sum(1, keepdims=True) + 1e-8)
                if 'clustering_method' in self.params and self.params['clustering_method'] == 'louvain':
                    from .clustering_methods import create_graph, run_louvain
                    graph = create_graph(data, n_neighbors=20, metric='cosine')
                    labels = run_louvain(graph)
                    self._gene_clusters = np.array(labels)
                elif 'clustering_method' in self.params and self.params['clustering_method'] == 'leiden':
                    from .clustering_methods import create_graph, run_leiden
                    graph = create_graph(data, n_neighbors=20, metric='cosine')
                    labels = run_leiden(graph)
                    self._gene_clusters = np.array(labels)
                elif 'clustering_method' in self.params and self.params['clustering_method'] == 'leiden_baseline':
                    from .clustering_methods import baseline_cluster
                    labels = baseline_cluster(data)
                    self._gene_clusters = np.array(labels)
                else:
                    # cluster using Leiden by default
                    from .clustering_methods import create_graph, run_leiden
                    graph = create_graph(data, n_neighbors=20, metric='cosine')
                    labels = run_leiden(graph)
                    self._gene_clusters = np.array(labels)
                np.savetxt(self.gene_clusters_f, self._gene_clusters, fmt='%d')
                self.has_gene_clusters = True
        return self._gene_clusters

    @property
    def t_scores(self):
        if self._t_scores is None:
            if self.has_t_scores:
                self._t_scores = dense_matrix_h5.load_array(self.t_scores_f)
            else:
                t = time.time()
                # this is complicated because we only want the cell subset,
                # not the gene subset...
                data = self.data_sampled_all_genes
                labels = self.labels
                # TODO: have some option for eps?
                self._t_scores, self._t_pvals = gene_extraction.pairwise_t(
                        data, labels,
                        eps=float(5*len(set(labels)))/data.shape[1],
                        normalize=self.params['normalize'],
                        use_fdr=self.params['use_fdr'])
                dense_matrix_h5.store_array(self.t_scores_f, self._t_scores)
                dense_matrix_h5.store_array(self.t_pvals_f, self._t_pvals)
                self.has_t_scores = True
                self.has_t_pvals = True
                self.profiling['t_scores'] = time.time() - t
        return self._t_scores

    def t_scores_view(self):
        """Returns a read-only view of the t-scores; doesn't require loading the whole array"""
        if self._t_scores is None:
            if self.has_t_scores:
                return dense_matrix_h5.load_array_view(self.t_scores_f)
        return self.t_scores

    @property
    def t_pvals(self):
        if self._t_pvals is None:
            if self.has_t_pvals:
                self._t_pvals = dense_matrix_h5.load_array(self.t_pvals_f)
            else:
                self.t_scores
        return self._t_pvals

    def t_pvals_view(self):
        """Returns a read-only view of the t-pval; doesn't require loading the whole array"""
        if self._t_pvals is None:
            if self.has_t_pvals:
                return dense_matrix_h5.load_array_view(self.t_pvals_f)
        return self.t_pvals

    @property
    def top_genes_1_vs_rest(self):
        if self._top_genes_1_vs_rest is None:
            if self.has_top_genes_1_vs_rest:
                with open(self.top_genes_1_vs_rest_f) as f:
                    self._top_genes_1_vs_rest = json.load(f)
                with open(self.pvals_1_vs_rest_f) as f:
                    self._pvals_1_vs_rest = json.load(f)
            else:
                t = time.time()
                data = self.data_sampled_all_genes
                labels = self.labels
                # this is due to a really bizarre up bug with json in python 3
                labels = labels.tolist()
                # TODO: have some option for eps?
                self._top_genes_1_vs_rest, self._pvals_1_vs_rest = gene_extraction.one_vs_rest_t(
                        data, labels,
                        eps=float(5*len(set(labels)))/data.shape[1],
                        test=self.params['one_vs_all_test'],
                        normalize=self.params['normalize'],
                        use_fdr=self.params['use_fdr'])
                with open(self.top_genes_1_vs_rest_f, 'w') as f:
                    json.dump(self._top_genes_1_vs_rest, f,
                            cls=SimpleEncoder)
                with open(self.pvals_1_vs_rest_f, 'w') as f:
                    json.dump(self._pvals_1_vs_rest, f,
                            cls=SimpleEncoder)
                self.has_top_genes_1_vs_rest = True
                self.has_pvals_1_vs_rest = True
                self.profiling['top_genes_1_vs_rest'] = time.time() - t
        if 0 not in self._top_genes_1_vs_rest.keys():
            new_top_genes = {}
            for k, v in self._top_genes_1_vs_rest.items():
                new_top_genes[int(k)] = v
            self._top_genes_1_vs_rest = new_top_genes
        return self._top_genes_1_vs_rest

    @property
    def pvals_1_vs_rest(self):
        if self._pvals_1_vs_rest is None:
            if self.has_pvals_1_vs_rest:
                with open(self.pvals_1_vs_rest_f) as f:
                    self._pvals_1_vs_rest = json.load(f)
            else:
                self.top_genes_1_vs_rest
        if 0 not in self._pvals_1_vs_rest.keys():
            new_pvals = {}
            for k, v in self._pvals_1_vs_rest.items():
                new_pvals[int(k)] = v
            self._pvals_1_vs_rest = new_pvals
        return self._pvals_1_vs_rest

    @property
    def separation_scores(self):
        if self._separation_scores is None:
            if self.has_separation_scores:
                self._separation_scores = np.loadtxt(self.separation_scores_f)
            else:
                self._separation_scores, self._separation_genes = gene_extraction.separation_scores_from_t(
                        self.t_scores, self.t_pvals)
                np.savetxt(self.separation_scores_f, self._separation_scores)
                np.savetxt(self.separation_genes_f, self._separation_genes, fmt='%d')
                self.has_separation_scores = True
                self.has_separation_genes = True
        return self._separation_scores

    @property
    def separation_genes(self):
        if self._separation_genes is None:
            if self.has_separation_genes:
                self._separation_genes = np.loadtxt(self.separation_genes_f, dtype=int)
            else:
                self.separation_scores
        return self._separation_genes

    @property
    def top_genes(self):
        """Dict of cluster : [(gene, c-score)...]"""
        if self._top_genes is None:
            if self.has_top_genes:
                with open(self.top_genes_f) as f:
                    self._top_genes = json.load(f)
                with open(self.pvals_f) as f:
                    self._pvals = json.load(f)
            else:
                self._top_genes, self._pvals = gene_extraction.c_scores_from_t(
                        self.t_scores, self.t_pvals)
                with open(self.top_genes_f, 'w') as f:
                    json.dump(self._top_genes, f,
                            cls=SimpleEncoder)
                with open(self.pvals_f, 'w') as f:
                    json.dump(self._pvals, f,
                            cls=SimpleEncoder)
                self.has_top_genes = True
                self.has_pvals = True
        if 0 not in self._top_genes.keys():
            new_top_genes = {}
            for k, v in self._top_genes.items():
                new_top_genes[int(k)] = v
            self._top_genes = new_top_genes
        return self._top_genes

    @property
    def pvals(self):
        """Dict of cluster : [(gene, p-val)...]"""
        if self._pvals is None:
            if self.has_pvals:
                with open(self.pvals_f) as f:
                    self._pvals = json.load(f)
            else:
                self.top_genes
        if 0 not in self._pvals.keys():
            new_pvals = {}
            for k, v in self._pvals.items():
                new_pvals[int(k)] = v
            self._pvals = new_pvals
        return self._pvals

    @property
    def entropy(self):
        if self._entropy is None:
            if self.has_entropy:
                self._entropy = np.loadtxt(self.entropy_f)
            else:
                self._entropy = entropy(self.w_sampled)
                np.savetxt(self.entropy_f, self._entropy)
                self.has_entropy = True
        return self._entropy

    def data_sampled_gene(self, gene_name, use_mw=False):
        """
        Returns vector containing the expression levels for each cell for
        the gene with the given name.

        If gene_name contains a comma, then it's assumed that the input is
        a list of gene names, and this will return the sum of the levels of
        the input genes.
        """
        if ',' in gene_name:
            gene_names = gene_name.split(',')
            data = self.data_sampled_gene(gene_names[0].strip())
            for g in gene_names[1:]:
                result = self.data_sampled_gene(g.strip(), use_mw)
                if len(result) > 0:
                    data += result
            return data
        gene_name_indices = np.where(self.gene_names == gene_name)[0]
        print('gene_name_indices:', gene_name_indices)
        if len(gene_name_indices) == 0:
            return []
        # TODO: what if there are multiple indices for a given gene name?
        if len(gene_name_indices) > 1:
            print('Warning: duplicate gene name detected. Returning sum of values of both genes.')
        gene_index = gene_name_indices[0]
        if use_mw:
            # we re-calculate the matrix multiplication every time...
            # and use caching to store the values???
            m = self.m_full
            w = self.w
            data_subset = m[gene_name_indices,:].dot(w).flatten()
            return data_subset[self.cell_sample]
        else:
            if os.path.exists(self.data_sampled_all_genes_f):
                return sparse_matrix_h5.load_row(
                        self.data_sampled_all_genes_f,
                        gene_index)
            else:
                data = self.data_sampled_all_genes
                if len(gene_name_indices) > 1:
                    data_gene = data[gene_name_indices, :].sum(0)
                else:
                    data_gene = data[gene_name_indices, :]
                if sparse.issparse(data_gene):
                    return data_gene.toarray().flatten()
                else:
                    return data_gene.flatten()

    @property
    def color_tracks(self):
        """
        Dict of color track name : tuple(is_discrete, filename)
        """
        if self._color_tracks is None:
            if os.path.exists(self.color_tracks_f):
                with open(self.color_tracks_f) as f:
                    self._color_tracks = json.load(f)
            else:
                self._color_tracks = {}
        return self._color_tracks

    def add_color_track(self, color_track_name, color_data, is_discrete=False):
        """
        Adds an external color track to the analysis, for viewing or diffexp calculations.

        color_data is a 1d numpy array.
        """
        if is_discrete:
            color_data = color_data.astype(str)
        # make color_track_name safe
        keep_chars = set(['-', '_', ' '])
        color_track_name = ''.join([c for c in color_track_name if c.isalnum() or (c in keep_chars)])
        color_track_filename = 'color_track_' + color_track_name[:50] + '.npy'
        color_track_filename = os.path.join(self.data_dir, color_track_filename)
        np.save(color_track_filename, color_data)
        self.color_tracks[color_track_name] = {'is_discrete': is_discrete, 'color_track_filename': color_track_filename}
        with open(self.color_tracks_f, 'w') as f:
            json.dump(self.color_tracks, f,
                    cls=SimpleEncoder)

    @property
    def custom_selections(self):
        """
        A dict of name : CustomColorMap object
        """
        if self._custom_selections is None:
            if os.path.exists(self.custom_selections_f):
                self._custom_selections = custom_cell_selection.load_json(self.custom_selections_f)
            else:
                self._custom_selections = {}
        return self._custom_selections


    def create_custom_selection(self, color_track_name, labels=None):
        """
        Create a new custom discrete color map with the given name.

        Args:
            color_track_name (str)
            labels (list): list of labels
        """
        if color_track_name in self.get_color_track_names():
            raise Exception('color track name already in use.')
        self.custom_selections[color_track_name] = custom_cell_selection.CustomColorMap(color_track_name)
        if labels is not None:
            self.custom_selections[color_track_name].labels = labels
        custom_cell_selection.save_json(self.custom_selections_f, self.custom_selections)

    def update_custom_color_track_label(self, color_track_name, label_name, label_criteria=None, color=None):
        """
        Re-writes the color track info with the new color track info...
        """
        color_track = self.custom_selections[color_track_name]
        has_updated_label = False
        for label in color_track.labels:
            if label.name == label_name:
                has_updated_label = True
                if label_criteria is not None:
                    label.criteria = label_criteria
                if color is not None and color != '#000000':
                    label.color = color
        # this is kind of a hack...
        if has_updated_label and label_criteria is None:
            return
        if not has_updated_label:
            new_label = custom_cell_selection.CustomLabel(label_name, label_criteria, color=color)
            color_track.labels.append(new_label)
        if color_track_name in self.color_tracks:
            results = self.color_tracks[color_track_name]
            # delete all temp files
            for k, v in results.items():
                os.remove(v)
            del self.color_tracks[color_track_name]
            with open(self.color_tracks_f, 'w') as f:
                json.dump(self.color_tracks, f,
                        cls=SimpleEncoder)
        custom_cell_selection.save_json(self.custom_selections_f, self._custom_selections)

    def get_color_track(self, color_track_name, return_colors=False):
        """
        Returns a tuple for a given color track name: data, is_discrete, where
        data is a 1d array, and is_discrete is a boolean.
        """
        if color_track_name in self.custom_selections:
            colormap = self.custom_selections[color_track_name]
            if return_colors:
                colors = colormap.get_colors()
                return colormap.label_cells(self), True, colors
            return colormap.label_cells(self), True
        elif color_track_name in self.color_tracks:
            if not isinstance(self.color_tracks[color_track_name], dict):
                is_discrete, filename = self.color_tracks[color_track_name]
            else:
                results = self.color_tracks[color_track_name]
                is_discrete = results['is_discrete']
                filename = results['color_track_filename']
            if is_discrete:
                data = np.load(filename).astype(str)
            else:
                data = np.load(filename)
            if len(data) > len(self.cell_sample):
                data = data[self.cell_subset][self.cell_sample]
            if return_colors:
                return data, is_discrete, None
            return data, is_discrete
        else:
            return None

    def get_color_track_names(self):
        """
        Returns all color track names
        """
        color_tracks_1 = set(self.color_tracks.keys())
        custom_selections = set(self.custom_selections.keys())
        # filter
        color_tracks_1.update(custom_selections)
        return list(color_tracks_1)

    def get_color_track_values(self, color_track_name):
        """
        Returns a set of values for a color track.
        """
        if color_track_name not in self.color_tracks:
            return None
        else:
            color_track, is_discrete = self.get_color_track(color_track_name)
            if is_discrete:
                return set(color_track)
            else:
                return  None

    def calculate_diffexp(self, color_track_name, mode='1_vs_rest', calc_pvals=True, eps=None):
        """
        Calculates 1 vs rest differential expression for a custom
        color track.
        """
        # first, try to retrieve results from disk...
        color_track, is_discrete = self.get_color_track(color_track_name)
        if not is_discrete:
            return None
        if color_track_name not in self.color_tracks:
            self.color_tracks[color_track_name] = {}
        results = self.color_tracks[color_track_name]
        if mode + '_scores' in results and mode + '_pvals' in results:
            scores_filename = results[mode + '_scores']
            pvals_filename = results[mode + '_pvals']
            if mode == '1_vs_rest':
                scores = dense_matrix_h5.H5Dict(scores_filename)
                pvals = dense_matrix_h5.H5Dict(pvals_filename)
            elif mode == 'pairwise':
                scores = dense_matrix_h5.H5Array(scores_filename)
                pvals = dense_matrix_h5.H5Array(pvals_filename)
            return scores, pvals
        # try to calculate diffexp, if values don't exist, and save results.
        scores_filename = os.path.join(self.data_dir,
                'diffexp_scores_' + color_track_name + '_' + mode + '_' + str(calc_pvals) + '.h5')
        pvals_filename = os.path.join(self.data_dir,
                'diffexp_pvals_' + color_track_name + '_' + mode + '_' + str(calc_pvals) + '.h5')
        data = self.data_sampled_all_genes
        if eps is None:
            eps = float(5*len(set(color_track)))/data.shape[1]
        if mode == '1_vs_rest':
            scores, pvals = gene_extraction.one_vs_rest_t(data, color_track,
                        eps=eps,
                        calc_pvals=calc_pvals, test='t',
                        normalize=self.params['normalize'],
                        use_fdr=self.params['use_fdr'])
            dense_matrix_h5.store_dict(scores_filename, scores)
            dense_matrix_h5.store_dict(pvals_filename, pvals)
        elif mode == 'pairwise':
            scores, pvals = gene_extraction.pairwise_t(data, color_track,
                        eps=eps,
                        calc_pvals=calc_pvals,
                        normalize=self.params['normalize'],
                        use_fdr=self.params['use_fdr'])
            dense_matrix_h5.store_array(scores_filename, scores)
            dense_matrix_h5.store_array(pvals_filename, pvals)
        result = self.color_tracks[color_track_name]
        if isinstance(result, dict):
            self.color_tracks[color_track_name][mode + '_scores'] = scores_filename
            self.color_tracks[color_track_name][mode + '_pvals'] = pvals_filename
        else:
            self.color_tracks[color_track_name] = {'is_discrete': result[0], 'color_track_filename': result[1]}
            self.color_tracks[color_track_name][mode + '_scores'] = scores_filename
            self.color_tracks[color_track_name][mode + '_pvals'] = pvals_filename
        with open(self.color_tracks_f, 'w') as f:
            json.dump(self.color_tracks, f,
                    cls=SimpleEncoder)
        return scores, pvals

    @property
    def cluster_names(self):
        """
        annotated names for clusters???
        """
        # TODO


    def save_json_reset(self):
        """
        Removes all cached data, saves to json
        """
        for key, val in self.__dict__.items():
            if key.startswith('_'):
                self.__dict__[key] = None
        with open(self.json_f, 'w') as f:
            json.dump(self.__dict__, f,
                    cls=SimpleEncoder)


    def recluster(self, split_or_merge='split',
            clusters_to_change=[], write_log_entry=False):
        """
        Runs split, merge, or new cluster. Updates m and w.

        clusters_to_change can be a list of either cluster ids or cell ids.
        split_or_merge is one of 'split', 'merge', 'new', or 'delete'
        """
        # TODO: copy m and w - run write_log_entry (saves results before the action)
        if write_log_entry:
            action = split_or_merge + ' ' + ','.join([str(x) for x in clusters_to_change])
            self.write_log_entry(action, save_m_w=True)
        data_sampled = self.data_sampled
        if 'init_means' in self.uncurl_kwargs:
            del self.uncurl_kwargs['init_means']
        if 'init_weights' in self.uncurl_kwargs:
            del self.uncurl_kwargs['init_weights']
        m_new = self.m_sampled
        w_new = self.w_sampled
        if split_or_merge == 'split':
            self.params['clusters'] = w_new.shape[0] + 1
            c = clusters_to_change[0]
            m_new, w_new = relabeling.split_cluster(data_sampled, m_new, w_new,
                    c, **self.uncurl_kwargs)
        elif split_or_merge == 'merge':
            self.params['clusters'] = w_new.shape[0] - len(clusters_to_change) + 1
            m_new, w_new = relabeling.merge_clusters(data_sampled, m_new, w_new,
                    clusters_to_change, **self.uncurl_kwargs)
        elif split_or_merge == 'new':
            self.params['clusters'] = w_new.shape[0] + 1
            m_new, w_new = relabeling.new_cluster(data_sampled, m_new, w_new,
                    clusters_to_change, **self.uncurl_kwargs)
        elif split_or_merge == 'delete':
            print('deleting cells')
            m_new, w_new, cells_to_include = relabeling.delete_cells(data_sampled, m_new, w_new,
                    clusters_to_change, **self.uncurl_kwargs)
            # remove cells from cell_sample
            self._cell_sample = self.cell_sample[cells_to_include]
            self.has_cell_sample = True
            np.savetxt(self.cell_sample_f, self.cell_sample, fmt='%d')
            self._data_sampled_all_genes = None
            self.has_data_sampled_all_genes = False
            self.data_sampled_all_genes
            self._baseline_vis = None
            self.has_baseline_vis = False
            self.baseline_vis
        # set w_sampled
        self._w_sampled = w_new
        np.savetxt(self.w_sampled_f, w_new)
        self._m_sampled = m_new
        np.savetxt(self.m_sampled_f, m_new)
        self.has_w_sampled = True
        self.has_m_sampled = True

    def relabel(self, clustering_method='argmax'):
        """
        Re-generates the labels without creating a new dimensionality
        reduction or uncurl run.
        """
        print('relabeling...')
        self.params['clustering_method'] = clustering_method.lower()
        self._labels = None
        self.has_labels = False
        self.labels
        print('done with labels')
        self.has_cluster_means = False
        self._cluster_means = None
        self.cluster_means
        self.has_top_genes_1_vs_rest = False
        self._top_genes_1_vs_rest = None
        self.top_genes_1_vs_rest
        print('done with top_genes_1_vs_rest')
        self.has_t_scores = False
        self._t_scores = None
        self.t_scores
        print('done with t_scores')
        self.has_top_genes = False
        self._top_genes = None
        self.top_genes
        print('done with top_genes')
        self.has_entropy = False
        self._entropy = None
        self.entropy
        print('done with entropy')
        self.has_separation_scores = False
        self._separation_scores = None
        self.separation_scores
        self.save_params_json()

    def run_full_analysis(self):
        """
        Runs the whole analysis pipeline.
        """
        self.run_uncurl()
        self.labels
        self.baseline_vis
        self.dim_red
        self.mds_means
        self.top_genes_1_vs_rest
        self.top_genes
        self.pvals
        self.entropy
        self.separation_scores
        self.data_sampled_all_genes
        # self.gene_dim_red
        # self.gene_clusters

    def run_batch_effect_correction(self, color_track_name,
            write_log_entry=False, save_corrected_data=True):
        """
        TODO: run batch effect correction on a given colormap
        """
        from .batch_correction import batch_correct_mnn
        # 1. take data_normalized, label set and create new sub-matrices..
        if write_log_entry:
            action = 'batch ' + color_track_name
            self.write_log_entry(action, save_m_w=True)
        data = self.data_sampled_all_genes
        try:
            colormap, is_discrete = self.get_color_track(color_track_name)
        except:
            print('Error: colormap not found - ' + color_track_name)
            return None
        if not is_discrete:
            print('Error: colormap is not discrete - ' + color_track_name)
            return None
        # label cells
        data_list = []
        indices_list = []
        for color in np.unique(colormap):
            data_list.append(data[:, colormap==color])
            indices_list.append((color, colormap==color))
        result_data = batch_correct_mnn(data_list)
        # set data to be non-negative ?
        result_data[result_data < 0] = 0
        result_data = result_data.astype(np.double)
        # map indices back...
        data_remapped = result_data.copy()
        curr_ind = 0
        for color, indices in indices_list:
            n_cells = sum(indices)
            print(color, curr_ind, n_cells, result_data.shape, data_remapped.shape)
            data_remapped[:, indices] = result_data[:, curr_ind:curr_ind+n_cells]
            curr_ind += n_cells
        # save output data
        # TODO: should we do diffexp using the batch corrected data?
        self._data_subset = data_remapped[self.gene_subset, :]
        if save_corrected_data:
            self._data_sampled_all_genes = data_remapped
            sparse_matrix_h5.store_matrix(self._data_sampled_all_genes, self.data_sampled_all_genes_f)
        # re-run post_analysis
        self.run_uncurl()
        self.run_post_analysis(run_baseline_vis=save_corrected_data)
        return data_remapped

    def run_post_analysis(self, run_baseline_vis=False):
        """
        Re-runs the whole analysis except for uncurl - can be used after split/merge.

        No need to change the baseline vis.
        """
        print('running post analysis')
        self.has_labels = False
        self._labels = None
        self.labels
        print('done with labels')
        self.has_cluster_means = False
        self._cluster_means = None
        self.cluster_means
        if run_baseline_vis:
            self.has_baseline_vis = False
            self._baseline_vis = None
            self.baseline_vis
        self.has_dim_red = False
        self._dim_red = None
        self.dim_red
        print('done with dim_red')
        self.has_mds_means = False
        self._mds_means = None
        self.mds_means
        print('done with mds_means')
        self.has_top_genes_1_vs_rest = False
        self._top_genes_1_vs_rest = None
        self.top_genes_1_vs_rest
        print('done with top_genes_1_vs_rest')
        self.has_t_scores = False
        self._t_scores = None
        self.t_scores
        print('done with t_scores')
        self.has_top_genes = False
        self._top_genes = None
        self.top_genes
        print('done with top_genes')
        #self._pvals = None
        #self.pvals
        self.has_entropy = False
        self._entropy = None
        self.entropy
        print('done with entropy')
        self.has_separation_scores = False
        self._separation_scores = None
        self.separation_scores
        if self.has_m_full:
            self.has_m_full = False
            self._m_full = None
            self.m_full
        self.has_gene_dim_red = False
        self._gene_dim_red = None
        self.gene_dim_red
        self.has_gene_clusters = False
        self._gene_clusters = None
        self.gene_clusters

    def delete_uncurl_results(self, files_to_save=None):
        """
        Deletes all results based off uncurl from file.
        """
        # delete all files except the data, gene names, init.txt, and params.json
        # TODO: this shouldn't have to know so much stuff about the uncurl_app side
        if files_to_save is None:
            files_to_save = set([])
        else:
            files_to_save = set(files_to_save)
        files_to_save.update(['data.txt', 'data.txt.gz', 'data.mtx', 'data.mtx.gz', 'init.txt', 'gene_names.txt', 'genes.csv',
            'preprocess.json', 'color_tracks.json', 'vis_summary.html'])
        for key, val in self.color_tracks.items():
            pass
        # TODO: save uploaded color tracks, delete c
        for filename in os.listdir(self.data_dir):
            if filename not in files_to_save and not filename.startswith('color_track_') and not filename.startswith('diffexp_'):
                os.remove(os.path.join(self.data_dir, filename))

    def get_data_subset(self, cell_ids):
        """
        Returns a data matrix only consisting of the given cells
        (but all genes).
        """
        data_subset = self.data_sampled_all_genes[:, cell_ids]
        return sparse.csc_matrix(data_subset)

    def get_clusters_subset(self, cluster_ids):
        """
        Returns a data matrix only consisting of the given clusters
        (but all genes).
        """
        cell_ids = np.zeros(len(self.labels), dtype=bool)
        for cluster in cluster_ids:
            cell_ids = cell_ids | (self.labels == cluster)
        data_subset = self.data_sampled_all_genes[:, cell_ids]
        return sparse.csc_matrix(data_subset)

    def load_params_json(self):
        """
        loads params.json and uncurl_kwargs.json from json file
        """
        import numbers
        if os.path.exists(os.path.join(self.data_dir, 'params.json')):
            with open(os.path.join(self.data_dir, 'params.json')) as f:
                params = json.load(f)
                updated_params = {}
                # convert params that should be numbers into numbers
                for k, p in params.items():
                    if not isinstance(p, bool) and not isinstance(p, numbers.Number):
                        try:
                            updated_params[k] = int(p)
                        except:
                            try:
                                updated_params[k] = float(p)
                            except:
                                updated_params[k] = p
                self.params.update(updated_params)
                if 'frac' in params:
                    self.params['genes_frac'] = float(params['frac'])
        if os.path.exists(os.path.join(self.data_dir, 'uncurl_kwargs.json')):
            with open(os.path.join(self.data_dir, 'uncurl_kwargs.json')) as f:
                uncurl_kwargs = json.load(f)
                if 'write_progress_file' in uncurl_kwargs:
                    uncurl_kwargs['write_progress_file'] = os.path.join(self.data_dir, 'progress.txt')
                self.uncurl_kwargs = uncurl_kwargs

    def save_params_json(self):
        """
        this should be called whenever params are changed.
        """
        with open(os.path.join(self.data_dir, 'params.json'), 'w') as f:
            json.dump(self.params, f)

    def load_params_from_folder(self):
        """
        If there is a saved json file named sc_analysis.json in the json path, this loads all the parameters from the file, and adds them to the current object.

        Returns:
            SCAnalysis object loaded from self.data_dir.
        """
        if os.path.exists(self.json_f):
            with open(self.json_f) as f:
                p = json.load(f)
                p2 = p.copy()
                # don't override True values with False values...
                for key, val in p.items():
                    if key not in self.__dict__:
                        continue
                    if isinstance(key, str) and key.startswith('has_') and key in self.__dict__ and self.__dict__[key] is True:
                        del p2[key]
                    if val is None and self.__dict__[key] is not None:
                        del p2[key]
                    # ugh we want to use params.json, not the params in sc_analysis.json
                    if isinstance(key, str) and key == 'params' and os.path.exists(os.path.join(self.data_dir, 'params.json')):
                        del p2[key]
                self.__dict__.update(p2)
                if 'profiling' not in p:
                    self.profiling = {}
        self.load_params_json()
        return self

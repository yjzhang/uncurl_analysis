import json
import os
import time

import numpy as np
import scipy.io
from scipy import sparse
import uncurl
from uncurl.sparse_utils import symmetric_kld

from . import gene_extraction, relabeling, sparse_matrix_h5, dense_matrix_h5, custom_cell_selection
from .entropy import entropy

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS

import simplex_sample

DIM_RED_OPTIONS = ['MDS', 'tSNE', 'TSVD', 'PCA', 'UMAP']

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
            dim_red_option='mds',
            baseline_dim_red='none',
            pval_n_perms=50,
            **uncurl_kwargs):
        """
        Args:
            data_dir (str): directory where data is stored
        """
        # note: each field contains file names, and whether or not
        # the analysis is complete.
        self.data_dir = data_dir
        self.clusters = clusters
        self.min_reads = min_reads
        self.max_reads = max_reads
        self.normalize = normalize
        self.is_sparse = data_is_sparse
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

        self.frac = frac
        self.gene_subset_f = os.path.join(data_dir, 'gene_subset.txt')
        self.has_gene_subset = os.path.exists(self.gene_subset_f)
        self._gene_subset = None

        self.uncurl_kwargs = uncurl_kwargs

        self.w_f = os.path.join(data_dir, 'w.txt')
        self.has_w = os.path.exists(self.w_f)
        self._w = None

        self.w_sampled_f = os.path.join(data_dir, 'w_sampled.txt')
        self.has_w_sampled = os.path.exists(self.w_sampled_f)
        self._w_sampled = None

        self.labels_f = os.path.join(data_dir, 'labels.txt')
        self.has_labels = os.path.exists(self.labels_f)
        self._labels = None

        self.cluster_names_f = os.path.join(data_dir, 'cluster_names.txt')
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

        self.cell_frac = cell_frac
        self.cell_sample_f = os.path.join(data_dir, 'cell_sample.txt')
        self.has_cell_sample = os.path.exists(self.cell_sample_f)
        self._cell_sample = None

        self.baseline_dim_red = baseline_dim_red.lower()
        self.baseline_vis_f = os.path.join(data_dir, 'baseline_vis.txt')
        self.has_baseline_vis = os.path.exists(self.baseline_vis_f)
        self._baseline_vis = None

        self.dim_red_option = dim_red_option.lower()
        self.dim_red_f = os.path.join(data_dir, 'mds_data.txt')
        self.has_dim_red = os.path.exists(self.dim_red_f)
        self._dim_red = None

        self.mds_means_f = os.path.join(data_dir, 'mds_means.txt')
        self.has_mds_means = os.path.exists(self.mds_means_f)
        self._mds_means = None

        self.top_genes_f = os.path.join(data_dir, 'top_genes.txt')
        self.has_top_genes = os.path.exists(self.top_genes_f)
        self._top_genes = None

        self.pval_n_perms = pval_n_perms
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

        self._color_tracks_cache = {}

        # custom cell selections
        self.custom_selections_f = os.path.join(data_dir, 'custom_selections.json')
        self._custom_selections = None


        # dict of output_name : running time
        self.profiling = {}

        self.json_f = os.path.join(data_dir, 'sc_analysis.json')


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
                if self.is_sparse:
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
                data = self.data_normalized[:, self.cell_subset]
                gene_subset = uncurl.max_variance_genes(data, nbins=5,
                        frac=self.frac)
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
                t = time.time()
                data = self.data
                read_counts = np.array(data.sum(0)).flatten()
                self._cell_subset = (read_counts >= self.min_reads) & (read_counts <= self.max_reads)
                np.savetxt(self.cell_subset_f, self._cell_subset, fmt='%d')
                self.has_cell_subset = True
                self.profiling['cell_subset'] = time.time() - t
            else:
                self._cell_subset = np.loadtxt(self.cell_subset_f, dtype=bool)
        return self._cell_subset

    @property
    def data_normalized(self):
        """
        Data before gene/cell filters, but read count-normalized.
        """
        if self._data_normalized is None:
            if self.normalize:
                self._data_normalized = uncurl.preprocessing.cell_normalize(self.data)
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
            data_gene_subset = data[self.gene_subset, :]
            self._data_subset = data_gene_subset[:, self.cell_subset]
        return self._data_subset


    @property
    def gene_names(self):
        """
        Array of gene names
        """
        if self._gene_names is None:
            if self.gene_names_f.endswith('.csv'):
                import pandas as pd
                try:
                    gene_names = pd.read_csv(self.gene_names_f)
                    self._gene_names = gene_names.gene_name
                except:
                    # default gene names
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
                clusters=self.clusters,
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
            # TODO: should we catch this result? is it worthwhile?
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
            self._labels = self.w_sampled.argmax(0)
            np.savetxt(self.labels_f, self._labels, fmt='%d')
            self.has_labels = True
        else:
            if self._labels is None:
                self._labels = np.loadtxt(self.labels_f, dtype=int)
        return self._labels

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
        if self._cell_sample is None and self.cell_frac == 1:
            self._cell_sample = np.arange(self.w.shape[1])
        else:
            if self._cell_sample is None:
                if self.has_cell_sample:
                    self._cell_sample = np.loadtxt(self.cell_sample_f, dtype=int)
                else:
                    k, cells = self.w.shape
                    n_samples = int(cells*self.cell_frac)
                    samples = simplex_sample.sample(k, n_samples)
                    indices = simplex_sample.data_sample(self.w, samples,
                            replace=False)
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
        Data after passed through the gene/cell filters, and sampled.
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
                baseline_dim_red = self.baseline_dim_red.lower()
                if baseline_dim_red == 'none':
                    return self.dim_red
                else:
                    data_sampled = self.data_sampled
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
                self.dim_red_option = self.dim_red_option.lower()
                w = self.w_sampled
                if self.dim_red_option == 'mds':
                    self._dim_red = uncurl.mds(self.m_sampled, w, 2)
                elif self.dim_red_option == 'tsne':
                    tsne = TSNE(2, metric=symmetric_kld)
                    self._dim_red = tsne.fit_transform(w.T).T
                elif self.dim_red_option == 'pca':
                    pca = PCA(2)
                    self._dim_red = pca.fit_transform(w.T).T
                elif self.dim_red_option == 'tsvd':
                    tsvd = TruncatedSVD(2)
                    self._dim_red = tsvd.fit_transform(w.T).T
                elif self.dim_red_option == 'umap':
                    from umap import UMAP
                    um = UMAP(metric='cosine')
                    self._dim_red = um.fit_transform(w.T).T
                np.savetxt(self.dim_red_f, self._dim_red)
                self.has_dim_red = True
                self.profiling['dim_red'] = time.time() - t
        return self._dim_red

    @property
    def t_scores(self):
        if self._t_scores is None:
            if self.has_t_scores:
                self._t_scores = dense_matrix_h5.load_array(self.t_scores_f)
            else:
                t = time.time()
                # this is complicated because we only want the cell subset,
                # not the gene subset...
                if self.has_w_sampled:
                    data = self.data_sampled_all_genes
                    w = self.w_sampled
                else:
                    data = self.data[:, self.cell_subset]
                    w = self.w
                labels = w.argmax(0)
                # TODO: have some option for eps?
                self._t_scores, self._t_pvals = gene_extraction.pairwise_t(
                        data, labels,
                        eps=float(5*len(set(labels)))/data.shape[1])
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
                if self.has_w_sampled:
                    data = self.data_sampled_all_genes
                    w = self.w_sampled
                else:
                    data = self.data[:, self.cell_subset]
                    w = self.w
                labels = w.argmax(0)
                # this is due to a really bizarre up bug with json in python 3
                labels = labels.tolist()
                # TODO: have some option for eps?
                # TODO: test could be 't' or 'u'???
                self._top_genes_1_vs_rest, self._pvals_1_vs_rest = gene_extraction.one_vs_rest_t(
                        data, labels,
                        eps=float(5*len(set(labels)))/data.shape[1],
                        test='u')
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
            # TODO: this is causing an error
            for g in gene_names[1:]:
                result = self.data_sampled_gene(g.strip(), use_mw)
                data += result
            return data
        gene_name_indices = np.where(self.gene_names == gene_name)[0]
        if len(gene_name_indices) == 0:
            return []
        # TODO: what if there are multiple indices for a given gene name?
        if len(gene_name_indices) > 1:
            print('Warning: duplicate gene name detected. Returning arbitrary gene.')
        gene_index = gene_name_indices[0]
        if use_mw:
            # we re-calculate the matrix multiplication every time...
            # and use caching to store the values???
            m = self.m_full
            w = self.w
            data_subset = m[gene_index,:].dot(w).flatten()
            return data_subset[self.cell_sample]
        else:
            if os.path.exists(self.data_sampled_all_genes_f):
                return sparse_matrix_h5.load_row(
                        self.data_sampled_all_genes_f,
                        gene_index)
            else:
                data = self.data_sampled_all_genes
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
        # TODO: deal with criteria?
        if color_track_name in self.get_color_track_names():
            raise Exception('color track name already in use.')
        self.custom_selections[color_track_name] = custom_cell_selection.CustomColorMap(color_track_name)
        if labels is not None:
            self.custom_selections[color_track_name].labels = labels
        custom_cell_selection.save_json(self.custom_selections_f, self.custom_selections)

    def update_custom_color_track_label(self, color_track_name, label_name, label_criteria=None):
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
        # this is kind of a hack...
        if has_updated_label and label_criteria is None:
            return
        if not has_updated_label:
            new_label = custom_cell_selection.CustomLabel(label_name, label_criteria)
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

    def get_color_track(self, color_track_name):
        """
        Returns a tuple for a given color track name: data, is_discrete, where
        data is a 1d array, and is_discrete is a boolean.
        """
        if not hasattr(self, '_color_tracks_cache'):
            self._color_tracks_cache = {}
        try:
            data, is_discrete = self._color_tracks_cache[color_track_name]
            return data, is_discrete
        except:
            pass
        if color_track_name in self.custom_selections:
            colormap = self.custom_selections[color_track_name]
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
            self._color_tracks_cache[color_track_name] = (data, is_discrete)
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
                        calc_pvals=calc_pvals)
            dense_matrix_h5.store_dict(scores_filename, scores)
            dense_matrix_h5.store_dict(pvals_filename, pvals)
        elif mode == 'pairwise':
            scores, pvals = gene_extraction.pairwise_t(data, color_track,
                        eps=eps,
                        calc_pvals=calc_pvals)
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


    def save_json_reset(self):
        """
        Removes all cached data, saves to json
        """
        # TODO: use some metaprogramming to get around this
        self._data = None
        self._data_normalized = None
        self._data_subset = None
        self._data_sampled_all_genes = None
        self._gene_names = None
        self._gene_subset = None
        self._cell_sample = None
        self._w = None
        self._w_sampled = None
        self._m = None
        self._m_sampled = None
        self._mds_means = None
        self._labels = None
        self._cell_subset = None
        self._baseline_vis = None
        self._dim_red = None
        self._top_genes = None
        self._pvals = None
        self._top_genes_1_vs_rest = None
        self._pvals_1_vs_rest = None
        self._t_scores = None
        self._t_pvals = None
        self._separation_scores = None
        self._entropy = None
        self._separation_genes = None
        with open(self.json_f, 'w') as f:
            json.dump(self.__dict__, f,
                    cls=SimpleEncoder)


    def recluster(self, split_or_merge='split',
            clusters_to_change=[]):
        """
        Runs split, merge, or new cluster. Updates m and w.

        clusters_to_change can be a list of either cluster ids or cell ids.
        """
        data_sampled = self.data_sampled
        if 'init_means' in self.uncurl_kwargs:
            del self.uncurl_kwargs['init_means']
        if 'init_weights' in self.uncurl_kwargs:
            del self.uncurl_kwargs['init_weights']
        m_new = self.m_sampled
        w_new = self.w_sampled
        if split_or_merge == 'split':
            self.clusters = w_new.shape[0] + 1
            c = clusters_to_change[0]
            m_new, w_new = relabeling.split_cluster(data_sampled, m_new, w_new,
                    c, **self.uncurl_kwargs)
        elif split_or_merge == 'merge':
            self.clusters = w_new.shape[0] - len(clusters_to_change) + 1
            m_new, w_new = relabeling.merge_clusters(data_sampled, m_new, w_new,
                    clusters_to_change, **self.uncurl_kwargs)
        elif split_or_merge == 'new':
            self.clusters = w_new.shape[0] + 1
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

    def run_post_analysis(self):
        """
        Re-runs the whole analysis except for uncurl - can be used after split/merge.

        No need to change the baseline vis.
        """
        print('running post analysis')
        self.has_labels = False
        self._labels = None
        self.labels
        print('done with labels')
        #self.has_baseline_vis = False
        #self._baseline_vis = None
        #self.baseline_vis
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
        files_to_save.update(['data.txt', 'data.txt.gz', 'data.mtx', 'data.mtx.gz', 'init.txt', 'gene_names.txt', 'params.json',
            'preprocess.json', 'color_tracks.json', 'vis_summary.html'])
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


    def load_params_from_folder(self):
        """
        If there is a file called 'params.json' in the folder, this loads all the parameters from the file, and adds them to the current object.

        Returns:
            SCAnalysis object loaded from self.data_dir.
        """
        if os.path.exists(self.json_f):
            with open(self.json_f) as f:
                p = json.load(f)
                p2 = p.copy()
                # don't override True values with False values...
                for key, val in p.items():
                    if isinstance(key, str) and key.startswith('has_') and key in self.__dict__ and self.__dict__[key] is True:
                        del p2[key]
                self.__dict__.update(p2)
                if 'profiling' not in p:
                    self.profiling = {}
        if os.path.exists(os.path.join(self.data_dir, 'params.json')):
            with open(os.path.join(self.data_dir, 'params.json')) as f:
                params = json.load(f)
                if 'normalize_data' in params:
                    self.normalize = True
                try:
                    self.clusters = int(params['k'])
                    self.frac = float(params['genes_frac'])
                    self.cell_frac = float(params['cell_frac'])
                    self.min_reads = int(params['min_reads'])
                    self.max_reads = int(params['max_reads'])
                    self.baseline_dim_red = params['baseline_vismethod']
                    self.dim_red_option = params['vismethod']
                except:
                    pass
        if os.path.exists(os.path.join(self.data_dir, 'uncurl_kwargs.json')):
            with open(os.path.join(self.data_dir, 'uncurl_kwargs.json')) as f:
                uncurl_kwargs = json.load(f)
                self.uncurl_kwargs = uncurl_kwargs
        return self

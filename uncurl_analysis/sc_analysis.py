import json
import os
import time

import numpy as np
import scipy.io
from scipy import sparse
import uncurl
from uncurl.sparse_utils import symmetric_kld

from . import gene_extraction, relabeling
from .entropy import entropy

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS

import simplex_sample

DIM_RED_OPTIONS = ['MDS', 'tSNE', 'TSVD', 'PCA', 'UMAP']

class SCAnalysis(object):
    """
    This class represents an ongoing single-cell RNA-Seq analysis.
    """

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

        self.gene_names_f = os.path.join(data_dir, 'gene_names.txt')
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

        self.m_f = os.path.join(data_dir, 'm.txt')
        self.has_m = os.path.exists(self.m_f)
        self._m = None

        self.m_sampled_f = os.path.join(data_dir, 'm_sampled.txt')
        self.has_m_sampled = os.path.exists(self.m_sampled_f)
        self._m_sampled = None

        self.gene_subset_sampled_f = os.path.join(data_dir, 'gene_subset_sampled.txt')
        self.has_gene_subset_sampled = os.path.exists(self.gene_subset_sampled_f)
        self._gene_subset_sampled = None

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

        self.t_scores_f = os.path.join(data_dir, 't_scores.npy')
        self.has_t_scores = os.path.exists(self.t_scores_f)
        self._t_scores = None
        self.t_pvals_f = os.path.join(data_dir, 't_pvals.npy')
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
            try:
                if self.is_sparse:
                    self._data = scipy.io.mmread(self.data_f)
                    self._data = sparse.csc_matrix(self._data)
                else:
                    self._data = np.loadtxt(self.data_f)
                return self._data
            except:
                return None
        else:
            return self._data

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
            self._data_subset = data[np.ix_(self.gene_subset, self.cell_subset)]
        return self._data_subset


    @property
    def gene_names(self):
        """
        Array of gene names
        """
        if self._gene_names is None:
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
        m, w, ll = uncurl.run_state_estimation(self.data_subset,
                clusters=self.clusters,
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
            return self.w[:, self.cell_sample]
        else:
            if self._w_sampled is None:
                self._w_sampled = np.loadtxt(self.w_sampled_f)
            return self._w_sampled

    @property
    def m_sampled(self):
        if not self.has_m_sampled:
            return self.m[self.gene_subset_sampled, :]
        else:
            if self._m_sampled is None:
                self._m_sampled = np.loadtxt(self.m_sampled_f)
            return self._m_sampled

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
                self._mds_means = uncurl.dim_reduce(self.m_sampled,
                        self.w_sampled, 2).T
                np.savetxt(self.mds_means_f, self._mds_means)
                self.has_mds_means = True
        return self._mds_means

    @property
    def cell_sample(self):
        """
        Cell sample (after applying data subset) - based on uniform
        simplex sampling on W.
        """
        if self.cell_frac == 1:
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
    def gene_subset_sampled(self):
        if self._gene_subset_sampled is None:
            if not self.has_gene_subset_sampled:
                data_sampled = self.data_subset[:, self.cell_sample]
                gene_subset_sampled = uncurl.max_variance_genes(data_sampled, 1, 1)
                self._gene_subset_sampled = gene_subset_sampled
                np.savetxt(self.gene_subset_sampled_f, gene_subset_sampled, fmt='%d')
                self.has_gene_subset_sampled = True
            else:
                self._gene_subset_sampled = np.loadtxt(self.gene_subset_sampled_f, dtype=int)
        return self._gene_subset_sampled

    @property
    def data_sampled(self):
        """
        Data after passed through the gene/cell filters, and sampled.
        """
        data_subset = self.data_subset
        cell_sample = self.cell_sample
        return data_subset[np.ix_(self.gene_subset_sampled, cell_sample)]

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
                self._t_scores = np.load(self.t_scores_f)
            else:
                t = time.time()
                if self.has_w_sampled:
                    data = self.data[:, self.cell_sample]
                    w = self.w_sampled
                else:
                    data = self.data
                    w = self.w
                self._t_scores, self._t_pvals = gene_extraction.pairwise_t(
                        data, w)
                np.save(self.t_scores_f, self._t_scores)
                np.save(self.t_pvals_f, self._t_pvals)
                self.has_t_scores = True
                self.has_t_pvals = True
                self.profiling['t_scores'] = time.time() - t
        return self._t_scores

    @property
    def t_pvals(self):
        if self._t_pvals is None:
            if self.has_t_pvals:
                self._t_pvals = np.load(self.t_pvals_f)
            else:
                self.t_scores
        return self._t_pvals

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
                    json.dump(self._top_genes, f)
                with open(self.pvals_f, 'w') as f:
                    json.dump(self._pvals, f)
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

    def save_json_reset(self):
        """
        Removes all cached data, saves to json
        """
        self._data = None
        self._data_normalized = None
        self._data_subset = None
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
        self._gene_subset_sampled = None
        self._baseline_vis = None
        self._dim_red = None
        self._top_genes = None
        self._pvals = None
        self._t_scores = None
        self._t_pvals = None
        self._separation_scores = None
        self._entropy = None
        self._separation_genes = None
        with open(self.json_f, 'w') as f:
            json.dump(self.__dict__, f)


    def recluster(self, split_or_merge='split',
            clusters_to_change=[]):
        """
        Runs split or merge
        """
        data_sampled = self.data_sampled
        if 'init_means' in self.uncurl_kwargs:
            del self.uncurl_kwargs['init_means']
        if 'init_weights' in self.uncurl_kwargs:
            del self.uncurl_kwargs['init_weights']
        m_new = self.m_sampled
        w_new = self.w_sampled
        if split_or_merge == 'split':
            self.clusters += 1
            c = clusters_to_change[0]
            m_new, w_new = relabeling.split_cluster(data_sampled, m_new, w_new,
                    c, **self.uncurl_kwargs)
        elif split_or_merge == 'merge':
            self.clusters -= len(clusters_to_change) + 1
            m_new, w_new = relabeling.merge_clusters(data_sampled, m_new, w_new,
                    clusters_to_change, **self.uncurl_kwargs)
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
        self.mds_means
        self.baseline_vis
        self.dim_red
        self.top_genes
        self.pvals
        self.entropy
        self.separation_scores

    def run_post_analysis(self):
        """
        Re-runs the whole analysis except for uncurl - can be used after split/merge.

        No need to change the baseline vis.
        """
        self.has_labels = False
        self._labels = None
        self.labels
        self.has_mds_means = False
        self._mds_means = None
        self.mds_means
        #self.has_baseline_vis = False
        #self._baseline_vis = None
        #self.baseline_vis
        self.has_dim_red = False
        self._dim_red = None
        self.dim_red
        self.has_t_scores = False
        self._t_scores = None
        self.t_scores
        self.has_top_genes = False
        self._top_genes = None
        self.top_genes
        #self._pvals = None
        #self.pvals
        self.has_entropy = False
        self._entropy = None
        self.entropy
        self.has_separation_scores = False
        self._separation_scores = None
        self.separation_scores

    def load_params_from_folder(self):
        """
        If there is a file called 'params.json' in the folder, this loads all the parameters from the file, and overwrites the current object.

        If a saved pickle file exists in the folder, this returns that pickle object.

        Returns:
            SCAnalysis object loaded from self.data_dir.
        """
        if os.path.exists(self.json_f):
            with open(self.json_f) as f:
                p = json.load(f)
                self.__dict__ = p
                if 'profiling' not in p:
                    self.profiling = {}
                return self
        if os.path.exists(os.path.join(self.data_dir, 'params.json')):
            with open(os.path.join(self.data_dir, 'params.json')) as f:
                params = json.load(f)
                if 'normalize_data' in params:
                    self.normalize = True
                try:
                    self.clusters = int(params['k'])
                    self.frac = float(params['gene_frac'])
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

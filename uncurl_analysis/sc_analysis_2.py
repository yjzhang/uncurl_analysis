import json
import os
import time

import numpy as np
import scipy.io
from scipy import sparse
import uncurl
from uncurl.sparse_utils import symmetric_kld

from . import gene_extraction, relabeling, sparse_matrix_h5, dense_matrix_h5
from .entropy import entropy
from .json_encoder import SimpleEncoder

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS

import simplex_sample

DIM_RED_OPTIONS = ['MDS', 'tSNE', 'TSVD', 'PCA', 'UMAP']

# TODO: build a new SCAnalysis object...
# somehow have UI hooks?

class SCAnalysisPlugin(object):
    """
    A plugin consists of one or more file-backed objects that are
    created by a single action.
    """

    def __init__(self, names, filenames=[], dependencies=[],
            loader_fns=None,
            saver_fns=None,
            creator_fn=None,
            options={},
            helper_functions=[]):
        """
        Args:
            names (list): list of strings, corresponding to the objects
                    provided by the plugin.
            filenames (list): list of strings, corresponding to filenames.
            dependencies (list): list of names, corresponding to
                    plugin object names that this plugin depends on.
            loader_fns (list): list of functions that take a path
                    object, and returns the object.
            saver_fn (function): function that takes an object and
                    a filename, and saves the object to file.
                    Returns None.
            creator_fn (function): given the dependencies and kwargs,
                    this returns a list
                    of objects that corresponds to each of the names.
            options (dict): map of strings to their default values
            helper_functions (list): list of functions that are
                    added to the SCObject, that depend on this plugin.
        """
        self.names = names
        self.name_indices = {n: i for i, n in enumerate(names)}
        self.filenames = filenames
        self.dependencies = dependencies
        self.loader_fns = loader_fns
        self.saver_fns = saver_fns
        self.creator_fn = creator_fn
        self.object_stores = [None for i in range(len(names))]
        self.options = options
        self.helper_functions = helper_functions

    def loader(self, name, sc_object):
        """
        Loads the object with the given name...
        """
        index = self.name_indices[name]
        filename = self.filenames[index]
        loader_fn = self.loader_fns[index]
        if isinstance(filename, list):
            for p in filename:
                path = os.path.join(sc_object.data_dir, p)
                try:
                    self.object_stores[index] = loader_fn(path, **sc_object.params)
                    return self.object_stores[index]
                except:
                    self.object_stores[index] = None
        else:
            path = os.path.join(sc_object.data_dir, filename)
            try:
                self.object_stores[index] = loader_fn(path, **sc_object.params)
            except:
                self.object_stores[index] = None
        return self.object_stores[index]

    def saver(self, name, sc_object):
        """
        Saves the temporary copy of the named object.
        """
        index = self.name_indices[name]
        filename = self.filenames[index]
        saver_fn = self.saver_fns[index]
        obj = self.object_stores[index]
        path = os.path.join(sc_object.data_dir, filename)
        try:
            saver_fn(obj, path, **sc_object.params)
        except:
            print('Error in saving file')

    def save_all(self, sc_object):
        """
        Saves every object created by this plugin.
        """
        for name, index in self.name_indices.items():
            filename = self.filenames[index]
            saver_fn = self.saver_fns[index]
            obj = self.object_stores[index]
            path = os.path.join(sc_object.data_dir, filename)
            try:
                saver_fn(obj, path, **sc_object.params)
            except:
                print('Error in saving file')

    def get(self, name, sc_object):
        """
        Returns the object with the given name. Returns None if it
        isn't stored in memory.
        """
        index = self.name_indices[name]
        if self.object_stores[index] is None:
            return None
        else:
            return self.object_stores[index]

    def creator(self, sc_object):
        """
        Creates the objects.
        """
        args = []
        for d in self.dependencies:
            try:
                args.append(getattr(sc_object, d))
            except:
                args.append(None)
        # TODO: maybe it would be better to do something different with params???
        # could we just select the params that we need for creator_fn???
        kwargs = sc_object.params
        objects = self.creator_fn(*args, **kwargs)
        #objects = self.creator_fn(sc_object)
        self.object_stores = list(objects)

    def accessor(self, name, sc_object):
        """
        returns a function that's used to access the object.
        """
        def accessor_fn():
            result = self.get(name, sc_object)
            if result is None:
                loader_result = self.loader(name, sc_object)
                if loader_result is None:
                    self.creator(sc_object)
                    self.save_all(sc_object)
            return self.get(name, sc_object)
        return accessor_fn

    def clear_temp(self):
        """
        Deletes all temporary references
        """
        del self.object_stores[:]
        self.object_stores = [None for i in range(len(self.names))]

    def clear_files(self, sc_object):
        """
        Deletes all files on disk
        """
        for filename in self.filenames:
            try:
                os.remove(os.path.join(sc_object.data_dir, filename))
            except:
                pass

    def add(self, sc_object):
        """
        Adds this plugin to the class...
        """
        for name in self.names:
            setattr(sc_object, name, self.accessor(name, sc_object))
        for option, val in self.options.items():
            if option not in sc_object.params:
                sc_object.params[option] = val
        for f in self.helper_functions:
            setattr(sc_object, f.__name__, f)

##### Saver/loader functions ########################################

def data_loader(path, is_sparse=True, **params):
    """
    Given a path, tries to load a mtx file, or a txt file of a matrix.
    """
    try:
        data = scipy.io.mmread(path)
        return sparse.csc_matrix(data)
    except:
        return np.loadtxt(path)


def data_saver(data, path, is_sparse=True, **params):
    """
    Given a path, tries to load a mtx file, or a txt file of a matrix.
    """
    try:
        scipy.io.mmwrite(data, path)
    except:
        np.savetxt(path, data)

def dense_loader(path, **params):
    return np.loadtxt(path)

def dense_saver(data, path, **params):
    np.savetxt(path, data)

def genes_loader(path, **params):
    return np.loadtxt(path, dtype=str)

def genes_saver(data, path, **params):
    np.savetxt(path, data)

def null_saver(data, path, **params):
    pass

def null_loader(path, **params):
    return None

def sparse_h5_saver(data, path, **params):
    sparse_matrix_h5.store_matrix(data, path)

def sparse_h5_loader(path, **params):
    return sparse_matrix_h5.load_matrix(path)

def dense_h5_saver(data, path, **params):
    dense_matrix_h5.store_array(path, data)

def dense_h5_loader(path, **params):
    return dense_matrix_h5.H5Array(path)

def dense_h5_dict_saver(data, path, **params):
    dense_matrix_h5.store_dict(path, data)

def dense_h5_dict_loader(path, **params):
    return dense_matrix_h5.H5Dict(path)

def np_saver(data, path, **params):
    np.save(path, data)

def np_loader(path, **params):
    return np.load(path)

def json_saver(data, path, **params):
    with open(path, 'w') as f:
        json.dump(data, f, cls=SimpleEncoder)

def json_loader(path, **params):
    with open(path) as f:
        return json.load(f)

##### creator functions #############################################

def none_creator(*args, **kwargs):
    return None

def default_gene_names(data, **kwargs):
    return [np.array(['gene_{0}'.format(i) for i in range(data.shape[0])])]

def run_qualNorm(data, init, **kwargs):
    if init is None:
        return None
    qn = uncurl.qualNorm(data, init)
    return qn

def run_data_normalized(data, **kwargs):
    if kwargs['normalize']:
        return uncurl.preprocessing.cell_normalize(data)
    else:
        return data

def gene_subset_creator(data_normalized, cell_subset, **kwargs):
    data = data_normalized[:, cell_subset]
    gene_subset = uncurl.max_variance_genes(data, nbins=5,
            frac=kwargs['frac'])
    gene_subset = np.array(gene_subset)
    return gene_subset

def cell_subset_creator(data, **kwargs):
    read_counts = np.array(data.sum(0)).flatten()
    cell_subset = (read_counts >= kwargs['min_reads']) & (read_counts <= kwargs['max_reads'])
    return cell_subset

def data_subset_creator(data_normalized, gene_subset, cell_subset, **kwargs):
    data_subset = data_normalized[np.ix_(gene_subset, cell_subset)]
    return data_subset

def cell_sample_creator(w, data_subset, **kwargs):
    # this does a simplex-based sampling...
    # TODO
    if kwargs['cell_frac'] == 1:
        return np.arange(w.shape[1])
    pass

def data_sampled_creator(data_subset, cell_sample, gene_subset_sampled, **kwargs):
    # TODO
    pass

def uncurl_kwargs_default(**kwargs):
    # TODO
    return dict(kwargs)

def run_uncurl(data_subset, uncurl_kwargs, **kwargs):
    if uncurl_kwargs is None:
        uncurl_kwargs = {}
    m, w, ll = uncurl.run_state_estimation(data_subset,
            clusters=kwargs['clusters'],
            **uncurl_kwargs)
    return m, w

def run_dim_red(m, w, cell_sample, **kwargs):
    # TODO: run dim red on result of uncurl, and also generate the means view.
    pass

def run_baseline_dim_red(data_sampled, **kwargs):
    # TODO
    pass

##### Helper functions ##############################################

def data_sampled_gene(sca, **kwargs):
    pass

##### plugins #######################################################


init_plugin = SCAnalysisPlugin(
        names=['init'],
        filenames=['init.txt'],
        dependencies=[],
        loader_fns=[dense_loader],
        saver_fns=[null_saver],
        creator_fn=none_creator,
        options={}
)

qualNorm_plugin = SCAnalysisPlugin(
        names=['qualNorm'],
        filenames=['qualnorm.txt'],
        dependencies=['data', 'init'],
        loader_fns=[dense_loader],
        saver_fns=[dense_saver],
        creator_fn=run_qualNorm,
        options={}
)

data_plugin = SCAnalysisPlugin(
        names=['data'],
        filenames=[['data.txt', 'data.txt.gz', 'data.mtx',
            'data.mtx.gz']],
        dependencies=[],
        loader_fns=[data_loader],
        saver_fns=[data_saver],
        creator_fn=none_creator,
        options={'is_sparse': True, 'data_filename': 'data.mtx'}
)

# do read-count normalization
data_normalized_plugin = SCAnalysisPlugin(
        names=['data_normalized'],
        filenames=[''],
        dependencies=['data'],
        loader_fns=[null_loader],
        saver_fns=[null_saver],
        creator_fn=run_data_normalized,
        options={'normalize': True}
)

# load gene names, or create default gene names
gene_names_plugin = SCAnalysisPlugin(
        names=['gene_names'],
        filenames=['gene_names.txt'],
        dependencies=[],
        loader_fns=[genes_loader],
        saver_fns=[genes_saver],
        creator_fn=default_gene_names,
        options={}
)

# gene subset based on uncurl.max_variance_genes
gene_subset_plugin = SCAnalysisPlugin(
        names=['gene_subset'],
        filenames=['gene_subset.txt'],
        dependencies=['data_normalized', 'gene_names'],
        loader_fns=[dense_loader],
        saver_fns=[dense_saver],
        creator_fn=gene_subset_creator,
        options={'frac': 0.2}
)

# cell_subset is based on min_read/max_read filtering
cell_subset_plugin = SCAnalysisPlugin(
        names=['cell_subset'],
        filenames=['cells_subset.txt'],
        dependencies=['data'],
        loader_fns=[dense_loader],
        saver_fns=[dense_saver],
        creator_fn=cell_subset_creator,
        options={'min_reads': 0, 'max_reads': 1e10}
)

# these are the cell samples that are derived from a simplex-based sampling
# on the result of uncurl.
cell_sample_plugin = SCAnalysisPlugin(
        names=['cell_sample', 'gene_subset_sampled'],
        filenames=['cell_sample.txt', 'gene_subset_sampled.txt'],
        dependencies=['w', 'data_subset'],
        loader_fns=[dense_loader, dense_loader],
        saver_fns=[dense_saver, dense_saver],
        creator_fn=cell_sample_creator,
        options={'cell_frac': 1.0}
)

# data_subset is a subset of the data that only contains a subset of the
# genes and cells, based on gene_subset and cell_subset.
data_subset_plugin = SCAnalysisPlugin(
        names=['data_subset'],
        filenames=[''],
        dependencies=['data_normalized', 'gene_subset', 'cell_subset'],
        loader_fns=[null_loader],
        saver_fns=[null_saver],
        creator_fn=data_subset_creator,
        options={}
)

# data_sampled contains the data sample that is used for visualization.
# data_sampled_all_genes is used for selecting specific genes for vis.
data_sampled_plugin = SCAnalysisPlugin(
        names=['data_sampled', 'data_sampled_all_genes'],
        filenames=['data_sampled.mtx', 'data_sampled_all_genes.h5'],
        dependencies=['data_subset', 'cell_sample', 'gene_subset_sampled'],
        loader_fns=[data_loader, sparse_h5_loader],
        saver_fns=[data_saver, sparse_h5_saver],
        creator_fn=data_sampled_creator,
        options={},
        helper_functions=[data_sampled_gene],
)

# uncurk_kwargs is a dict that contains all parameters for uncurl.
uncurl_kwargs_plugin = SCAnalysisPlugin(
        names=['uncurl_kwargs'],
        filenames=['uncurl_kwargs.json'],
        dependencies=[],
        loader_fns=[json_loader],
        saver_fns=[null_saver],
        creator_fn=uncurl_kwargs_default,
        options={}
)

# uncurl_plugin generates M and W
uncurl_plugin = SCAnalysisPlugin(
        names=['m', 'w'],
        filenames=['m.txt', 'w.txt'],
        dependencies=['data_subset', 'uncurl_kwargs'],
        loader_fns=[dense_loader, dense_loader],
        saver_fns=[dense_saver, dense_saver],
        creator_fn=run_uncurl,
        options={}
)

dim_red_plugin = SCAnalysisPlugin(
        names=['dim_red', 'mds_means'],
        filenames=['mds_data.txt', 'mds_means.txt'],
        dependencies=['m', 'w', 'cell_sample'],
        loader_fns=[dense_loader],
        saver_fns=[dense_saver],
        creator_fn=run_dim_red,
        options={'dim_red_option': 'tsne'},
)

baseline_dim_red_plugin = SCAnalysisPlugin(
        names=['baseline_dim_red'],
        filenames=['baseline_vis.txt'],
        dependencies=['data_sample'],
        loader_fns=[dense_loader],
        saver_fns=[dense_saver],
        creator_fn=run_baseline_dim_red,
        options={'dim_red_option': 'tsne'},
)

pairwise_diffexp_plugin = SCAnalysisPlugin(
        names=['t_scores', 't_pvals'],
        filenames=['t_scores.h5', 't_pvals.h5'],
        dependencies=['w'],
        loader_fns=[json_loader, json_loader],
        saver_fns=[json_saver, json_saver],
        creator_fn=run_pairwise_diffexp,
        options={},
)

top_genes_c_score_plugin = SCAnalysisPlugin(
        names=['top_genes', 'pvals'],
        filenames=['top_genes.txt', 'gene_pvals.txt'],
        dependencies=['w'],
        loader_fns=[json_loader, json_loader],
        saver_fns=[json_saver, json_saver],
        creator_fn=run_c_score,
        options={},
)

top_genes_1_vs_rest_plugin = SCAnalysisPlugin(
        names=['top_genes_1_vs_rest', 'pvals_1_vs_rest'],
        filenames=['top_genes_1_vs_rest.txt',
            'gene_pvals_1_vs_rest.txt'],
        dependencies=['w'],
        loader_fns=[np_loader, np_loader, json_loader, json_loader],
        saver_fns=[np_saver, np_saver, json_saver, json_saver],
        creator_fn=run_1_vs_rest,
        options={},
)


# TODO: data tracks
tracks_plugin = SCAnalysisPlugin(
        names=[''],
        filenames=[''],
        dependencies=[''],
        loader_fns=[null_loader],
        saver_fns=[null_saver],
        creator_fn=run_tracks,
        options={},
)

ALL_DEFAULT_PLUGINS = [
        init_plugin,
        qualNorm_plugin,
        data_plugin,
        data_normalized_plugin,
        gene_names_plugin,
        gene_subset_plugin,
        cell_subset_plugin,
        data_subset_plugin,
        data_sampled_plugin,
        uncurl_kwargs_plugin,
        uncurl_plugin,
        dim_red_plugin,
        baseline_dim_red_plugin,
        pairwise_diffexp_plugin,
        top_genes_c_score_plugin,
        top_genes_1_vs_rest_plugin,
        tracks_plugin,
        ]

##### classes, default objects #######################################

class SCAnalysis2(object):
    """
    This class represents an ongoing single-cell RNA-Seq analysis.
    """
    # TODO: re-design this class to have more of a plugin-like framework?

    def __init__(self, data_dir, plugins=[], **params):
        """
        Args:
            data_dir (str): directory where data is stored
        """
        self.json_f = os.path.join(data_dir, 'sc_analysis_2.json')
        self.data_dir = data_dir
        self.params = params
        self.plugins = plugins
        for plugin in self.plugins:
            plugin.add(self)

    def save_json_reset(self):
        """
        Clears
        """
        for plugin in self.plugins:
            plugin.clear_temp()
        # TODO: write out to json
        with open(self.json_f, 'w') as f:
            json.dump(self.params, f,
                    cls=SimpleEncoder)


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
                self.params = p
                if 'profiling' not in p:
                    self.profiling = {}
                return self
        if os.path.exists(os.path.join(self.data_dir, 'params.json')):
            with open(os.path.join(self.data_dir, 'params.json')) as f:
                params = json.load(f)
                if 'normalize_data' in params:
                    self.params['normalize'] = True
                try:
                    self.params['clusters'] = int(params['k'])
                    self.params['frac'] = float(params['genes_frac'])
                    self.params['cell_frac'] = float(params['cell_frac'])
                    self.params['min_reads'] = int(params['min_reads'])
                    self.params['max_reads'] = int(params['max_reads'])
                    self.params['baseline_dim_red'] = params['baseline_vismethod']
                    self.params['dim_red_option'] = params['vismethod']
                    for plugin in self.plugins:
                        for key, val in self.params.items():
                            if key in plugin:
                                plugin.options[key] = val
                except:
                    pass
        #if os.path.exists(os.path.join(self.data_dir, 'uncurl_kwargs.json')):
        #    with open(os.path.join(self.data_dir, 'uncurl_kwargs.json')) as f:
        #        uncurl_kwargs = json.load(f)
        #        self.uncurl_kwargs = uncurl_kwargs
        return self


def create_sc_analysis(data_dir, **params):
    """
    Creates a "default" SCAnalysis2 object.
    """
    # TODO
    # initialize all the plugins?

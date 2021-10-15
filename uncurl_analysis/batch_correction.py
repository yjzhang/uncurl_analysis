# TODO: use a library for batch correction
from scipy import sparse
import mnnpy


def batch_correct_mnn(data_list, frac=1.0):
    """
    Given a data list (list of scipy sparse matrices), this uses
    mnnpy to return a new matrix that is the input matrices merged and batch-corrected.

    The matrices should be of shape (genes, cells).
    """
    indices = list(range(data_list[0].shape[0]))
    top_genes = None
    if frac < 1.0:
        from uncurl import preprocessing
        data_merged = sparse.hstack(data_list)
        top_genes = preprocessing.max_variance_genes(data_merged, frac=frac)
    new_data_list = [x.T for x in data_list]
    result = mnnpy.mnn_correct(*new_data_list,
            var_index=indices, var_subset=top_genes)
    result_data = result[0]
    result_data = sparse.csc_matrix(result_data.T)
    return result_data

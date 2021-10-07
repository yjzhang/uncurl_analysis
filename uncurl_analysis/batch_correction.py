# TODO: use a library for batch correction
import mnnpy


def batch_correct_mnn(data_list, **kwargs):
    """
    Given a data list (list of scipy sparse matrices), this uses
    mnnpy to return a new data_list of the same matrices, but batch-corrected.
    """
    result = mnnpy.mnn_correct(data_list, **kwargs)
    return result

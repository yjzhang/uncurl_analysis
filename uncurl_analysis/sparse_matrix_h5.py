# Using an LIL matrix and pytables, this is a way to do fast on-disc
# gene lookup for a sparse matrix.

# TODO: i just found out that h5sparse exists so maybe we could use that instead.
import os

import numpy as np
from scipy import sparse
import tables

def store_matrix(lil_matrix, h5_filename):
    """
    Writes a sparse matrix to an h5 file.

    Args:
        lil_matrix: sparse matrix
        h5_filename: path where a h5 file can be written. If a file with that
                name exists, it will be deleted.
    """
    if os.path.exists(h5_filename):
        os.remove(h5_filename)
    lil_matrix = sparse.lil_matrix(lil_matrix)
    filters = tables.Filters(complevel=5, complib='zlib')
    matrix_file = tables.open_file(h5_filename, mode='w', filters=filters,
            title='matrix')
    data_table = matrix_file.create_vlarray(matrix_file.root,
                    'data', tables.Float64Atom(shape=()),
                    'data',
                    filters=tables.Filters(1))
    for row in lil_matrix.data:
        data_table.append(row)
    rows = matrix_file.create_vlarray(matrix_file.root,
                    'rows', tables.Int64Atom(shape=()),
                    "ragged array of ints",
                    filters=filters)
    for row in lil_matrix.rows:
        rows.append(row)
    matrix_file.create_array(matrix_file.root,
                    'shape', obj=np.array(lil_matrix.shape), title='Matrix shape')
    matrix_file.close()

def to_array(data, row, n_cols):
    array = np.zeros(n_cols)
    array[row] = data
    return array

def load_row(h5_filename, row_number):
    """
    Gets a single row from a stored matrix.

    Args:
        h5_filename (str): path to h5 file that contains array (from store_matrix)
        row_number (int)

    Returns:
        dense numpy array containing the row
    """
    f = tables.open_file(h5_filename, 'r')
    data_f = f.get_node('/data')
    data = data_f[row_number]
    rows_f = f.get_node('/rows')
    row = rows_f[row_number]
    shape = f.get_node('/shape').read()
    f.close()
    return to_array(data, row, shape[1])

def load_matrix(h5_filename):
    """Loads matrix stored in h5 file, returning a LIL matrix."""
    f = tables.open_file(h5_filename, 'r')
    data_f = f.get_node('/data')
    data = data_f.read()
    rows_f = f.get_node('/rows')
    row = rows_f.read()
    shape = f.get_node('/shape').read()
    f.close()
    mat = sparse.lil_matrix((shape[0], shape[1]))
    # this is for compatibility with scipy 1.5
    mat.rows = np.array([list(r) for r in row])
    mat.data = np.array([list(d) for d in data])
    return mat

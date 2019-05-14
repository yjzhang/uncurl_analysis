# store numpy arrays in tables
import os

import tables

class H5Array(object):
    """
    Read-only h5-backed array view, without loading the file directly.
    """

    def __init__(self, h5_filename):
        self.h5_filename = h5_filename

    def __getitem__(self, key):
        f = tables.open_file(self.h5_filename, 'r')
        data_f = f.get_node('/data')
        data = data_f.__getitem__(key)
        data_f.close()
        f.close()
        return data

    @property
    def shape(self):
        f = tables.open_file(self.h5_filename, 'r')
        data_f = f.get_node('/data')
        shape = data_f.shape
        data_f.close()
        f.close()
        return shape

    def __len__(self):
        return self.shape[0]

    def toarray(self):
        """
        Returns an actual np array representing the object.
        """
        f = tables.open_file(self.h5_filename, 'r')
        data_f = f.get_node('/data')
        data = data_f.read()
        data_f.close()
        f.close()
        return data


class H5Dict(object):
    """
    Read-only h5-backed dict view, without loading the file directly.
    Really naive... should be for few keys, with large arrays per key.
    All keys are treated as strings.

    All items are stored under root.
    """

    def __init__(self, h5_filename):
        self.h5_filename = h5_filename

    def __setitem__(self, key, val):
        key = '_' + str(key)
        f = tables.open_file(self.h5_filename, 'a')
        filters = tables.Filters(complevel=5, complib='zlib')
        data_table = f.create_carray('/',
                        key, obj=val,
                        filters=filters)
        data_table.close()
        f.close()

    def __getitem__(self, key, f=None):
        close_f = False
        if f is None:
            close_f = True
            f = tables.open_file(self.h5_filename, 'r')
        data_f = f.get_node('/_' + str(key))
        data = data_f.read()
        data_f.close()
        if close_f:
            f.close()
        return data

    def __len__(self):
        return len(self.keys())

    def keys(self):
        """Returns all keys"""
        f = tables.open_file(self.h5_filename, 'r')
        keys = [x.name[1:] for x in f.list_nodes('/')]
        f.close()
        return keys

    def items(self):
        """Returns an iterator over all key-item pairs"""
        keys = self.keys()
        f = tables.open_file(self.h5_filename, 'r')
        for key in keys:
            yield key, self.__getitem__(key, f)
        f.close()

def store_array(h5_filename, data):
    """
    Writes a numpy array to an h5 file

    Args:
        h5_filename: path where a h5 file can be written. If a file already
            exists with that name, it will be deleted.
        data: np array
    """
    if os.path.exists(h5_filename):
        os.remove(h5_filename)
    filters = tables.Filters(complevel=5, complib='zlib')
    matrix_file = tables.open_file(h5_filename, mode='w', filters=filters,
            title='matrix')
    data_table = matrix_file.create_carray('/',
                    'data', obj=data,
                    filters=filters)
    data_table.close()
    matrix_file.close()

def store_dict(h5_filename, data):
    """
    Writes a dict to an h5 file, where each key is a separate node indexed
    at root.
    """
    d = H5Dict(h5_filename)
    for key, value in data.items():
        d[key] = value

def load_array(h5_filename):
    """Loads matrix stored in h5 file, returning a numpy array."""
    f = tables.open_file(h5_filename, 'r')
    data_f = f.get_node('/data')
    data = data_f.read()
    f.close()
    return data

def load_array_view(h5_filename):
    """Loads a matrix as a H5Array object, a read-only view."""
    return H5Array(h5_filename)

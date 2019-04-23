# code to support custom cell selection criteria
import json

import numpy as np


class LabelCriterion(object):
    """
    This class represents a custom criterion for selecting a subset of cells.
    """

    def __init__(self, selection_type=None, comparison=None, target=None, value=None, and_or='and'):
        """
        Args:
            selection_type (str): 'cluster', 'gene', 'read_counts', or some custom label
            comparison (str): '=', '!=', '>=', '<='
            target (str or number): a representation of the target...
        """
        self.selection_type = selection_type
        self.comparison = comparison
        self.target = target
        self.value = value
        self.and_or = and_or

    def select(self, sca):
        """
        Returns a list of indices corresponding to the cells selected by the criterion. All these indices are in reference to cell_sample.

        Args:
            sca (SCAnalysis object)
        """
        if self.selection_type == 'cluster':
            labels = sca.labels
            if self.comparison == '=':
                return np.where(labels == int(self.target))[0]
            elif self.comparison == '!=':
                return np.where(labels != int(self.target))[0]
        elif self.selection_type == 'gene':
            gene_values = sca.data_sampled_gene(self.value)
            if self.comparison == '>=':
                return np.where(gene_values >= float(self.target))[0]
            elif self.comparison == '<=':
                return np.where(gene_values <= float(self.target))[0]
        elif self.selection_type == 'read_counts':
            pass
        else:
            color_track, is_discrete = sca.get_color_track(self.selection_type)
            return np.where(color_track == str(self.target))[0]

class CustomLabel(object):
    """
    this class represents a single label
    """

    def __init__(self, name, criteria=None):
        """
        Args:
            name (str)
            criteria: list of LabelCriterion objects
        """
        self.name = name
        if criteria is None:
            self.criteria = []
        else:
            self.criteria = criteria

    def select_cells(self, sca):
        """
        Selects cells corresponding to the given label
        """
        all_indices = set()
        for c in self.criteria:
            indices = c.select(sca)
            m = c.and_or
            if m == 'and' and len(all_indices) > 0:
                all_indices.intersection_update(indices)
            else:
                all_indices.update(indices)
        return np.array(list(all_indices))


class CustomColorMap(object):
    """
    This class represents a custom color map
    """

    def __init__(self, name, labels=None):
        """
        Args:
            name (str)
            labels (list of CustomLabel objects)
        """
        self.name = name
        if labels is None:
            self.labels = []
        else:
            self.labels = labels

    def label_cells(self, sca):
        """
        Labels the cells in the given SCAnalysis object.
        """
        cell_labels = np.array(['default' for x in range(len(sca.cell_sample))])
        for label in self.labels:
            indices = label.select_cells(sca)
            if len(indices) == 0:
                continue
            print(indices)
            cell_labels[indices] = label.name
        return cell_labels


def load_json(json_filename):
    """
    Loads a dict of name:CustomColorMap objects from json
    """
    with open(json_filename) as f:
        data = json.load(f)
    output = {}
    for key, val in data.items():
        cm = CustomColorMap('')
        cm.__dict__ = val
        labels_list = []
        for labels in cm.labels:
            new_label = CustomLabel('')
            new_label.__dict__ = labels
            criteria = []
            for criterion in new_label.criteria:
                nc = LabelCriterion()
                nc.__dict__ = criterion
                criteria.append(nc)
            new_label.criteria = criteria
            labels_list.append(new_label)
        cm.labels = labels_list
        output[key] = cm
    return output

def save_json(json_filename, data):
    """
    Saves a dict of name:CustomColorMap objects to the given json filename.
    """
    def json_default(x):
        return x.__dict__
    with open(json_filename, 'w') as f:
        json.dump(data, f, default=json_default)

def create_json(data):
    """
    Creates a str representation as a json
    """
    def json_default(x):
        return x.__dict__
    return json.dumps(data, default=json_default)

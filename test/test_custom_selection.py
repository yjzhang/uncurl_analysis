import os
import shutil
import unittest

from uncurl_analysis import custom_cell_selection, sc_analysis

from scipy import sparse
import scipy.io
from scipy.io import loadmat

class CustomSelectionTest(unittest.TestCase):

    def setUp(self):
        dat = loadmat('data/10x_pooled_400.mat')
        self.data = sparse.csc_matrix(dat['data'])
        self.labels = dat['labels'].flatten()
        self.data_dir = '/tmp/uncurl_analysis/test'
        try:
            shutil.rmtree(self.data_dir)
            os.makedirs(self.data_dir)
        except:
            os.makedirs(self.data_dir)
        scipy.io.mmwrite(os.path.join(self.data_dir, 'data.mtx'), self.data)
        self.sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx')
        self.sca.add_color_track('true_labels', self.labels, is_discrete=True)


    def test_label_criteria(self):
        criterion_1 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='0')
        results = criterion_1.select(self.sca)
        self.assertTrue(len(results)==50)
        self.assertTrue((self.labels[results] == 0).all())

        criterion_1 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='!=', target='0')
        results = criterion_1.select(self.sca)
        self.assertTrue(len(results)==350)
        self.assertTrue((self.labels[results] != 0).all())

        criterion_1 = custom_cell_selection.LabelCriterion(selection_type='cluster', comparison='=', target='0')
        results = criterion_1.select(self.sca)
        self.assertTrue((self.sca.labels[results] == 0).all())

    def test_custom_label(self):
        c1 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='0', and_or='or')
        c2 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='1', and_or='or')
        c3 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='2', and_or='or')
        label1 = custom_cell_selection.CustomLabel('label1', criteria=[c1, c2, c3])
        results = label1.select_cells(self.sca)
        self.assertTrue(len(results)==150)
        self.assertTrue(((self.labels[results] == 0) | (self.labels[results] == 1) | (self.labels[results] == 2)).all())


        c4 = custom_cell_selection.LabelCriterion(selection_type='cluster', comparison='=', target='0', and_or='and')
        c5 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='4', and_or='or')
        c6 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='6', and_or='or')
        c7 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='7', and_or='or')
        label1 = custom_cell_selection.CustomLabel('label1', criteria=[c1, c2, c3, c4, c5, c6, c7])
        results = label1.select_cells(self.sca)
        if len(results) > 0:
            self.assertTrue(((self.labels[results] == 0) | (self.labels[results] == 1) | (self.labels[results] == 2)\
                    | (self.labels[results]==4) | (self.labels[results]==6) | (self.labels[results]==7)).all())
            self.assertTrue((self.sca.labels[results] == 0).all())

        # test colormaps
        label1 = custom_cell_selection.CustomLabel('label1', criteria=[c1, c2, c3])
        label2 = custom_cell_selection.CustomLabel('label2', criteria=[c5, c6, c7])
        cmap1 = custom_cell_selection.CustomColorMap('cmap1', [label1, label2])
        labels = cmap1.label_cells(self.sca)
        self.assertTrue((labels=='label1').sum() == 150)
        self.assertTrue((labels=='label2').sum() == 150)
        self.assertTrue(((self.labels[labels=='label1'] == 0) | (self.labels[labels=='label1'] == 1) | (self.labels[labels=='label1'] == 2)).all())
        self.assertTrue(((self.labels[labels=='label2'] == 4) | (self.labels[labels=='label2'] == 6) | (self.labels[labels=='label2'] == 7)).all())
        # test json saving/loading
        custom_cell_selection.save_json(os.path.join(self.data_dir, 'cmap.json'), {'cmap1': cmap1})
        cmap1 = custom_cell_selection.load_json(os.path.join(self.data_dir, 'cmap.json'))['cmap1']
        print(custom_cell_selection.create_json(cmap1))
        labels = cmap1.label_cells(self.sca)
        self.assertTrue((labels=='label1').sum() == 150)
        self.assertTrue((labels=='label2').sum() == 150)
        self.assertTrue(((self.labels[labels=='label1'] == 0) | (self.labels[labels=='label1'] == 1) | (self.labels[labels=='label1'] == 2)).all())
        self.assertTrue(((self.labels[labels=='label2'] == 4) | (self.labels[labels=='label2'] == 6) | (self.labels[labels=='label2'] == 7)).all())

        # test adding colormaps to sca
        self.sca.create_custom_selection('cmap1', cmap1.labels)
        self.sca = sc_analysis.SCAnalysis(self.data_dir,
                clusters=8,
                data_filename='data.mtx')
        scores, pvals = self.sca.calculate_diffexp('cmap1', mode='pairwise')
        print(scores.shape)
        self.assertTrue(scores.shape == (3, 3, self.sca.data.shape[0]))
        print(pvals.shape)
        # update color map criteria
        c8 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='8', and_or='or')
        c9 = custom_cell_selection.LabelCriterion(selection_type='true_labels', comparison='=', target='9', and_or='or')
        self.sca.update_custom_color_track_label('cmap1', 'label2', [c5, c8, c9])
        data, is_discrete = self.sca.get_color_track('cmap1')
        self.assertTrue((data=='label1').sum()==150)
        self.assertTrue((data=='label2').sum()==150)
        self.assertTrue(((self.labels[data=='label2'] == 4) | (self.labels[data=='label2'] == 8) | (self.labels[data=='label2'] == 9)).all())


if __name__ == '__main__':
    unittest.main()


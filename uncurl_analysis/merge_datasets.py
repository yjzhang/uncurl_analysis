# TODO: functions to merge data matrices
import os
import numpy as np
from scipy import sparse
import scipy.io

def merge_files(data_paths, gene_paths, dataset_names, output_path,
        keep_genes=True, use_batch_correction=False):
    """
    Merges multiple files into a single file...

    Args:
        data_paths (list of paths to data files - mtx or txt files, in genes x cells order)
        gene_paths (list of paths to gene txt files)
        dataset_names (names of each dataset)
        output_path (directory to write to)
        keep_genes (bool): whether or not to include genes that only occur in some data files (values will be set to zero).
        use_batch_correction (bool): whether or not to use batch effect correction (with the MNN method)

    This saves an output file as 'data.mtx.gz' in output_path, and a genes file as 'gene_names.txt',
    and returns two files: a sparse

    It also deletes all the temporary files.
    """
    if len(data_paths) == 1 and len(gene_paths) == 1:
        dirname, data_filename = os.path.split(data_paths[0])
        data_filename = data_filename.replace('_1', '')
        data_output_path = os.path.join(dirname, data_filename)
        os.rename(data_paths[0], data_output_path)
        if gene_paths[0] is not None:
            dirname, gene_filename = os.path.split(gene_paths[0])
            gene_filename = gene_filename.replace('_1', '')
            gene_output_path = os.path.join(dirname, gene_filename)
            os.rename(gene_paths[0], gene_output_path)
        return data_output_path, gene_output_path
    all_data = []
    all_genes = []
    genes_set = set()
    data_array = []
    for data_path, gene_path, dataset_name in zip(data_paths, gene_paths, dataset_names):
        # TODO: if gene_path is None... what do we do?
        if gene_path is not None:
            genes = np.loadtxt(gene_path, dtype=str)
        if data_path.endswith('mtx.gz') or data_path.endswith('mtx'):
            data = scipy.io.mmread(data_path)
        else:
            data = np.loadtxt(data_path)
        # TODO: infer gene shape/data
        n_genes = genes.shape[0]
        if n_genes == data.shape[1]:
            data = data.T
        data_array += [dataset_name]*data.shape[1]
        all_genes.append(genes)
        if keep_genes or len(genes_set) == 0:
            genes_set.update(genes)
        else:
            genes_set = genes_set.intersection(genes)
        all_data.append(data)
    np.savetxt(os.path.join(output_path, 'samples.txt'), data_array, fmt='%s')

    # combine gene lists
    # decide whether any of the genes are different
    all_genes_same = True
    for g1 in all_genes:
        for g2 in all_genes:
            if len(g1) != len(g2) or (g1!=g2).any():
                all_genes_same = False
                break
    data_output_path = os.path.join(output_path, 'data.mtx')
    genes_output_path = os.path.join(output_path, 'gene_names.txt')
    if all_genes_same:
        combined_genes = all_genes[0]
        # combine matrices...
        # TODO: make sure that transposing the matrices is done before...
        # we assume that all input matrices are of shape (gene, cell)
        if use_batch_correction:
            from .batch_correction import batch_correct_mnn
            # use batch effect correction
            output_matrix = batch_correct_mnn(all_data)
        else:
            output_matrix = sparse.hstack(all_data)
        # save output matrix as mtx.gz
        scipy.io.mmwrite(data_output_path, output_matrix)
        import subprocess
        subprocess.call(['gzip', data_output_path])
        data_output_path += '.gz'
        # save gene path
        os.rename(gene_paths[0], genes_output_path)
    else:
        # TODO: what if multiple genes have the same name?
        # then we just take the max value
        combined_genes = np.array(list(genes_set))
        modified_matrices = []
        # do the mapping...
        # TODO: add an option to remove genes that only occur in one file
        for genes, data in zip(all_genes, all_data):
            data = sparse.csr_matrix(data)
            # use bmat - concatenate by rows
            gene_to_index = {g:i for i, g in enumerate(genes)}
            #for i, g in enumerate(genes):
            #    if g in gene_to_index:
            #        gene_to_index[g].append(i)
            #    else:
            #        gene_to_index[g] = [i]
            sub_blocks = []
            for gene in combined_genes:
                if gene not in gene_to_index:
                    if keep_genes:
                        sub_blocks.append([sparse.csr_matrix(np.zeros(data.shape[1]))])
                    else:
                        continue
                else:
                    sub_blocks.append([data[gene_to_index[gene], :]])
                    #sub_blocks.append([data[gene_to_index[gene], :].max(0)])
            data_new = sparse.bmat(sub_blocks)
            modified_matrices.append(data_new)
        if use_batch_correction:
            from .batch_correction import batch_correct_mnn
            output_matrix = batch_correct_mnn(modified_matrices)
        else:
            output_matrix = sparse.hstack(modified_matrices)
        scipy.io.mmwrite(data_output_path, output_matrix)
        import subprocess
        subprocess.call(['gzip', data_output_path])
        data_output_path += '.gz'
        # save gene path
        np.savetxt(genes_output_path, combined_genes, fmt='%s')
    # remove uploaded files
    for path in gene_paths:
        try:
            os.remove(path)
        except:
            pass
    for path in data_paths:
        try:
            os.remove(path)
        except:
            pass
    return data_output_path, genes_output_path



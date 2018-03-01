# correlations with bulk data

# method for correlating cell types (that's not rank correlation)?

# each bulk dataset defines a categorical distribution over genes (bulk datasets are normalized so that they sum to 1)

# so P(cell | bulk) = P_multinomial(cell | bulk_genes, cell_read_count)

# or... each bulk dataset defines a Poisson distribution?

# P(cell_i | bulk) = P_poisson(cell_i | \lambda=bulk_i*cell_read_count)
# and assume that each gene is independent...
# so P(cell | bulk) = \prod_i P_poisson(cell_i | bulk_i*cell_read_count)

# what about comparing a cluster (of single cells) mean to a bulk mean? can we use the same Poisson thing?

# TODO: test classification accuracy of this method vs Euclidean distance, correlation, etc.

# now, what if we have a ton of bulk datasets, and we want to query them? can we do some kinda locality-sensitive hash for probability?

# and how do we do gene matching? we have to have a standard for gene names (an ensembl translator?)

# library of bulk datasets

# alternatively, we can convert cell to a distribution and calculate divergence (but what about presence of zeros? do we just take the nonzeros?)

from scipy.stats import pearsonr, spearmanr

def log_prob_poisson(bulk_data, cell):
    """
    Log-probability of cell given bulk, where
    P(cell | bulk) = \prod_i P_poisson(cell_i | bulk_i*cell_read_count)

    Assumes the same genes in both datasets.

    Args:
        bulk_data (array): 1d array
        cell (array): 1d array
    """
    cell_read_count = cell.sum()
    b = bulk_data*cell_read_count

def rank_correlation(bulk_dataset, cell):
    return spearmanr(bulk_dataset, cell)[0]

def pearson_correlation(bulk_dataset, cell):
    return pearsonr(bulk_dataset, cell)[0]

def cosine(bulk_dataset, cell):
    pass

def bulk_lookup(datasets, cell, method='poisson'):
    """
    Returns a list of (dataset, value) pairs sorted by descending value,
    where value indicates similarity between the cell and the dataset.

    Potential metrics:
        - corr/pearson
        - rank_corr/spearman
        - cosine (normalized cosine distance)
        - poisson (log-probability)
    """
    pass

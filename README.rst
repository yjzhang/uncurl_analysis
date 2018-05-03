Analysis tools for UNCURL
=========================

.. image:: https://travis-ci.org/yjzhang/uncurl_analysis.svg
    :target: https://travis-ci.org/yjzhang/uncurl_analysis

Installation
------------

1. Clone the repository.

2. Run ``pip install -r requirements.txt``

3. In the main directory, run ``pip install .``


Cluster-specific gene expression
--------------------------------

These methods help to identify the genes that are overexpressed in each cluster.

The ``gene_extraction.pairwise_t`` function returns two 3d arrays of shape (k, k, genes), 

The outputs, ``c_scores`` and ``c_pvals``, are dicts of cluster labels to tuples ``(gene_index, c_score_or_pval)``

Example:

.. code-block:: python

    from uncurl_analysis import gene_extraction

    scores, pvals = gene_extraction.pairwise_t(data, labels)
    c_scores, c_pvals = gene_extraction.c_scores_from_t(scores, pvals)


Identifying similar bulk datasets
---------------------------------

Given some known datasets and a single-cell query dataset, we can get a similarity score between the query dataset and each of the known datasets.

This assumes that all genes are already aligned between the two datasets.

Example:

.. code-block:: python

    from uncurl_analysis import bulk_data

    # bulk_means is a dict of label : array. cell is a 1d array.
    scores = bulk_data.bulk_lookup(bulk_means, cell, method='poisson')

SCAnalysis: a framework for managing scRNA-seqdata analysis
-----------------------------------------------------------


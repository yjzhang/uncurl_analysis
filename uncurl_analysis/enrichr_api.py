# API to connect to ENRICHR

# TODO: use library?

import json
import requests

ENRICHR_URL = 'http://amp.pharm.mssm.edu/Enrichr/addList'

def enrichr_add_list(gene_list, description=''):
    """
    Args:
        gene_list (list): a list of gene names
    """
    payload = {
            'list': (None, '\n'.join(gene_list)),
            'description': (None, description)
            }

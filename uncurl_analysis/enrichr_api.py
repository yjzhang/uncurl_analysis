# API to connect to ENRICHR

# TODO: use library?

import json
import requests

ENRICHR_URL = 'http://amp.pharm.mssm.edu/Enrichr/'

# somewhat arbitrary list of enrichr libraries, that may be helpful
# for identifying cell types.
ENRICHR_LIBRARIES = [
        'Human_Gene_Atlas',
        'Mouse_Gene_Atlas',
        'Allen_Brain_Atlas_up',
#        'GTEx_Tissue_Sample_Gene_Expression_Profiles_up',
        'ARCHS4_Tissues',
        'ARCHS4_Cell-lines',
        'GO_Biological_Process_2017b',
        'GO_Cellular_Component_2017b',
        'GO_Molecular_Function_2017b',
        'Jensen_COMPARTMENTS',
        'Jensen_DISEASES',
        'Jensen_TISSUES',
        ]

def enrichr_add_list(gene_list, description=''):
    """
    Args:
        gene_list (list): a list of gene names

    Returns:
        user_list_id
    """
    query_param = 'addList'
    payload = {
            'list': (None, '\n'.join(gene_list)),
            'description': (None, description)
            }
    try:
        response = requests.post(ENRICHR_URL + query_param, files=payload,
            timeout=5)
    except requests.exceptions.Timeout:
        return 'timeout'
    if not response.ok:
        raise Exception('Error analyzing gene list')
    data = json.loads(response.text)
    user_list_id = data['userListId']
    #short_id = data['shortId']
    #print(data)
    return user_list_id

def enrichr_query(user_list_id,
        gene_set_library='Human_Gene_Atlas'):
    """
    Args:
        user_list_id (str): output of enrichr_add_list
        gene_set_library (str): something in ENRICHR_LIBRARIES, or another Enrichr library - see http://amp.pharm.mssm.edu/Enrichr/#stats

    Returns:
        list of top terms identified by enrichr, where each term is represented by a list containing: Rank, Term name, P-value, Z-score, Combined score, Overlapping genes, Adjusted p-value
    """
    query_param = 'enrich'
    query_string = '?userListId={0}&backgroundType={1}'
    try:
        response =  requests.get(ENRICHR_URL
                + query_param
                + query_string.format(user_list_id, gene_set_library),
                timeout=5)
    except requests.exceptions.Timeout:
        return 'timeout'
    if not response.ok:
        raise Exception('Error fetching enrichment results')
    data = json.loads(response.text)
    #print(data)
    return data[gene_set_library]

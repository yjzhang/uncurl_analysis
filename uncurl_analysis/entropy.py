import numpy as np
from scipy.special import xlogy

def entropy(w, eps=1e-10):
    """
    Returns entropy of W
    """
    return -(w*np.log2(w+eps)).sum(0)

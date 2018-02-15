import numpy as np

def entropy(w):
    """
    Returns entropy of W
    """
    return (w*np.log2(w)).sum(0)


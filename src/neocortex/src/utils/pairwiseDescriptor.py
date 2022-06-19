import numpy as np
from scipy import sparse

# Pairwise (raw descriptors)
def pairwiseDescriptors(D1,D2):
    # Pairwise comparison
    if sparse.issparse(D1):
        S = D1.dot(D2.transpose())
        D1 = D1.toarray()
        D2 = D2.toarray()
    else:
        S = D1.dot(np.transpose(D2))

    nOnes_D1 = np.sum(D1, axis=1)
    nOnes_D2 = np.sum(D2, axis=1)
    D1t = np.transpose(np.vstack((np.ones(len(nOnes_D1)),nOnes_D1)))
    D2t = np.vstack((nOnes_D2,np.ones(len(nOnes_D2))))
    mean_nOnes = D1t.dot(D2t)/2
    S = S / mean_nOnes
    return S
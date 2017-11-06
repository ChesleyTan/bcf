import numpy as np
from scipy.spatial.distance import cdist

def llc_coding_approx(B, X, k_nn=5, beta=1e-4):
    D = cdist(X, B, 'euclidean')
    N = X.shape[0]
    I = np.zeros((N, k_nn), 'int32')
    for i in range(N):
        d = D[i, :]
        idx = np.argsort(d)
        I[i, :] = idx[:k_nn]
    II = np.eye(k_nn)
    coeffs = np.zeros((N, B.shape[0]))
    for i in range(N):
        idx = I[i, :]
        z = B[idx, :] - np.tile(X[i, :], (k_nn, 1)) # shift ith point to origin
        z = z.dot(z.transpose())
        z = z + II * beta * np.trace(z)             # regularization (K>D)
        w = np.linalg.solve(z, np.ones((k_nn, 1)))
        w = w / np.sum(w)                           # enforce sum(w) = 1
        coeffs[i, idx] = w.ravel()
    return coeffs

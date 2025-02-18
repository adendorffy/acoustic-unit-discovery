import numpy as np
import scipy.spatial.distance as distance
import numba

def segment(sequence, codebook, gamma):
    dists = distance.cdist(sequence, codebook).astype(np.float32)
    alpha, P = _segment(dists, gamma)
    return _backtrack(alpha, P)

@numba.njit()
def _segment(dists, gamma):
    T, K = dists.shape

    alpha = np.zeros(T + 1, dtype=np.float32)
    P = np.zeros((T + 1, 2), dtype=np.int32)
    D = np.zeros((T, T, K), dtype=np.float32)

    for t in range(T):
        for k in range(K):
            D[t, t, k] = dists[t, k]
    for t in range(T):
        for s in range(t + 1, T):
            D[t, s, :] = D[t, s - 1, :] + dists[s, :] - gamma

    for t in range(T):
        alpha[t + 1] = np.inf
        for s in range(t + 1):
            k = np.argmin(D[s, t, :])
            alpha_min = alpha[s] + D[s, t, k]
            if alpha_min < alpha[t + 1]:
                P[t + 1, :] = s, k
                alpha[t + 1] = alpha_min
    return alpha, P


@numba.njit()
def _backtrack(alpha, P):
    rhs = len(alpha) - 1
    segments = []
    boundaries = [rhs]
    while rhs != 0:
        lhs, code = P[rhs, :]
        segments.append(code)
        boundaries.append(lhs)
        rhs = lhs
    segments.reverse()
    boundaries.reverse()
    return np.array(segments), np.array(boundaries)
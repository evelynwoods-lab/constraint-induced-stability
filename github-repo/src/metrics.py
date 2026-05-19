"""
metrics.py — Representation similarity metrics.

Implements:
  - Linear CKA (Kornblith et al., 2019)
  - RSA — Representational Similarity Analysis (Kriegeskorte et al., 2008)
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist


# ── Linear CKA ────────────────────────────────────────────────────────────────

def _hsic(A: np.ndarray, B: np.ndarray) -> float:
    """
    Unbiased HSIC estimator (Gretton et al., 2005).
    A, B: (n, n) kernel matrices (centered).
    """
    n = A.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(A @ H @ B @ H)) / (n - 1) ** 2


def linear_cka(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """
    Linear Centred Kernel Alignment.

    Args:
        Z1: (n, d1) representation matrix
        Z2: (n, d2) representation matrix

    Returns:
        CKA score in [0, 1]. Value 1 = perfect linear alignment.

    Reference:
        Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
        Similarity of Neural Network Representations Revisited. ICML.
    """
    Z1c = Z1 - Z1.mean(axis=0, keepdims=True)
    Z2c = Z2 - Z2.mean(axis=0, keepdims=True)

    K1 = Z1c @ Z1c.T   # (n, n)
    K2 = Z2c @ Z2c.T   # (n, n)

    num = _hsic(K1, K2)
    denom = np.sqrt(_hsic(K1, K1) * _hsic(K2, K2)) + 1e-10
    return num / denom


# ── RSA — Representational Similarity Analysis ────────────────────────────────

def rsa(
    Z1: np.ndarray,
    Z2: np.ndarray,
    n_samples: int = 200,
    metric: str = "cosine",
    seed: int = 42,
) -> float:
    """
    Representational Similarity Analysis.
    Computes the Pearson correlation between the Representational
    Dissimilarity Matrices (RDMs) of Z1 and Z2.

    Args:
        Z1: (n, d1) representation matrix
        Z2: (n, d2) representation matrix
        n_samples: number of samples for RDM (subsampled for speed)
        metric: distance metric for RDM ('cosine', 'euclidean', etc.)
        seed: random seed for subsampling

    Returns:
        RSA score (Pearson r) in [-1, 1]. Value 1 = identical structure.

    Reference:
        Kriegeskorte, N., Mur, M., & Bandettini, P. (2008).
        Representational similarity analysis. Frontiers in Systems Neuroscience.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(Z1))
    idx = rng.choice(len(Z1), n, replace=False)

    R1 = cdist(Z1[idx], Z1[idx], metric=metric)
    R2 = cdist(Z2[idx], Z2[idx], metric=metric)

    # Upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    r1_vec = R1[triu_idx]
    r2_vec = R2[triu_idx]

    r, _ = pearsonr(r1_vec, r2_vec)
    return float(r)


# ── Latent drift ──────────────────────────────────────────────────────────────

def latent_drift(Z: np.ndarray) -> float:
    """
    Average frame-to-frame distance in the latent sequence.
    Corresponds to Stab_f in the paper (Lemma 1).

    Args:
        Z: (T, d) latent sequence (in temporal order)
    Returns:
        mean ||z_t - z_{t+1}||
    """
    diffs = np.linalg.norm(Z[1:] - Z[:-1], axis=1)
    return float(diffs.mean())

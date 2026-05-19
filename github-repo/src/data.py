"""
data.py — Lorenz-63 data generation and windowing.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def lorenz63(
    n_steps: int = 30000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    warmup: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a Lorenz-63 trajectory.

    Returns:
        trajectory: np.ndarray of shape (n_steps, 3)
    """
    np.random.seed(seed)
    total = n_steps + warmup
    x = np.zeros((total, 3))
    x[0] = [0.1, 0.0, 0.0]

    for t in range(total - 1):
        dx = sigma * (x[t, 1] - x[t, 0])
        dy = x[t, 0] * (rho - x[t, 2]) - x[t, 1]
        dz = x[t, 0] * x[t, 1] - beta * x[t, 2]
        x[t + 1] = x[t] + dt * np.array([dx, dy, dz])

    return x[warmup:]  # discard warmup


def make_windows(
    trajectory: np.ndarray,
    k: int = 16,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert trajectory to overlapping windows of length k.

    Args:
        trajectory: (T, 3) array
        k: window length (context depth, controls encoder memory)
        normalize: whether to normalise per channel

    Returns:
        windows: (N, k, 3) array where N = T - k
    """
    N = len(trajectory) - k
    windows = np.stack([trajectory[i : i + k] for i in range(N)])

    if normalize:
        mu = windows.mean(axis=(0, 1), keepdims=True)
        std = windows.std(axis=(0, 1), keepdims=True) + 1e-8
        windows = (windows - mu) / std

    return windows


def make_lorenz63_dataset(
    n: int = 30000,
    k: int = 16,
    train_frac: float = 0.8,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple:
    """
    Full pipeline: generate Lorenz-63, window, split, return DataLoaders.

    Args:
        n: number of trajectory steps
        k: context window length
        train_frac: fraction for training
        batch_size: batch size for DataLoaders
        seed: random seed

    Returns:
        (train_loader, test_loader): tuple of DataLoaders
    """
    traj = lorenz63(n_steps=n, seed=seed)
    windows = make_windows(traj, k=k, normalize=True)

    N = len(windows)
    n_train = int(N * train_frac)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    X_train = torch.tensor(windows[idx[:n_train]], dtype=torch.float32)
    X_test = torch.tensor(windows[idx[n_train:]], dtype=torch.float32)

    train_ds = TensorDataset(X_train, X_train)
    test_ds = TensorDataset(X_test, X_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_test_tensor(n: int = 30000, k: int = 16, n_test: int = 1000) -> torch.Tensor:
    """Return a fixed test tensor for evaluation."""
    traj = lorenz63(n_steps=n)
    windows = make_windows(traj, k=k)
    return torch.tensor(windows[-n_test:], dtype=torch.float32)

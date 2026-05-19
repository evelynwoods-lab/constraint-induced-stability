"""
evaluate.py — Evaluation: CKA, RSA, latent drift across conditions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics import linear_cka, rsa, latent_drift
from src.train import get_latents


def evaluate_alignment(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    test_loader: DataLoader,
    n_rsa_samples: int = 200,
    device: str = "cpu",
) -> dict:
    """
    Compute CKA and RSA between two encoder models.

    Args:
        model1, model2:  trained encoder models
        test_loader:     DataLoader for test set
        n_rsa_samples:   number of samples for RSA RDM

    Returns:
        dict with keys 'cka', 'rsa', 'drift_1', 'drift_2'
    """
    Z1 = get_latents(model1, test_loader, device=device).numpy()
    Z2 = get_latents(model2, test_loader, device=device).numpy()

    return {
        "cka":     linear_cka(Z1, Z2),
        "rsa":     rsa(Z1, Z2, n_samples=n_rsa_samples),
        "drift_1": latent_drift(Z1),
        "drift_2": latent_drift(Z2),
    }


def run_seed_sweep(
    model_cls1,
    model_cls2,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_seeds: int = 5,
    epochs: int = 20,
    lipschitz_target: float = 1.0,
    device: str = "cpu",
    **train_kwargs,
) -> dict:
    """
    Run alignment evaluation across multiple random seeds.

    Returns:
        dict with 'cka_mean', 'cka_std', 'rsa_mean', 'rsa_std'
    """
    from src.train import train_encoder

    cka_list, rsa_list = [], []

    for seed in range(n_seeds):
        torch.manual_seed(seed * 100)
        np.random.seed(seed * 100)

        m1 = model_cls1()
        m2 = model_cls2()

        train_encoder(m1, train_loader, epochs=epochs,
                      lipschitz_target=lipschitz_target,
                      verbose=False, device=device, **train_kwargs)
        train_encoder(m2, train_loader, epochs=epochs,
                      lipschitz_target=lipschitz_target,
                      verbose=False, device=device, **train_kwargs)

        metrics = evaluate_alignment(m1, m2, test_loader, device=device)
        cka_list.append(metrics["cka"])
        rsa_list.append(metrics["rsa"])

        print(f"  Seed {seed}: CKA={metrics['cka']:.4f}  RSA={metrics['rsa']:.4f}")

    return {
        "cka_mean": float(np.mean(cka_list)),
        "cka_std":  float(np.std(cka_list)),
        "rsa_mean": float(np.mean(rsa_list)),
        "rsa_std":  float(np.std(rsa_list)),
        "cka_all":  cka_list,
        "rsa_all":  rsa_list,
    }

"""
train.py — Training script with Lipschitz regularization.

The Lipschitz penalty enforces constraint (C2): d_Z(f(x), f(x')) ≤ ε · d_M(x, x').
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def lipschitz_penalty(
    model: nn.Module,
    x_batch: torch.Tensor,
    eps_target: float = 1.0,
    n_pairs: int = 32,
) -> torch.Tensor:
    """
    Empirical Lipschitz regularization.

    Penalises ||f(x) - f(x')|| / (ε · ||x - x'||) > 1
    This enforces: d_Z(f(x), f(x')) ≤ ε · d_M(x, x')

    Args:
        model:       encoder with .encode() method
        x_batch:     (batch, k, 3) input batch
        eps_target:  Lipschitz target ε
        n_pairs:     number of random pairs to evaluate

    Returns:
        penalty: scalar tensor (0 if all pairs satisfy constraint)
    """
    n = min(n_pairs, len(x_batch))
    i1 = torch.randperm(len(x_batch))[:n]
    i2 = torch.randperm(len(x_batch))[:n]

    x1, x2 = x_batch[i1], x_batch[i2]
    z1 = model.encode(x1)
    z2 = model.encode(x2)

    # ||x1 - x2|| in input space
    dx = (x1 - x2).reshape(n, -1).norm(dim=1) + 1e-8

    # ||z1 - z2|| / (ε · ||x1 - x2||)
    ratio = z1.sub(z2).norm(dim=1) / (eps_target * dx)

    # Penalise when ratio > 1
    return F.relu(ratio - 1.0).mean()


def train_encoder(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 30,
    lr: float = 3e-4,
    lipschitz_target: float = 1.0,
    lip_weight: float = 0.05,
    clip_grad: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
) -> list:
    """
    Train an autoencoder with Lipschitz regularization.

    Loss = MSE(reconstruction) + lip_weight * Lipschitz_penalty

    Args:
        model:            encoder model with .encode() and .forward()
        train_loader:     DataLoader returning (x, x) pairs
        epochs:           number of training epochs
        lr:               learning rate
        lipschitz_target: ε — target Lipschitz constant
        lip_weight:       weight of Lipschitz penalty
        clip_grad:        gradient clipping norm
        device:           'cpu' or 'cuda'
        verbose:          print progress

    Returns:
        loss_history: list of per-epoch losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)

            z, x_hat = model(x_batch)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_hat, x_batch)

            # Lipschitz penalty (enforces ε constraint)
            lip_loss = lipschitz_penalty(
                model, x_batch,
                eps_target=lipschitz_target,
                n_pairs=32,
            )

            loss = recon_loss + lip_weight * lip_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  Loss={avg_loss:.4f}")

    return loss_history


@torch.no_grad()
def get_latents(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Extract latent representations for an entire DataLoader.

    Returns:
        Z: (N, bottleneck_dim) tensor
    """
    model.eval()
    model = model.to(device)
    zs = []
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        zs.append(model.encode(x_batch).cpu())
    return torch.cat(zs, dim=0)

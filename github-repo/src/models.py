"""
models.py — CNN and Transformer encoder architectures.

Both architectures satisfy the (B, ε, k) constraint framework:
  - Bottleneck dimension controls B (entropy bound)
  - Lipschitz regularization controls ε (applied during training)
  - Context window k is fixed at data-generation time
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    1D Convolutional autoencoder.

    Input:  (batch, k, 3)   — k time steps, 3 Lorenz dimensions
    Latent: (batch, bottleneck_dim)
    Output: (batch, k, 3)   — reconstruction
    """

    def __init__(self, k: int = 16, in_dim: int = 3, bottleneck_dim: int = 8):
        super().__init__()
        self.k = k
        self.bottleneck_dim = bottleneck_dim

        # Encoder: (B, 3, k) -> (B, 64, k//4) -> flatten -> bottleneck
        self.conv_enc = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(2),
            nn.Flatten(),
        )

        # Compute flat size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, k)
            flat_size = self.conv_enc(dummy).shape[1]

        self.bottleneck = nn.Linear(flat_size, bottleneck_dim)

        # Decoder: bottleneck -> reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, flat_size),
            nn.GELU(),
            nn.Unflatten(1, (64, k // 4)),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, in_dim, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, k, in_dim)
        Returns:
            z: (batch, bottleneck_dim)
        """
        h = self.conv_enc(x.permute(0, 2, 1))   # (B, 3, k) -> (B, flat)
        return self.bottleneck(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, bottleneck_dim)
        Returns:
            x_hat: (batch, k, in_dim)
        """
        return self.decoder(z).permute(0, 2, 1)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


class TransformerEncoder(nn.Module):
    """
    Transformer-based autoencoder.

    Input:  (batch, k, 3)
    Latent: (batch, bottleneck_dim)
    Output: (batch, k, 3)
    """

    def __init__(
        self,
        k: int = 16,
        in_dim: int = 3,
        bottleneck_dim: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 64,
    ):
        super().__init__()
        self.k = k
        self.bottleneck_dim = bottleneck_dim

        # Input projection + positional embedding
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, k, d_model) * 0.01)

        # Transformer layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Bottleneck projection (pool all positions then project)
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model * k, 64),
            nn.GELU(),
            nn.Linear(64, bottleneck_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.GELU(),
            nn.Linear(64, k * in_dim),
        )
        self._k = k
        self._in_dim = in_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, k, in_dim)
        Returns:
            z: (batch, bottleneck_dim)
        """
        h = self.input_proj(x) + self.pos_embed          # (B, k, d_model)
        h = self.transformer(h)                           # (B, k, d_model)
        h = h.reshape(h.shape[0], -1)                     # (B, k*d_model)
        return self.bottleneck(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).reshape(-1, self._k, self._in_dim)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

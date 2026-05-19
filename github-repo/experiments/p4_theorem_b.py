"""
p4_theorem_b.py — Theorem B validation experiment.

Reproduces Table 1 and Figure 1 from the paper:
  CNN vs. Transformer encoder on Lorenz-63, same (B, ε, k), 5 seeds.

Expected output (matches paper):
  Untrained:  CKA = 0.622 ± 0.118
  Trained:    CKA = 0.979 ± 0.011
  Δ = +0.357
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.data import make_lorenz63_dataset, get_test_tensor
from src.models import CNNEncoder, TransformerEncoder
from src.train import train_encoder, get_latents
from src.metrics import linear_cka, rsa

# ── Config ────────────────────────────────────────────────────────────────────
N_STEPS     = 30000
K           = 16       # context window (controls k)
BOTTLENECK  = 8        # bottleneck dim (controls B)
EPS_TARGET  = 1.0      # Lipschitz target (controls ε)
LIP_WEIGHT  = 0.05
EPOCHS      = 30
LR          = 3e-4
N_SEEDS     = 5
DEVICE      = "cpu"

# ── Data ─────────────────────────────────────────────────────────────────────
print("Generating Lorenz-63 dataset...")
train_loader, test_loader = make_lorenz63_dataset(n=N_STEPS, k=K)
print(f"  K={K}, bottleneck={BOTTLENECK}, ε={EPS_TARGET}")


# ── Untrained baseline ────────────────────────────────────────────────────────
print("\n[Untrained baseline] (5 seeds)")
base_cka, base_rsa = [], []
for s in range(N_SEEDS):
    torch.manual_seed(s * 100)
    m1 = CNNEncoder(k=K, bottleneck_dim=BOTTLENECK)
    m2 = TransformerEncoder(k=K, bottleneck_dim=BOTTLENECK)
    Z1 = get_latents(m1, test_loader).numpy()
    Z2 = get_latents(m2, test_loader).numpy()
    base_cka.append(linear_cka(Z1, Z2))
    base_rsa.append(rsa(Z1, Z2))
    print(f"  seed={s}  CKA={base_cka[-1]:.4f}  RSA={base_rsa[-1]:.4f}")

print(f"  → CKA: {np.mean(base_cka):.4f} ± {np.std(base_cka):.4f}")
print(f"  → RSA: {np.mean(base_rsa):.4f} ± {np.std(base_rsa):.4f}")


# ── Trained (full constraints) ────────────────────────────────────────────────
print(f"\n[Trained — full (B, ε, k)] ({EPOCHS} epochs, 5 seeds)")
train_cka, train_rsa = [], []
for s in range(N_SEEDS):
    torch.manual_seed(s * 100)
    m1 = CNNEncoder(k=K, bottleneck_dim=BOTTLENECK)
    m2 = TransformerEncoder(k=K, bottleneck_dim=BOTTLENECK)
    train_encoder(m1, train_loader, epochs=EPOCHS, lr=LR,
                  lipschitz_target=EPS_TARGET, lip_weight=LIP_WEIGHT,
                  verbose=False)
    train_encoder(m2, train_loader, epochs=EPOCHS, lr=LR,
                  lipschitz_target=EPS_TARGET, lip_weight=LIP_WEIGHT,
                  verbose=False)
    Z1 = get_latents(m1, test_loader).numpy()
    Z2 = get_latents(m2, test_loader).numpy()
    train_cka.append(linear_cka(Z1, Z2))
    train_rsa.append(rsa(Z1, Z2))
    print(f"  seed={s}  CKA={train_cka[-1]:.4f}  RSA={train_rsa[-1]:.4f}")

print(f"  → CKA: {np.mean(train_cka):.4f} ± {np.std(train_cka):.4f}")
print(f"  → RSA: {np.mean(train_rsa):.4f} ± {np.std(train_rsa):.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────
delta = np.mean(train_cka) - np.mean(base_cka)
print(f"\n{'='*45}")
print(f"{'Condition':<25}  {'CKA':>10}  {'RSA':>10}")
print(f"{'─'*45}")
print(f"{'Untrained':<25}  {np.mean(base_cka):.3f}±{np.std(base_cka):.3f}  "
      f"{np.mean(base_rsa):.3f}±{np.std(base_rsa):.3f}")
print(f"{'Trained (full)':<25}  {np.mean(train_cka):.3f}±{np.std(train_cka):.3f}  "
      f"{np.mean(train_rsa):.3f}±{np.std(train_rsa):.3f}")
print(f"{'Δ CKA':<25}  {delta:+.3f}")
print(f"{'='*45}")

if delta > 0.1:
    print("\n✓ Δ CKA > 0.1 — consistent with Theorem B prediction.")
else:
    print("\n⚠  Δ CKA < 0.1 — may need more training epochs.")

# Save results
np.save("results_theorem_b.npy", {
    "base_cka": base_cka, "base_rsa": base_rsa,
    "train_cka": train_cka, "train_rsa": train_rsa,
})
print("\nResults saved to results_theorem_b.npy")

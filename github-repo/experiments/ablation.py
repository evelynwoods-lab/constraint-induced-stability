"""
ablation.py — Constraint ablation study.

Tests: Full (B,ε,k) vs. No Lipschitz vs. No Bottleneck constraint.
Reproduces Table 1 (ablation) from the paper.

Expected output:
  Full (B,ε,k):         CKA=0.979  RSA=0.887
  No Lipschitz (ε→∞):   CKA=0.975  RSA=0.901
  No Bottleneck (B→∞):  CKA=0.962  RSA=0.782
"""

import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from src.data import make_lorenz63_dataset
from src.models import CNNEncoder, TransformerEncoder
from src.train import train_encoder, get_latents
from src.metrics import linear_cka, rsa

# ── Config ────────────────────────────────────────────────────────────────────
K        = 16
EPOCHS   = 20
N_SEEDS  = 2      # use 5 for full paper results; 2 for quick check

conditions = [
    {
        "name":            "Full (B, ε, k)",
        "bottleneck_dim":  8,
        "lip_weight":      0.05,
        "eps_target":      1.0,
    },
    {
        "name":            "No Lipschitz (ε→∞)",
        "bottleneck_dim":  8,
        "lip_weight":      0.0,    # ← disable Lipschitz penalty
        "eps_target":      1.0,
    },
    {
        "name":            "No Bottleneck (B→∞)",
        "bottleneck_dim":  48,     # ← wide bottleneck
        "lip_weight":      0.05,
        "eps_target":      1.0,
    },
]

train_loader, test_loader = make_lorenz63_dataset(n=20000, k=K)

print(f"{'Condition':<28}  {'CKA':>13}  {'RSA':>13}")
print("─" * 58)

all_results = {}
for cond in conditions:
    cka_list, rsa_list = [], []
    for s in range(N_SEEDS):
        torch.manual_seed(s * 77)
        np.random.seed(s * 77)

        bd = cond["bottleneck_dim"]
        m1 = CNNEncoder(k=K, bottleneck_dim=bd)
        m2 = TransformerEncoder(k=K, bottleneck_dim=bd)

        for m in [m1, m2]:
            train_encoder(
                m, train_loader,
                epochs=EPOCHS,
                lip_weight=cond["lip_weight"],
                lipschitz_target=cond["eps_target"],
                verbose=False,
            )

        Z1 = get_latents(m1, test_loader).numpy()
        Z2 = get_latents(m2, test_loader).numpy()
        cka_list.append(linear_cka(Z1, Z2))
        rsa_list.append(rsa(Z1, Z2))

    mc, sc = np.mean(cka_list), np.std(cka_list)
    mr, sr = np.mean(rsa_list), np.std(rsa_list)
    all_results[cond["name"]] = {"cka": (mc, sc), "rsa": (mr, sr)}
    print(f"  {cond['name']:<26}  {mc:.3f}±{sc:.3f}   {mr:.3f}±{sr:.3f}")

print("─" * 58)
print("\nKey finding:")
full  = all_results["Full (B, ε, k)"]["cka"][0]
nobd  = all_results["No Bottleneck (B→∞)"]["cka"][0]
norsa = all_results["No Bottleneck (B→∞)"]["rsa"][0]
frsa  = all_results["Full (B, ε, k)"]["rsa"][0]
print(f"  CKA drop (no B): {full:.3f} → {nobd:.3f}  Δ={nobd-full:+.3f}")
print(f"  RSA drop (no B): {frsa:.3f} → {norsa:.3f}  Δ={norsa-frsa:+.3f}")

if nobd < full:
    print("\n✓ Removing B reduces CKA — consistent with Lemma 2 / Prop 1.1(S2).")
else:
    print("\n⚠  Unexpected: removing B did not reduce CKA. Check bottleneck size.")

np.save("results_ablation.npy", all_results, allow_pickle=True)
print("\nResults saved to results_ablation.npy")

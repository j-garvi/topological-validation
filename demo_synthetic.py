#!/usr/bin/env python3
"""
Demo: Topological Validation on Synthetic Vortex Fields.

This self-contained script demonstrates the TDA validation pipeline without
requiring any external data files. It creates a synthetic vorticity field with
well-separated vortices, generates a "hallucinated" reconstruction with
spurious structures, and shows how persistent homology detects the structural
corruption that RMSE alone cannot distinguish.

Usage:
    python demo_synthetic.py

Output:
    - Console: RMSE, H₀ counts, W₂ distance for each reconstruction
    - File: demo_validation_output.png (comparison plot)

Author: Jesús Garví-Gualda
License: MIT
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tda_pipeline import validate, plot_persistence_diagram


def make_vortex(xx, yy, cx, cy, strength=1.0, sigma=0.08):
    """Create a well-separated Gaussian vortex centered at (cx, cy)."""
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return strength * np.exp(-r2 / (2 * sigma ** 2))


def main():
    print("=" * 65)
    print("  Topological Validation Demo — Synthetic Vortex Field")
    print("=" * 65)

    N = 256
    x = np.linspace(0, 4, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)

    # ─── Ground Truth: 5 well-separated vortices ─────────────────────
    vorticity_true = (
        make_vortex(xx, yy, 0.5, 0.5, strength=1.0, sigma=0.08) +
        make_vortex(xx, yy, 1.3, 0.5, strength=0.9, sigma=0.08) +
        make_vortex(xx, yy, 2.1, 0.5, strength=0.85, sigma=0.08) +
        make_vortex(xx, yy, 2.9, 0.5, strength=0.80, sigma=0.08) +
        make_vortex(xx, yy, 3.5, 0.5, strength=0.75, sigma=0.08)
    )

    # ─── Good reconstruction (faithful, tiny noise) ──────────────────
    np.random.seed(42)
    vorticity_good = vorticity_true + 0.005 * np.random.randn(N, N)

    # ─── Bad reconstruction (competitive RMSE but hallucinated vortices)
    np.random.seed(123)
    vorticity_bad = vorticity_true.copy()
    # Add 7 spurious hallucinated vortices (small amplitude ≈ low RMSE)
    for cx, cy, s in [(0.3, 0.2, 0.35), (0.8, 0.8, 0.30),
                       (1.6, 0.2, 0.32), (2.4, 0.8, 0.28),
                       (2.7, 0.2, 0.25), (3.2, 0.8, 0.30),
                       (3.8, 0.3, 0.27)]:
        vorticity_bad += make_vortex(xx, yy, cx, cy, strength=s, sigma=0.05)
    vorticity_bad += 0.003 * np.random.randn(N, N)

    # ─── Validate ──────────────────────────────────────────────────────
    # Negate the field: sublevel-set filtration finds minima; negating
    # turns vortex peaks into deep minima → they become persistent H₀
    # features born early in the filtration.
    ref = -vorticity_true
    good = -vorticity_good
    bad = -vorticity_bad

    print("\n── Good Reconstruction (noise only, no hallucinations) ──")
    res_good = validate(good, ref, normalize=True)
    print(f"   RMSE      = {res_good['rmse']:.4f}")
    print(f"   H₀ (ref)  = {res_good['H0_ref']}")
    print(f"   H₀ (recon)= {res_good['H0_recon']}")
    print(f"   W₂        = {res_good['W2']:.4f}")

    print("\n── Bad Reconstruction (7 hallucinated vortices) ──")
    res_bad = validate(bad, ref, normalize=True)
    print(f"   RMSE      = {res_bad['rmse']:.4f}")
    print(f"   H₀ (ref)  = {res_bad['H0_ref']}")
    print(f"   H₀ (recon)= {res_bad['H0_recon']}")
    print(f"   W₂        = {res_bad['W2']:.4f}")

    # ─── Key finding ───────────────────────────────────────────────────
    print("\n── Summary ──")
    rmse_ratio = res_bad['rmse'] / res_good['rmse'] if res_good['rmse'] > 0 else float('inf')
    w2_ratio = res_bad['W2'] / res_good['W2'] if res_good['W2'] > 0 else float('inf')
    print(f"   RMSE ratio (bad/good)       = {rmse_ratio:.2f}x")
    print(f"   H₀ overcounting (bad)       = {res_bad['H0_recon']} vs {res_bad['H0_ref']} true")
    print(f"   W₂ ratio (bad/good)         = {w2_ratio:.2f}x")
    if res_bad['H0_recon'] > res_bad['H0_ref']:
        print("   → The bad reconstruction hallucinates extra structures that")
        print("     RMSE alone cannot detect. Topological validation catches it.")
    else:
        print("   → Pipeline correctly processed both fields.")

    # ─── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 1: Vorticity fields
    vmin, vmax = vorticity_true.min(), vorticity_true.max()
    levels = np.linspace(vmin - 0.05, vmax + 0.05, 30)

    for ax, field, title in zip(
        axes[0],
        [vorticity_true, vorticity_good, vorticity_bad],
        ['Ground Truth (5 vortices)',
         f'Good Recon (RMSE={res_good["rmse"]:.4f})',
         f'Bad Recon (RMSE={res_bad["rmse"]:.4f})']
    ):
        ax.contourf(xx, yy, field, levels=levels, cmap='RdBu_r', extend='both')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    # Row 2: Persistence diagrams
    plot_persistence_diagram(res_good['dgm_ref'], dim=0,
                             title=f'Reference ($H_0$={res_good["H0_ref"]})',
                             ax=axes[1, 0], color='#2ca02c')
    plot_persistence_diagram(res_good['dgm_recon'], dim=0,
                             title=f'Good ($H_0$={res_good["H0_recon"]}, $W_2$={res_good["W2"]:.3f})',
                             ax=axes[1, 1], color='#1f77b4')
    plot_persistence_diagram(res_bad['dgm_recon'], dim=0,
                             title=f'Bad ($H_0$={res_bad["H0_recon"]}, $W_2$={res_bad["W2"]:.3f})',
                             ax=axes[1, 2], color='#d62728')

    fig.suptitle('Topological Validation: RMSE vs Persistent Homology',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    out_path = 'demo_validation_output.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Plot saved to: {out_path}")


if __name__ == '__main__':
    main()

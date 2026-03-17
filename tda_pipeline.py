#!/usr/bin/env python3
"""
Topological Validation Pipeline for Scalar Field Reconstructions.

Implements persistent homology-based structural validation using GUDHI.
Detects structural hallucinations in reconstructed scalar fields by comparing
the H₀ persistence diagram (connected components) and Wasserstein-2 distance
against a reference field.

Reference:
    Garví-Gualda, J. (2026). "Beyond RMSE: Structural validation of
    physics-informed neural network reconstructions via persistent homology."

Dependencies:
    - gudhi >= 3.9
    - numpy >= 1.24
    - scipy >= 1.11

Author: Jesús Garví-Gualda
License: MIT
"""

import numpy as np

# GUDHI for persistent homology
import gudhi
from gudhi.wasserstein import wasserstein_distance as _gudhi_wasserstein


# ==============================================================================
#  Core Functions
# ==============================================================================

def normalize_field(field: np.ndarray) -> np.ndarray:
    """
    Normalize a scalar field by its maximum absolute value.

    This ensures that Wasserstein distances measure purely structural
    differences, not amplitude scaling mismatches.

    Args:
        field: 2D numpy array (scalar field on a regular grid).

    Returns:
        Normalized field with values in [-1, 1].
    """
    max_abs = np.max(np.abs(field))
    if max_abs < 1e-15:
        return field.copy()
    return field / max_abs


def compute_persistence(field_2d: np.ndarray, max_dim: int = 0) -> dict:
    """
    Compute sublevel-set persistent homology on a 2D scalar field.

    Uses GUDHI's CubicalComplex for grid data. Infinite-death features
    (the essential class in H₀) are clamped to the field maximum so that
    all features participate in Wasserstein distance calculations.

    Args:
        field_2d: 2D numpy array (the scalar field).
        max_dim: Maximum homology dimension (0 = components, 1 = loops).

    Returns:
        Dictionary with keys:
            'H0': (n, 2) array of (birth, death) pairs
            'H0_essential': boolean mask for essential features
            'H0_lifetimes': array of lifetimes
            (similarly for H1 if max_dim >= 1)
    """
    cc = gudhi.CubicalComplex(
        top_dimensional_cells=field_2d.flatten(),
        dimensions=list(field_2d.shape)
    )
    cc.compute_persistence()

    result = {}
    field_max = float(np.max(field_2d))

    for dim in range(max_dim + 1):
        name = f'H{dim}'
        pairs = cc.persistence_intervals_in_dimension(dim)

        if len(pairs) > 0:
            is_essential = ~np.isfinite(pairs[:, 1])
            clamped = pairs.copy()
            clamped[is_essential, 1] = field_max

            lifetimes = clamped[:, 1] - clamped[:, 0]
            nontrivial = lifetimes > 1e-12
            clamped = clamped[nontrivial]
            is_essential = is_essential[nontrivial]
            lifetimes = lifetimes[nontrivial]
        else:
            clamped = np.empty((0, 2))
            is_essential = np.empty(0, dtype=bool)
            lifetimes = np.empty(0)

        result[name] = clamped
        result[f'{name}_essential'] = is_essential
        result[f'{name}_lifetimes'] = lifetimes

    return result


def count_significant_features(
    persistence_result: dict,
    dim: int = 0,
    threshold_ratio: float = 0.10
) -> int:
    """
    Count topologically significant features (H_dim) after thresholding.

    Features with persistence (lifetime) below `threshold_ratio` times the
    maximum persistence are considered numerical noise and excluded.

    Args:
        persistence_result: Output from compute_persistence().
        dim: Homology dimension (0 for connected components).
        threshold_ratio: Fraction of max persistence below which features
                         are discarded (default 0.10 = 10%).

    Returns:
        Number of significant features (int).
    """
    name = f'H{dim}'
    is_essential = persistence_result.get(f'{name}_essential', np.array([], dtype=bool))
    lifetimes = persistence_result.get(f'{name}_lifetimes', np.array([]))

    if len(lifetimes) == 0:
        return 0

    # Only count finite (non-essential) features
    finite_mask = ~is_essential if len(is_essential) > 0 else np.ones(len(lifetimes), dtype=bool)
    finite_lifetimes = lifetimes[finite_mask]

    if len(finite_lifetimes) == 0:
        return 0

    max_pers = float(np.max(finite_lifetimes))
    threshold = threshold_ratio * max_pers
    return int(np.sum(finite_lifetimes > threshold))


def topological_distance(
    dgm1: dict,
    dgm2: dict,
    dim: int = 0,
    order: int = 2
) -> float:
    """
    Compute the Wasserstein-p distance between two persistence diagrams.

    Args:
        dgm1, dgm2: Persistence results from compute_persistence().
        dim: Homology dimension.
        order: Wasserstein order (2 = W₂).

    Returns:
        Wasserstein-p distance (float).
    """
    name = f'H{dim}'
    d1 = dgm1.get(name, np.empty((0, 2)))
    d2 = dgm2.get(name, np.empty((0, 2)))

    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    if len(d1) == 0:
        d1 = np.empty((0, 2))
    if len(d2) == 0:
        d2 = np.empty((0, 2))

    return float(_gudhi_wasserstein(d1, d2, order=order))


def validate(
    field_recon: np.ndarray,
    field_ref: np.ndarray,
    threshold_ratio: float = 0.10,
    normalize: bool = True
) -> dict:
    """
    One-call topological validation of a reconstructed field against a reference.

    Computes:
        - L² RMSE between the fields
        - H₀ feature count (significant connected components) for both fields
        - Wasserstein-2 distance between persistence diagrams

    Args:
        field_recon: Reconstructed scalar field (2D numpy array).
        field_ref: Reference/ground-truth scalar field (2D numpy array).
        threshold_ratio: Persistence threshold as fraction of max persistence.
        normalize: Whether to normalize fields before computing TDA.

    Returns:
        Dictionary with keys:
            'rmse': L² RMSE between fields
            'H0_recon': significant H₀ count for reconstructed field
            'H0_ref': significant H₀ count for reference field
            'W2': Wasserstein-2 distance between persistence diagrams
    """
    # RMSE (on un-normalized fields)
    rmse = float(np.sqrt(np.mean((field_recon - field_ref) ** 2)))

    # Normalize if requested
    if normalize:
        f_recon = normalize_field(field_recon)
        f_ref = normalize_field(field_ref)
    else:
        f_recon = field_recon
        f_ref = field_ref

    # Compute persistence
    dgm_recon = compute_persistence(f_recon)
    dgm_ref = compute_persistence(f_ref)

    # Significant feature counts
    h0_recon = count_significant_features(dgm_recon, dim=0, threshold_ratio=threshold_ratio)
    h0_ref = count_significant_features(dgm_ref, dim=0, threshold_ratio=threshold_ratio)

    # Wasserstein-2 distance
    w2 = topological_distance(dgm_recon, dgm_ref, dim=0, order=2)

    return {
        'rmse': rmse,
        'H0_recon': h0_recon,
        'H0_ref': h0_ref,
        'W2': w2,
        'dgm_recon': dgm_recon,
        'dgm_ref': dgm_ref,
    }


# ==============================================================================
#  Visualization
# ==============================================================================

def plot_persistence_diagram(
    persistence_result: dict,
    dim: int = 0,
    title: str = "",
    ax=None,
    color: str = "#1f77b4",
    threshold_ratio: float = 0.10,
):
    """
    Plot a publication-quality persistence diagram.

    Args:
        persistence_result: Output from compute_persistence().
        dim: Homology dimension.
        title: Plot title.
        ax: Matplotlib axes (creates new figure if None).
        color: Marker color.
        threshold_ratio: Draws a shaded band for the noise region.

    Returns:
        Matplotlib axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    name = f'H{dim}'
    pairs = persistence_result[name]
    is_essential = persistence_result.get(f'{name}_essential', np.zeros(len(pairs), dtype=bool))

    finite_mask = ~is_essential
    finite_pairs = pairs[finite_mask] if len(pairs) > 0 else np.empty((0, 2))

    # Diagonal
    plot_pairs = finite_pairs if len(finite_pairs) > 0 else pairs
    if len(plot_pairs) > 0:
        all_vals = np.concatenate([plot_pairs[:, 0], plot_pairs[:, 1]])
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    else:
        lo, hi = 0, 1
    pad = max((hi - lo) * 0.15, 0.05)
    diag = [lo - pad, hi + pad]
    ax.plot(diag, diag, 'k-', alpha=0.2, linewidth=1)
    ax.fill_between(diag, diag, [hi + pad] * 2, color='grey', alpha=0.04)

    # Scatter
    if len(finite_pairs) > 0:
        births = finite_pairs[:, 0]
        deaths = finite_pairs[:, 1]
        size = max(40, min(120, 800 // max(len(finite_pairs), 1)))
        ax.scatter(births, deaths, c=color, s=size, alpha=0.85,
                   edgecolors='#333', linewidths=0.5, zorder=10)

    # Stats
    n_sig = count_significant_features(persistence_result, dim=dim, threshold_ratio=threshold_ratio)
    ax.text(0.97, 0.03, f"$H_{dim}$ = {n_sig}", transform=ax.transAxes,
            va='bottom', ha='right', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#ccc', alpha=0.9))

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    if title:
        ax.set_title(title, fontweight='bold')

    return ax


def plot_lifetime_histogram(
    persistence_results: dict,
    dim: int = 0,
    labels: list = None,
    colors: list = None,
    threshold_ratio: float = 0.10,
    ax=None,
):
    """
    Plot persistence lifetime histogram for multiple fields.

    Args:
        persistence_results: Dict mapping label -> compute_persistence() output.
        dim: Homology dimension.
        labels: Override labels (keys of persistence_results used if None).
        colors: Override colors.
        threshold_ratio: Draws the threshold line.
        ax: Matplotlib axes.

    Returns:
        Matplotlib axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    name = f'H{dim}'
    default_colors = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', '#9467bd']
    if colors is None:
        colors = default_colors
    if labels is None:
        labels = list(persistence_results.keys())

    all_lifetimes = []
    for i, (label, dgm) in enumerate(persistence_results.items()):
        is_essential = dgm.get(f'{name}_essential', np.array([], dtype=bool))
        lifetimes = dgm.get(f'{name}_lifetimes', np.array([]))
        finite_mask = ~is_essential if len(is_essential) > 0 else np.ones(len(lifetimes), dtype=bool)
        lt = lifetimes[finite_mask]
        all_lifetimes.append(lt)

    max_life = max(np.max(lt) for lt in all_lifetimes if len(lt) > 0) if all_lifetimes else 1
    bins = np.linspace(0, max_life * 1.1, 50)

    for i, (label, lt) in enumerate(zip(labels, all_lifetimes)):
        c = colors[i % len(colors)]
        ax.hist(lt, bins=bins, alpha=0.5, label=f'{label} (n={len(lt)})',
                color=c, edgecolor='k', linewidth=0.3)

    threshold_val = threshold_ratio * max_life
    ax.axvline(x=threshold_val, color='k', ls='--', lw=1.5,
               label=f'{int(threshold_ratio*100)}% threshold')

    ax.set_xlabel('Feature Lifetime (Persistence)')
    ax.set_ylabel('Count')
    ax.set_title(f'Persistence Histogram ($H_{dim}$)')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    return ax

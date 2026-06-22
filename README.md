# Topological Validation Pipeline


**Topological validation of scalar field reconstructions via persistent homology.**

A lightweight post-hoc validation tool that uses persistent homology to detect structural hallucinations in reconstructed scalar fields. Standard pointwise metrics (RMSE) cannot distinguish between structurally correct and structurally corrupted reconstructions — this pipeline fills that gap.

## Quick Start

```bash
pip install gudhi numpy scipy matplotlib
python demo_synthetic.py
```

## Reproducibility

`demo_synthetic.py` is a self-contained, deterministic reproduction: it builds a
reference vortex field, a noise-only reconstruction, and a reconstruction with
hallucinated structures, then prints RMSE, H0 counts, and W2 for each. It
requires no external data and runs in a few seconds on a single CPU core. The
core pipeline (`tda_pipeline.py`) has no solver dependencies and applies to any
2D scalar field supplied as a NumPy array.

## What It Does

Given a reconstructed scalar field and a reference, the pipeline computes:

| Metric | What it measures | Standard metric? |
|--------|-----------------|-----------------|
| **RMSE** | Pointwise accuracy | Yes |
| **H0 count** | Number of significant topological features (vortices, shocks, etc.) | Novel |
| **W2 distance** | Wasserstein-2 between persistence diagrams (structural similarity) | Novel |

## Usage

### Python API

```python
from tda_pipeline import validate

# field_recon, field_ref: 2D numpy arrays (e.g., vorticity on a grid)
result = validate(-vorticity_recon, -vorticity_ref)

print(f"RMSE = {result['rmse']:.4f}")
print(f"H₀ (reference)      = {result['H0_ref']}")
print(f"H₀ (reconstruction) = {result['H0_recon']}")
print(f"W₂ distance         = {result['W2']:.4f}")
```

### Command Line

```bash
# Validate a reconstruction against a reference (both as .npz files)
python examples/validate_field.py \
    --recon my_reconstruction.npz \
    --ref   ground_truth.npz \
    --key   vorticity \
    --negate
```

### Core Functions

| Function | Purpose |
|----------|---------|
| `normalize_field(f)` | Max-abs normalization for scale-invariant comparison |
| `compute_persistence(f)` | Sublevel-set persistent homology via GUDHI |
| `count_significant_features(dgm)` | H₀ count with 10% persistence threshold |
| `topological_distance(dgm1, dgm2)` | Wasserstein-2 distance |
| `validate(recon, ref)` | One-call: returns RMSE + H₀ + W₂ |

## Dependencies

- **GUDHI** ≥ 3.9 — persistent homology computation
- **NumPy** ≥ 1.24 — array operations
- **SciPy** ≥ 1.11 — (for examples with RBF interpolation)
- **Matplotlib** ≥ 3.7 — visualization

## Method

The pipeline uses **sublevel-set persistent homology** on scalar fields discretized on regular grids:

1. **Normalize** the field by its maximum absolute value
2. **Compute** cubical persistent homology using GUDHI's `CubicalComplex`
3. **Threshold** features at 10% of maximum persistence (separates signal from noise)
4. **Compare** persistence diagrams via the Wasserstein-2 distance

The computation scales as **O(N)** for H₀ features (Union-Find with path compression), where N is the number of grid cells. On a single CPU core, a 4096×4096 grid completes in ~20 seconds.

## Citation

```bibtex
@misc{garvi2026topological,
  title={Topological validation of neural {PDE} solver reconstructions:
         a persistent-homology layer with a maximum-principle guarantee},
  author={Garv{\'\i}-Gualda, Jes{\'u}s},
  year={2026},
  note={Preprint}
}
```

## License

MIT

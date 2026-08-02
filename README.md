# Topological Validation Pipeline

**Topological validation of scalar field reconstructions via persistent homology.**

A lightweight post-hoc validation tool that uses persistent homology to detect
structural hallucinations in reconstructed scalar fields. Standard pointwise
metrics such as RMSE cannot distinguish between a structurally correct
reconstruction and a structurally corrupted one, and this pipeline fills that
gap.

## Quick Start

```bash
pip install -r requirements.txt
python demo_synthetic.py
```

Installing from `requirements.txt` matters: `gudhi.wasserstein` needs the `POT`
package, and GUDHI does not pull it in. Installing GUDHI alone leads to
`ModuleNotFoundError: No module named 'ot'` as soon as the Wasserstein distance
is computed.

## Reproducibility

`demo_synthetic.py` is a self-contained, deterministic reproduction. It builds a
reference vortex field, a noise-only reconstruction, and a reconstruction with
hallucinated structures, then prints RMSE, H0 counts, and W2 for each. It needs
no external data and runs in a few seconds on a single CPU core. The core
pipeline (`tda_pipeline.py`) has no solver dependencies and applies to any 2D
scalar field supplied as a NumPy array.

Running the demo prints these values:

| Reconstruction | RMSE | H0 (reference) | H0 (reconstruction) | W2 |
|---|---|---|---|---|
| Good (noise only) | 0.0050 | 4 | 4 | 0.2045 |
| Bad (7 hallucinated vortices) | 0.0349 | 4 | 11 | 0.3903 |

The bad reconstruction has an RMSE only about 7 times larger than the good one,
which on its own reads as a mildly noisier fit. The H0 count goes from 4 to 11,
which says plainly that seven structures were invented.

These numbers are stable across the version ranges in `requirements.txt`. The
tracked figure `demo_validation_output.png` is not: matplotlib changes rendering
details between releases, so a regenerated figure is visually equivalent but not
byte-identical. `requirements-lock.txt` records the exact versions used to
produce the tracked figure and the table above (verified on 25 July 2026 with
Python 3.13).

`test_pipeline.py` holds regression tests with known-count synthetic fields on
square and non-square grids, including a transpose-invariance check. Run them
with `pytest` or directly with `python test_pipeline.py`.

## Counting convention for H0

The pipeline reports **reduced H0**: the essential class is excluded from the
count. In a sublevel-set filtration of a field with K well-separated basins, one
component is born first and never dies, so K basins give K-1 finite H0 features.

This is why the synthetic demo builds 5 vortices and reports H0 = 4, and why the
corrupted field with 5 + 7 = 12 vortices reports H0 = 11. The convention is
deliberate and matches the protocol used in the manuscript: what matters for
validation is the difference between two counts computed the same way, and the
essential class contributes the same constant offset to both. If you need the
raw number of basins, add 1.

## What It Does

Given a reconstructed scalar field and a reference, the pipeline computes:

| Metric | What it measures | Where it comes from |
|--------|-----------------|---------------------|
| **RMSE** | Pointwise accuracy | Standard error metric |
| **H0 count** | Number of significant topological features such as vortices or shocks | Standard persistent homology, applied here as a structural check |
| **W2 distance** | Wasserstein-2 between persistence diagrams, that is, structural similarity | Standard distance between diagrams, applied here to compare reconstruction against reference |

Neither the H0 count nor the Wasserstein distance is a new mathematical object.
Both are established tools in topological data analysis. What this repository
contributes is their use as a validation layer for field reconstructions,
together with the evidence that they separate cases RMSE ranks as comparable.

## Usage

### Python API

```python
from tda_pipeline import validate

# field_recon, field_ref: 2D numpy arrays (for example, vorticity on a grid)
result = validate(-vorticity_recon, -vorticity_ref)

print(f"RMSE = {result['rmse']:.4f}")
print(f"H0 (reference)      = {result['H0_ref']}")
print(f"H0 (reconstruction) = {result['H0_recon']}")
print(f"W2 distance         = {result['W2']:.4f}")
```

The fields are negated because the filtration is a sublevel-set one, which finds
minima. Negating turns vorticity peaks into deep minima, so they appear as
persistent H0 features born early in the filtration.

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
| `count_significant_features(dgm)` | Reduced H0 count with a 10% persistence threshold |
| `topological_distance(dgm1, dgm2)` | Wasserstein-2 distance |
| `validate(recon, ref)` | One call, returns RMSE, H0 counts and W2 |

## Dependencies

- **GUDHI** >= 3.9, persistent homology computation
- **NumPy** >= 1.24, array operations
- **POT** >= 0.9, required by `gudhi.wasserstein` for the Wasserstein distance
- **Matplotlib** >= 3.7, visualization

## Method

The pipeline uses **sublevel-set persistent homology** on scalar fields
discretized on regular grids:

1. **Normalize** the field by its maximum absolute value
2. **Compute** cubical persistent homology using GUDHI's `CubicalComplex`
3. **Threshold** features at 10% of maximum persistence, which separates signal
   from numerical noise
4. **Compare** persistence diagrams via the Wasserstein-2 distance

H0 features are computed by union-find with path compression once the
filtration values are ordered, giving O(N alpha(N)) after the sort and
O(N log N) overall with a comparison sort, where N is the number of grid
cells. Measured wall-clock time is close to linear over the tested range: on a
single CPU core, a 4096x4096 grid completes in about 20 seconds.

## Citation

This code accompanies the preprint deposited on Zenodo:

```bibtex
@misc{garvi2026topological,
  title={Topological validation of neural {PDE} solvers: pointwise error
         metrics cannot certify derived structure},
  author={Garv{\'\i}-Gualda, Jes{\'u}s},
  year={2026},
  howpublished={Zenodo preprint},
  doi={10.5281/zenodo.21707631}
}
```

The DOI above points to the current version; 10.5281/zenodo.18958345 resolves
to all versions, including the earlier deposit titled "Beyond RMSE". The
manuscript is under review, and this citation will be updated when it appears.

`CITATION.cff` carries the same information in machine-readable form.

## License

MIT

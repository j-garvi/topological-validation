"""Regression tests for tda_pipeline.

Run with pytest, or directly: python test_pipeline.py
"""

import numpy as np

from tda_pipeline import compute_persistence, count_significant_features, validate


def gaussian_wells(shape, centers, width=4.0):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    f = np.zeros(shape)
    for cy, cx in centers:
        f -= np.exp(-(((y - cy) / width) ** 2 + ((x - cx) / width) ** 2))
    return f


def test_square_field_count():
    # Two wells on a square grid: reduced H0 count is 1.
    f = gaussian_wells((64, 64), [(20, 20), (44, 44)])
    dgm = compute_persistence(f)
    assert count_significant_features(dgm) == 1


def test_nonsquare_field_count():
    # Five wells on a 40x90 grid: reduced H0 count is 4. The flattened
    # C-order construction reported 64 here before the shaped-array fix.
    f = gaussian_wells((40, 90), [(10, 15), (10, 45), (10, 75), (30, 30), (30, 60)])
    dgm = compute_persistence(f)
    assert count_significant_features(dgm) == 4


def test_transpose_invariance():
    # The count is a property of the field, not of its storage orientation.
    f = gaussian_wells((40, 90), [(10, 15), (10, 45), (10, 75), (30, 30), (30, 60)])
    a = count_significant_features(compute_persistence(f))
    b = count_significant_features(compute_persistence(f.T.copy()))
    assert a == b == 4


def test_essential_class_excluded_from_count():
    # A single-well field has one component in total; the essential class is
    # excluded, so the reduced count is 0.
    f = gaussian_wells((64, 64), [(32, 32)])
    dgm = compute_persistence(f)
    assert count_significant_features(dgm) == 0


def test_validate_identical_fields():
    f = gaussian_wells((40, 90), [(10, 15), (30, 60)])
    out = validate(f, f)
    assert out["rmse"] == 0.0
    assert out["H0_recon"] == out["H0_ref"]
    assert out["W2"] == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")

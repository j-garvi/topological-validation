#!/usr/bin/env python3
"""
Example: Validate an arbitrary scalar field from a .npz file.

Usage:
    python examples/validate_field.py --recon recon.npz --ref ref.npz --key vorticity

The .npz files should each contain a 2D array accessible via the given key.
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path so we can import tda_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tda_pipeline import validate


def main():
    parser = argparse.ArgumentParser(
        description='Topological validation of a reconstructed scalar field.'
    )
    parser.add_argument('--recon', required=True, help='Path to reconstructed field (.npz)')
    parser.add_argument('--ref', required=True, help='Path to reference field (.npz)')
    parser.add_argument('--key', default='field', help='Array key inside .npz files (default: "field")')
    parser.add_argument('--threshold', type=float, default=0.10,
                        help='Persistence threshold ratio (default: 0.10)')
    parser.add_argument('--negate', action='store_true',
                        help='Negate the field before TDA (use for vorticity)')
    args = parser.parse_args()

    # Load fields
    recon_data = np.load(args.recon)
    ref_data = np.load(args.ref)

    if args.key not in recon_data:
        print(f"Error: key '{args.key}' not found in {args.recon}")
        print(f"  Available keys: {list(recon_data.keys())}")
        sys.exit(1)

    if args.key not in ref_data:
        print(f"Error: key '{args.key}' not found in {args.ref}")
        print(f"  Available keys: {list(ref_data.keys())}")
        sys.exit(1)

    field_recon = recon_data[args.key]
    field_ref = ref_data[args.key]

    if args.negate:
        field_recon = -field_recon
        field_ref = -field_ref

    print(f"Reconstructed field shape: {field_recon.shape}")
    print(f"Reference field shape:     {field_ref.shape}")

    # Validate
    result = validate(field_recon, field_ref, threshold_ratio=args.threshold)

    print(f"\n{'─' * 45}")
    print(f"  RMSE         = {result['rmse']:.6f}")
    print(f"  H₀ (ref)     = {result['H0_ref']}")
    print(f"  H₀ (recon)   = {result['H0_recon']}")
    print(f"  W₂           = {result['W2']:.6f}")
    print(f"{'─' * 45}")

    if result['H0_recon'] > result['H0_ref']:
        excess = result['H0_recon'] - result['H0_ref']
        print(f"  ⚠ WARNING: {excess} hallucinated features detected!")
    elif result['H0_recon'] < result['H0_ref']:
        deficit = result['H0_ref'] - result['H0_recon']
        print(f"  ⚠ NOTE: {deficit} features under-resolved (conservative error).")
    else:
        print(f"  ✅ Topological agreement: H₀ counts match.")


if __name__ == '__main__':
    main()

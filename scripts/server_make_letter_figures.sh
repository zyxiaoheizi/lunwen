#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-outputs/letter_figures}"

"${PYTHON_BIN}" scripts/plot_letter_figures.py \
  --p8-comparison "${P8_COMPARISON:-outputs/p8/comparison/grid_method_comparison.csv}" \
  --pilot-comparisons \
    "${P8_COMPARISON:-outputs/p8/comparison/grid_method_comparison.csv}" \
    "${P24_COMPARISON:-outputs/p24/comparison/grid_method_comparison.csv}" \
    "${P48_COMPARISON:-outputs/p48/comparison/grid_method_comparison.csv}" \
  --snr-csv "${SNR_CSV:-outputs/p8/snr_curve/snr_nmse_curve.csv}" \
  --ablation-csv "${ABLATION_CSV:-outputs/p8/ablations/comparison/grid_method_comparison.csv}" \
  --active-csv "${ACTIVE_CSV:-outputs/p8/active_basis/active_basis_results.csv}" \
  --outdir "${OUTDIR}"

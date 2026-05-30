#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTDIR="${OUTDIR:-outputs/letter_figures}"

"${PYTHON_BIN}" scripts/plot_letter_figures.py \
  --p8-comparison "${P8_COMPARISON:-outputs/p8/comparison/grid_method_comparison.csv}" \
  --pilot-comparisons \
    "${P8_COMPARISON:-outputs/p8/comparison/grid_method_comparison.csv}" \
    "${P16_COMPARISON:-outputs/p16/comparison/grid_method_comparison.csv}" \
    "${P24_COMPARISON:-outputs/p24/comparison/grid_method_comparison.csv}" \
    "${P32_COMPARISON:-outputs/p32/comparison/grid_method_comparison.csv}" \
    "${P48_COMPARISON:-outputs/p48/comparison/grid_method_comparison.csv}" \
  --snr-csv "${SNR_CSV:-outputs/p8/snr_curve/snr_nmse_curve.csv}" \
  --ber-csv "${BER_CSV:-outputs/p8/ber_curve/ber_curve.csv}" \
  --ablation-csv "${ABLATION_CSV:-outputs/p8/ablations/comparison/grid_method_comparison.csv}" \
  --active-csv "${ACTIVE_CSV:-outputs/p8/active_basis/active_basis_results.csv}" \
  --complexity-csv "${COMPLEXITY_CSV:-outputs/p8/complexity/neural_complexity.csv}" \
  --outdir "${OUTDIR}"

CHANNEL_EXAMPLE_TEST="${CHANNEL_EXAMPLE_TEST:-data/grid/grid_p8_tdl_a_test_4000.npz}"
CHANNEL_EXAMPLE_CKPT="${CHANNEL_EXAMPLE_CKPT:-outputs/p8/pf_msbnet_a_best/pf_msbnet_grid_best.pt}"
if [[ "${MAKE_CHANNEL_EXAMPLE:-1}" == "1" && -f "${CHANNEL_EXAMPLE_TEST}" && -f "${CHANNEL_EXAMPLE_CKPT}" ]]; then
  "${PYTHON_BIN}" scripts/plot_channel_example.py \
    --test "${CHANNEL_EXAMPLE_TEST}" \
    --checkpoint "${CHANNEL_EXAMPLE_CKPT}" \
    --outdir "${OUTDIR}"
fi

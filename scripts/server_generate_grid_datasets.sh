#!/usr/bin/env bash
set -euo pipefail

# Paper-facing 2D time-frequency datasets.
# Defaults follow the ChannelNet/DeepPilotDesign style:
# 72 subcarriers x 14 OFDM symbols with 48 pilots.

TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
VAL_SAMPLES="${VAL_SAMPLES:-4000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p data/grid

echo "[1/6] Train: Rayleigh TDL-A"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TRAIN_SAMPLES}" \
  --profile tdl-a \
  --num-pilots 48 \
  --seed 202700 \
  --out data/grid/grid_tdl_a_train_${TRAIN_SAMPLES}.npz

echo "[2/6] Val: Rayleigh TDL-A"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${VAL_SAMPLES}" \
  --profile tdl-a \
  --num-pilots 48 \
  --seed 202701 \
  --out data/grid/grid_tdl_a_val_${VAL_SAMPLES}.npz

echo "[3/6] Test: Rayleigh TDL-A"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TEST_SAMPLES}" \
  --profile tdl-a \
  --num-pilots 48 \
  --seed 202702 \
  --out data/grid/grid_tdl_a_test_${TEST_SAMPLES}.npz

echo "[4/6] Shift test: Rayleigh TDL-B"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TEST_SAMPLES}" \
  --profile tdl-b \
  --num-pilots 48 \
  --seed 202703 \
  --out data/grid/grid_tdl_b_test_${TEST_SAMPLES}.npz

echo "[5/6] Shift test: Rayleigh TDL-C"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TEST_SAMPLES}" \
  --profile tdl-c \
  --num-pilots 48 \
  --seed 202704 \
  --out data/grid/grid_tdl_c_test_${TEST_SAMPLES}.npz

echo "[6/6] Shift test: Rician TDL-A"
"${PYTHON_BIN}" scripts/generate_grid_dataset.py \
  --samples "${TEST_SAMPLES}" \
  --profile tdl-a \
  --channel rician \
  --rician-k 5 \
  --num-pilots 48 \
  --seed 202705 \
  --out data/grid/grid_rician_tdl_a_test_${TEST_SAMPLES}.npz

echo "Done. Datasets are under data/grid"

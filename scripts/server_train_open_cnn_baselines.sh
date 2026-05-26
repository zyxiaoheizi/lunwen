#!/usr/bin/env bash
set -euo pipefail

# Train paper-code-aligned CNN baselines:
# - srcnn: ChannelNet/DeepPilotDesign SRCNN super-resolution module
# - channelnet: SRCNN + DnCNN restoration pipeline

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"

TRAIN="${TRAIN:-data/grid/grid_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_rician_tdl_a_test_4000.npz}"

COMMON_TESTS=("${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}")

echo "[1/2] Training SRCNN baseline"
"${PYTHON_BIN}" scripts/train_grid_cnn.py \
  --model srcnn \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --outdir outputs/cnn_srcnn

echo "[2/2] Training ChannelNet-style SRCNN + DnCNN baseline"
"${PYTHON_BIN}" scripts/train_grid_cnn.py \
  --model channelnet \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --depth 8 \
  --num-workers "${NUM_WORKERS}" \
  --outdir outputs/cnn_channelnet

echo "Done. Summaries are under outputs/cnn_srcnn and outputs/cnn_channelnet"

#!/usr/bin/env bash
set -euo pipefail

# Train paper-code-aligned CNN baselines:
# - srcnn: ChannelNet/DeepPilotDesign SRCNN super-resolution module
# - channelnet: SRCNN + DnCNN restoration pipeline
# - reesnet: residual CNN / ReEsNet-style estimator

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-120}"
SRCNN_EPOCHS="${SRCNN_EPOCHS:-${EPOCHS}}"
DNCNN_EPOCHS="${DNCNN_EPOCHS:-${EPOCHS}}"
REESNET_EPOCHS="${REESNET_EPOCHS:-${EPOCHS}}"
CHANNELNET_FINE_TUNE_EPOCHS="${CHANNELNET_FINE_TUNE_EPOCHS:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SRCNN_LR="${SRCNN_LR:-1e-3}"
DNCNN_LR="${DNCNN_LR:-1e-3}"
REESNET_LR="${REESNET_LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
CHANNELNET_HIDDEN="${CHANNELNET_HIDDEN:-64}"
CHANNELNET_DEPTH="${CHANNELNET_DEPTH:-8}"
REESNET_HIDDEN="${REESNET_HIDDEN:-64}"
REESNET_DEPTH="${REESNET_DEPTH:-8}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"

TRAIN="${TRAIN:-data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_${PILOT_TAG}_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz}"

COMMON_TESTS=("${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}")

echo "Training ${PILOT_TAG} CNN baselines"
echo "SRCNN epochs=${SRCNN_EPOCHS}, DnCNN epochs=${DNCNN_EPOCHS}, ReEsNet epochs=${REESNET_EPOCHS}"
echo "lr: SRCNN=${SRCNN_LR}, DnCNN=${DNCNN_LR}, ReEsNet=${REESNET_LR}; weight_decay=${WEIGHT_DECAY}"

echo "[1/3] Training SRCNN baseline"
"${PYTHON_BIN}" scripts/train_grid_cnn.py \
  --model srcnn \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${SRCNN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${SRCNN_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-workers "${NUM_WORKERS}" \
  --outdir "outputs/${PILOT_TAG}/cnn_srcnn"

echo "[2/3] Training ChannelNet paper-style pipeline: fixed SRCNN -> DnCNN"
"${PYTHON_BIN}" scripts/train_channelnet_pipeline.py \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --srcnn-checkpoint "outputs/${PILOT_TAG}/cnn_srcnn/srcnn_grid_best.pt" \
  --epochs "${DNCNN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${DNCNN_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --hidden-channels "${CHANNELNET_HIDDEN}" \
  --depth "${CHANNELNET_DEPTH}" \
  --fine-tune-epochs "${CHANNELNET_FINE_TUNE_EPOCHS}" \
  --num-workers "${NUM_WORKERS}" \
  --outdir "outputs/${PILOT_TAG}/cnn_channelnet"

echo "[3/3] Training ReEsNet-style residual CNN baseline"
"${PYTHON_BIN}" scripts/train_grid_cnn.py \
  --model reesnet \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${REESNET_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${REESNET_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --hidden-channels "${REESNET_HIDDEN}" \
  --depth "${REESNET_DEPTH}" \
  --num-workers "${NUM_WORKERS}" \
  --outdir "outputs/${PILOT_TAG}/cnn_reesnet"

echo "Done. Summaries are under outputs/${PILOT_TAG}/"

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
CHANNELNET_FINE_TUNE_EPOCHS="${CHANNELNET_FINE_TUNE_EPOCHS:-20}"
CHANNELNET_FINE_TUNE_LR="${CHANNELNET_FINE_TUNE_LR:-1e-4}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
SRCNN_LR="${SRCNN_LR:-1e-3}"
DNCNN_LR="${DNCNN_LR:-1e-3}"
REESNET_LR="${REESNET_LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
SRCNN_WEIGHT_DECAY="${SRCNN_WEIGHT_DECAY:-${WEIGHT_DECAY}}"
DNCNN_WEIGHT_DECAY="${DNCNN_WEIGHT_DECAY:-${WEIGHT_DECAY}}"
REESNET_WEIGHT_DECAY="${REESNET_WEIGHT_DECAY:-${WEIGHT_DECAY}}"
CHANNELNET_HIDDEN="${CHANNELNET_HIDDEN:-64}"
CHANNELNET_DEPTH="${CHANNELNET_DEPTH:-8}"
REESNET_HIDDEN="${REESNET_HIDDEN:-64}"
REESNET_DEPTH="${REESNET_DEPTH:-8}"
PILOTS="${PILOTS:-16}"
PILOT_TAG="p${PILOTS}"

OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"

TRAIN="${TRAIN:-data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_${PILOT_TAG}_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz}"

COMMON_TESTS=("${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}")

echo "Training ${PILOT_TAG} CNN baselines"
echo "SRCNN epochs=${SRCNN_EPOCHS}, DnCNN epochs=${DNCNN_EPOCHS}, ReEsNet epochs=${REESNET_EPOCHS}"
echo "batch_size=${BATCH_SIZE}, device=${DEVICE}, num_workers=${NUM_WORKERS}, out_root=${OUT_ROOT}"
echo "lr: SRCNN=${SRCNN_LR}, DnCNN=${DNCNN_LR}, ReEsNet=${REESNET_LR}"
echo "weight_decay: SRCNN=${SRCNN_WEIGHT_DECAY}, DnCNN=${DNCNN_WEIGHT_DECAY}, ReEsNet=${REESNET_WEIGHT_DECAY}"
echo "depth/hidden: ChannelNet=${CHANNELNET_DEPTH}/${CHANNELNET_HIDDEN}, ReEsNet=${REESNET_DEPTH}/${REESNET_HIDDEN}"
echo "early_stopping: patience=${EARLY_STOPPING_PATIENCE}, min_delta=${EARLY_STOPPING_MIN_DELTA}; channelnet_finetune=${CHANNELNET_FINE_TUNE_EPOCHS}@${CHANNELNET_FINE_TUNE_LR}"

echo "[1/3] Training SRCNN baseline"
"${PYTHON_BIN}" scripts/train_grid_cnn.py \
  --model srcnn \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${SRCNN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${SRCNN_LR}" \
  --weight-decay "${SRCNN_WEIGHT_DECAY}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --outdir "${OUT_ROOT}/cnn_srcnn"

echo "[2/3] Training ChannelNet paper-style pipeline: fixed SRCNN -> DnCNN"
"${PYTHON_BIN}" scripts/train_channelnet_pipeline.py \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --srcnn-checkpoint "${OUT_ROOT}/cnn_srcnn/srcnn_grid_best.pt" \
  --epochs "${DNCNN_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${DNCNN_LR}" \
  --weight-decay "${DNCNN_WEIGHT_DECAY}" \
  --hidden-channels "${CHANNELNET_HIDDEN}" \
  --depth "${CHANNELNET_DEPTH}" \
  --fine-tune-epochs "${CHANNELNET_FINE_TUNE_EPOCHS}" \
  --fine-tune-lr "${CHANNELNET_FINE_TUNE_LR}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --outdir "${OUT_ROOT}/cnn_channelnet"

echo "[3/3] Training ReEsNet-style residual CNN baseline"
"${PYTHON_BIN}" scripts/train_grid_cnn.py \
  --model reesnet \
  --train "${TRAIN}" \
  --val "${VAL}" \
  --test "${COMMON_TESTS[@]}" \
  --epochs "${REESNET_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${REESNET_LR}" \
  --weight-decay "${REESNET_WEIGHT_DECAY}" \
  --hidden-channels "${REESNET_HIDDEN}" \
  --depth "${REESNET_DEPTH}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --outdir "${OUT_ROOT}/cnn_reesnet"

echo "Done. Summaries are under ${OUT_ROOT}/"

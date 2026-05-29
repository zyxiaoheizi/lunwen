#!/usr/bin/env bash
set -euo pipefail

# Train PF-MSBNet-A and PF-MSBNet-A2 on mixed-profile spatial-correlation p8 data.

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
TRAIN_PER_PROFILE="${TRAIN_PER_PROFILE:-8000}"
VAL_PER_PROFILE="${VAL_PER_PROFILE:-1000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}_spcorr_mixed}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
NUM_BASIS="${NUM_BASIS:-32}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-128}"
POS_BANDS="${POS_BANDS:-6}"
PCA_MAX_OBSERVATIONS="${PCA_MAX_OBSERVATIONS:-8192}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
RUN_A="${RUN_A:-1}"
RUN_A2="${RUN_A2:-1}"

TRAIN_FILES=(
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_train_${TRAIN_PER_PROFILE}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_b_train_${TRAIN_PER_PROFILE}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_c_train_${TRAIN_PER_PROFILE}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_rician_tdl_a_train_${TRAIN_PER_PROFILE}.npz"
)
VAL_FILES=(
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_val_${VAL_PER_PROFILE}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_b_val_${VAL_PER_PROFILE}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_c_val_${VAL_PER_PROFILE}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_rician_tdl_a_val_${VAL_PER_PROFILE}.npz"
)
TEST_FILES=(
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho00_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho03_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho07_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_a_rho09_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_b_rho07_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_tdl_c_rho07_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_spcorr_rician_tdl_a_rho07_test_${TEST_SAMPLES}.npz"
)

train_one() {
  local name="$1"
  local token_encoder="$2"
  local outdir="${OUT_ROOT}/${name}"
  local extra=()
  if [[ "${token_encoder}" == "mimo_tokenformer" ]]; then
    extra+=(--token-encoder mimo_tokenformer --link-attention-heads 4 --link-attention-layers 1 --link-attention-dropout 0)
  else
    extra+=(--token-encoder flat)
  fi

  echo "Training ${name} on mixed spatial-correlation data"
  "${PYTHON_BIN}" scripts/train_pf_msbnet.py \
    --train "${TRAIN_FILES[@]}" \
    --val "${VAL_FILES[@]}" \
    --test "${TEST_FILES[@]}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --num-basis "${NUM_BASIS}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --pos-bands "${POS_BANDS}" \
    --basis-init pca \
    --pca-max-observations "${PCA_MAX_OBSERVATIONS}" \
    --use-attention \
    --attention-heads 4 \
    --attention-layers 1 \
    --attention-dropout 0 \
    --gate-dropout 0 \
    --basis-dropout 0 \
    --pilot-noise-std 0 \
    --gate-temperature 1.0 \
    --lambda-pilot 0 \
    --lambda-orth 1e-4 \
    --lambda-gate 1e-5 \
    --lambda-gate-entropy 0 \
    --lambda-attention-entropy 0 \
    --min-regularization 1e-4 \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --grad-clip-norm 1.0 \
    --outdir "${outdir}" \
    "${extra[@]}"
}

if [[ "${RUN_A}" == "1" ]]; then
  train_one "pf_msbnet_a" "flat"
fi
if [[ "${RUN_A2}" == "1" ]]; then
  train_one "pf_msbnet_a2" "mimo_tokenformer"
fi

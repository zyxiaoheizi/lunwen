#!/usr/bin/env bash
set -euo pipefail

# Train the PF-MSBNet paper suite for ablation tables. The final full model
# and base model are trained with their own reproducible recipes, while the
# remaining w/o variants share one controlled ablation recipe.

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
MODEL_ROOT="${MODEL_ROOT:-outputs/${PILOT_TAG}}"
ABLATION_ROOT="${ABLATION_ROOT:-${MODEL_ROOT}/ablations}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
TRAIN_BASE="${TRAIN_BASE:-1}"
TRAIN_TUNED="${TRAIN_TUNED:-1}"
TRAIN_ABLATIONS="${TRAIN_ABLATIONS:-1}"
TRAIN_FULL_A_STRICT="${TRAIN_FULL_A_STRICT:-0}"
BASE_EPOCHS="${BASE_EPOCHS:-120}"
BASE_LR="${BASE_LR:-5e-4}"
BASE_EARLY_STOPPING_PATIENCE="${BASE_EARLY_STOPPING_PATIENCE:-20}"
TUNED_EPOCHS="${TUNED_EPOCHS:-80}"
TUNED_LR="${TUNED_LR:-5e-4}"
TUNED_EARLY_STOPPING_PATIENCE="${TUNED_EARLY_STOPPING_PATIENCE:-12}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-60}"
ABLATION_LR="${ABLATION_LR:-5e-4}"
ABLATION_EARLY_STOPPING_PATIENCE="${ABLATION_EARLY_STOPPING_PATIENCE:-8}"
TRAIN="${TRAIN:-data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_${PILOT_TAG}_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz}"

train_base_model() {
  echo "Training final reference: PF-MSBNet base"
  PILOTS="${PILOTS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${MODEL_ROOT}" \
  OUTDIR="${MODEL_ROOT}/pf_msbnet" \
  TRAIN="${TRAIN}" \
  VAL="${VAL}" \
  TEST_TDLA="${TEST_TDLA}" \
  TEST_TDLB="${TEST_TDLB}" \
  TEST_TDLC="${TEST_TDLC}" \
  TEST_RICIAN="${TEST_RICIAN}" \
  EPOCHS="${BASE_EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LR="${BASE_LR}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICE="${DEVICE}" \
  USE_ATTENTION=0 \
  EARLY_STOPPING_PATIENCE="${BASE_EARLY_STOPPING_PATIENCE}" \
  bash scripts/server_train_pf_msbnet.sh
}

train_tuned_model() {
  echo "Training final full model: PF-MSBNet-A tuned"
  PILOTS="${PILOTS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${MODEL_ROOT}" \
  OUTDIR="${MODEL_ROOT}/pf_msbnet_a_best" \
  TRAIN="${TRAIN}" \
  VAL="${VAL}" \
  TEST_TDLA="${TEST_TDLA}" \
  TEST_TDLB="${TEST_TDLB}" \
  TEST_TDLC="${TEST_TDLC}" \
  TEST_RICIAN="${TEST_RICIAN}" \
  EPOCHS="${TUNED_EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LR="${TUNED_LR}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICE="${DEVICE}" \
  EARLY_STOPPING_PATIENCE="${TUNED_EARLY_STOPPING_PATIENCE}" \
  bash scripts/server_train_pf_msbnet_a.sh
}

COMMON=(
  --train "${TRAIN}"
  --val "${VAL}"
  --test "${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}"
  --epochs "${ABLATION_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${ABLATION_LR}"
  --num-basis 32
  --hidden-channels 128
  --basis-init pca
  --pca-max-observations 8192
  --attention-heads 4
  --attention-layers 1
  --attention-dropout 0
  --gate-dropout 0
  --basis-dropout 0
  --pilot-noise-std 0
  --lambda-pilot 0
  --lambda-orth 1e-4
  --lambda-gate 1e-5
  --lambda-gate-entropy 0
  --lambda-attention-entropy 0
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --early-stopping-patience "${ABLATION_EARLY_STOPPING_PATIENCE}"
  --grad-clip-norm 1.0
)

train_variant() {
  local name="$1"
  shift
  echo "Training ablation: ${name}"
  "${PYTHON_BIN}" scripts/train_pf_msbnet.py "${COMMON[@]}" --outdir "${ABLATION_ROOT}/${name}" "$@"
}

if [[ "${TRAIN_BASE}" == "1" ]]; then
  train_base_model
fi

if [[ "${TRAIN_TUNED}" == "1" ]]; then
  train_tuned_model
fi

if [[ "${TRAIN_ABLATIONS}" == "1" ]]; then
  if [[ "${TRAIN_FULL_A_STRICT}" == "1" ]]; then
    train_variant "full_a" --use-attention
  fi
  train_variant "no_pilot_attention"
  train_variant "no_basis_gate" --use-attention --disable-basis-gate
  train_variant "no_learned_w" --use-attention --disable-learned-pilot-weights
  train_variant "no_learned_lambda" --use-attention --disable-learned-regularization
fi

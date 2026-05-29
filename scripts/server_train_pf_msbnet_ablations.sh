#!/usr/bin/env bash
set -euo pipefail

# Train PF-MSBNet-A ablations for paper tables. Defaults target p8 because
# ablations are most informative in the few-pilot regime.

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}/ablations}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
LR="${LR:-5e-4}"
TRAIN="${TRAIN:-data/grid/grid_${PILOT_TAG}_tdl_a_train_32000.npz}"
VAL="${VAL:-data/grid/grid_${PILOT_TAG}_tdl_a_val_4000.npz}"
TEST_TDLA="${TEST_TDLA:-data/grid/grid_${PILOT_TAG}_tdl_a_test_4000.npz}"
TEST_TDLB="${TEST_TDLB:-data/grid/grid_${PILOT_TAG}_tdl_b_test_4000.npz}"
TEST_TDLC="${TEST_TDLC:-data/grid/grid_${PILOT_TAG}_tdl_c_test_4000.npz}"
TEST_RICIAN="${TEST_RICIAN:-data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_4000.npz}"

COMMON=(
  --train "${TRAIN}"
  --val "${VAL}"
  --test "${TEST_TDLA}" "${TEST_TDLB}" "${TEST_TDLC}" "${TEST_RICIAN}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
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
  --early-stopping-patience 8
  --grad-clip-norm 1.0
)

train_variant() {
  local name="$1"
  shift
  echo "Training ablation: ${name}"
  "${PYTHON_BIN}" scripts/train_pf_msbnet.py "${COMMON[@]}" --outdir "${OUT_ROOT}/${name}" "$@"
}

train_variant "full_a" --use-attention
train_variant "no_pilot_attention"
train_variant "no_basis_gate" --use-attention --disable-basis-gate
train_variant "no_learned_w" --use-attention --disable-learned-pilot-weights
train_variant "no_learned_lambda" --use-attention --disable-learned-regularization

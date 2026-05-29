#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-8}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
ABLATION_ROOT="${ABLATION_ROOT:-${OUT_ROOT}/ablations}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
INCLUDE_TUNED="${INCLUDE_TUNED:-1}"
INCLUDE_EXISTING_VARIANTS="${INCLUDE_EXISTING_VARIANTS:-0}"
INCLUDE_STRICT_ABLATIONS="${INCLUDE_STRICT_ABLATIONS:-1}"
INCLUDE_FULL_A_STRICT="${INCLUDE_FULL_A_STRICT:-0}"

TESTS=(
  "data/grid/grid_${PILOT_TAG}_tdl_a_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_b_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_c_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_${TEST_SAMPLES}.npz"
)

CHECKPOINTS=()

add_checkpoint() {
  local spec="$1"
  path="${spec#*=}"
  if [[ -f "${path}" ]]; then
    CHECKPOINTS+=(--checkpoint "${spec}")
  else
    echo "skip missing checkpoint: ${path}"
  fi
}

if [[ "${INCLUDE_TUNED}" == "1" ]]; then
  add_checkpoint "PF-MSBNet base=${OUT_ROOT}/pf_msbnet/pf_msbnet_grid_best.pt"
  add_checkpoint "PF-MSBNet-A tuned=${OUT_ROOT}/pf_msbnet_a_best/pf_msbnet_grid_best.pt"
fi

if [[ "${INCLUDE_EXISTING_VARIANTS}" == "1" ]]; then
  add_checkpoint "PF-MSBNet-A attn-only=${OUT_ROOT}/pf_msbnet_a_attn_only/pf_msbnet_grid_best.pt"
  add_checkpoint "PF-MSBNet-A reg=${OUT_ROOT}/pf_msbnet_a_reg/pf_msbnet_grid_best.pt"
  add_checkpoint "PF-MSBNet-A balanced=${OUT_ROOT}/pf_msbnet_a_balanced/pf_msbnet_grid_best.pt"
  add_checkpoint "PF-MSBNet-A2=${OUT_ROOT}/pf_msbnet_a2/pf_msbnet_grid_best.pt"
fi

if [[ "${INCLUDE_STRICT_ABLATIONS}" == "1" ]]; then
  if [[ "${INCLUDE_FULL_A_STRICT}" == "1" ]]; then
    add_checkpoint "Full-A strict=${ABLATION_ROOT}/full_a/pf_msbnet_grid_best.pt"
  fi
  for spec in \
    "w/o pilot attention=${ABLATION_ROOT}/no_pilot_attention/pf_msbnet_grid_best.pt" \
    "w/o basis gate=${ABLATION_ROOT}/no_basis_gate/pf_msbnet_grid_best.pt" \
    "w/o learned W=${ABLATION_ROOT}/no_learned_w/pf_msbnet_grid_best.pt" \
    "w/o learned lambda=${ABLATION_ROOT}/no_learned_lambda/pf_msbnet_grid_best.pt"; do
    add_checkpoint "${spec}"
  done
fi

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No ablation checkpoints found under ${OUT_ROOT} or ${ABLATION_ROOT}."
  exit 1
fi

"${PYTHON_BIN}" scripts/compare_grid_methods.py \
  --test "${TESTS[@]}" \
  "${CHECKPOINTS[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --outdir "${ABLATION_ROOT}/comparison"

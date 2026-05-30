#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PILOTS="${PILOTS:-16}"
PILOT_TAG="p${PILOTS}"
OUT_ROOT="${OUT_ROOT:-outputs/${PILOT_TAG}}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32000}"
TEST_SAMPLES="${TEST_SAMPLES:-4000}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LMMSE_PROFILE="${LMMSE_PROFILE:-tdl-a}"
LMMSE_DELAY_SPREAD_NS="${LMMSE_DELAY_SPREAD_NS:-300}"
INCLUDE_PAPER_LMMSE="${INCLUDE_PAPER_LMMSE:-1}"
INCLUDE_ALMMSE="${INCLUDE_ALMMSE:-1}"
ALMMSE_TIME_RANK="${ALMMSE_TIME_RANK:-1}"
ALMMSE_FREQ_RANK="${ALMMSE_FREQ_RANK:-2}"
INCLUDE_EMPIRICAL_LMMSE="${INCLUDE_EMPIRICAL_LMMSE:-0}"
INCLUDE_ORACLE_LMMSE="${INCLUDE_ORACLE_LMMSE:-0}"
INCLUDE_MISMATCHED_LMMSE="${INCLUDE_MISMATCHED_LMMSE:-0}"
INCLUDE_DELAY_BEM="${INCLUDE_DELAY_BEM:-0}"
INCLUDE_ORACLE_DELAY_BEM="${INCLUDE_ORACLE_DELAY_BEM:-0}"
INCLUDE_SPARSE_DELAY_BEM="${INCLUDE_SPARSE_DELAY_BEM:-0}"
INCLUDE_ORACLE_SPARSE_DELAY_BEM="${INCLUDE_ORACLE_SPARSE_DELAY_BEM:-0}"
DELAY_BEM_REGULARIZATION="${DELAY_BEM_REGULARIZATION:-1e-2}"
DELAY_BEM_TIME_ORDER="${DELAY_BEM_TIME_ORDER:-0}"
DELAY_BEM_TAPS="${DELAY_BEM_TAPS:-}"
SPARSE_DELAY_BEM_ATOMS="${SPARSE_DELAY_BEM_ATOMS:-4}"
INCLUDE_SRCNN="${INCLUDE_SRCNN:-1}"
INCLUDE_CHANNELNET="${INCLUDE_CHANNELNET:-1}"
INCLUDE_REESNET="${INCLUDE_REESNET:-1}"
INCLUDE_FIXED_BASIS="${INCLUDE_FIXED_BASIS:-0}"
INCLUDE_PL_ERN="${INCLUDE_PL_ERN:-0}"
INCLUDE_PF_MSB_BASE="${INCLUDE_PF_MSB_BASE:-1}"
INCLUDE_PROPOSED="${INCLUDE_PROPOSED:-1}"
INCLUDE_A2="${INCLUDE_A2:-0}"

TESTS=(
  "data/grid/grid_${PILOT_TAG}_tdl_a_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_b_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_tdl_c_test_${TEST_SAMPLES}.npz"
  "data/grid/grid_${PILOT_TAG}_rician_tdl_a_test_${TEST_SAMPLES}.npz"
)

CHECKPOINTS=()
if [[ "${INCLUDE_SRCNN}" == "1" && -f "${OUT_ROOT}/cnn_srcnn/srcnn_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "SRCNN=${OUT_ROOT}/cnn_srcnn/srcnn_grid_best.pt")
fi
if [[ "${INCLUDE_CHANNELNET}" == "1" && -f "${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ChannelNet=${OUT_ROOT}/cnn_channelnet/channelnet_grid_best.pt")
fi
if [[ "${INCLUDE_REESNET}" == "1" && -f "${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "ReEsNet=${OUT_ROOT}/cnn_reesnet/reesnet_grid_best.pt")
fi
if [[ "${INCLUDE_FIXED_BASIS}" == "1" && -f "${OUT_ROOT}/fixed_basis_ridge/fixed_basis_ridge_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "Fixed-Basis Ridge=${OUT_ROOT}/fixed_basis_ridge/fixed_basis_ridge_grid_best.pt")
fi
if [[ "${INCLUDE_PL_ERN}" == "1" && -f "${OUT_ROOT}/pl_ern_reesnet/plern_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PL-ERN(ReEsNet)=${OUT_ROOT}/pl_ern_reesnet/plern_grid_best.pt")
fi
if [[ "${INCLUDE_PL_ERN}" == "1" && -f "${OUT_ROOT}/pl_ern_channelnet/plern_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PL-ERN(ChannelNet)=${OUT_ROOT}/pl_ern_channelnet/plern_grid_best.pt")
fi
if [[ "${INCLUDE_PF_MSB_BASE}" == "1" && -f "${OUT_ROOT}/pf_msbnet/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet=${OUT_ROOT}/pf_msbnet/pf_msbnet_grid_best.pt")
fi
if [[ "${INCLUDE_PROPOSED}" == "1" && -f "${OUT_ROOT}/pf_msbnet_a_best/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A tuned=${OUT_ROOT}/pf_msbnet_a_best/pf_msbnet_grid_best.pt")
elif [[ "${INCLUDE_PROPOSED}" == "1" && -f "${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A=${OUT_ROOT}/pf_msbnet_a/pf_msbnet_grid_best.pt")
fi
if [[ "${INCLUDE_A2}" == "1" && -f "${OUT_ROOT}/pf_msbnet_a2/pf_msbnet_grid_best.pt" ]]; then
  CHECKPOINTS+=(--checkpoint "PF-MSBNet-A2=${OUT_ROOT}/pf_msbnet_a2/pf_msbnet_grid_best.pt")
fi

LMMSE_ARGS=(
  --lmmse-profile "${LMMSE_PROFILE}"
  --lmmse-delay-spread-ns "${LMMSE_DELAY_SPREAD_NS}"
)
if [[ "${INCLUDE_ORACLE_LMMSE}" == "1" ]]; then
  LMMSE_ARGS+=(--include-oracle-lmmse)
fi
if [[ "${INCLUDE_MISMATCHED_LMMSE}" == "1" ]]; then
  LMMSE_ARGS+=(--include-mismatched-lmmse)
fi
if [[ -f "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz" ]]; then
  if [[ "${INCLUDE_PAPER_LMMSE}" == "1" ]]; then
    LMMSE_ARGS+=(
      --include-paper-lmmse
      --paper-lmmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz"
    )
  fi
  if [[ "${INCLUDE_ALMMSE}" == "1" ]]; then
    LMMSE_ARGS+=(
      --include-almmse
      --almmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz"
      --almmse-time-rank "${ALMMSE_TIME_RANK}"
      --almmse-freq-rank "${ALMMSE_FREQ_RANK}"
    )
  fi
  if [[ "${INCLUDE_EMPIRICAL_LMMSE}" == "1" ]]; then
    LMMSE_ARGS+=(
      --include-empirical-lmmse
      --empirical-lmmse-train "data/grid/grid_${PILOT_TAG}_tdl_a_train_${TRAIN_SAMPLES}.npz"
    )
  fi
fi
if [[ "${INCLUDE_DELAY_BEM}" == "1" ]]; then
  LMMSE_ARGS+=(
    --include-delay-bem
    --delay-bem-regularization "${DELAY_BEM_REGULARIZATION}"
    --delay-bem-time-order "${DELAY_BEM_TIME_ORDER}"
  )
  if [[ -n "${DELAY_BEM_TAPS}" ]]; then
    LMMSE_ARGS+=(--delay-bem-taps "${DELAY_BEM_TAPS}")
  fi
fi
if [[ "${INCLUDE_ORACLE_DELAY_BEM}" == "1" ]]; then
  LMMSE_ARGS+=(
    --include-oracle-delay-bem
    --delay-bem-regularization "${DELAY_BEM_REGULARIZATION}"
    --delay-bem-time-order "${DELAY_BEM_TIME_ORDER}"
  )
fi
if [[ "${INCLUDE_SPARSE_DELAY_BEM}" == "1" ]]; then
  LMMSE_ARGS+=(
    --include-sparse-delay-bem
    --sparse-delay-bem-atoms "${SPARSE_DELAY_BEM_ATOMS}"
    --delay-bem-regularization "${DELAY_BEM_REGULARIZATION}"
    --delay-bem-time-order "${DELAY_BEM_TIME_ORDER}"
  )
  if [[ -n "${DELAY_BEM_TAPS}" ]]; then
    LMMSE_ARGS+=(--delay-bem-taps "${DELAY_BEM_TAPS}")
  fi
fi
if [[ "${INCLUDE_ORACLE_SPARSE_DELAY_BEM}" == "1" ]]; then
  LMMSE_ARGS+=(
    --include-oracle-sparse-delay-bem
    --sparse-delay-bem-atoms "${SPARSE_DELAY_BEM_ATOMS}"
    --delay-bem-regularization "${DELAY_BEM_REGULARIZATION}"
    --delay-bem-time-order "${DELAY_BEM_TIME_ORDER}"
  )
fi

echo "Comparing ${PILOT_TAG}: out_root=${OUT_ROOT}, train_samples=${TRAIN_SAMPLES}, test_samples=${TEST_SAMPLES}"

"${PYTHON_BIN}" scripts/compare_grid_methods.py \
  --test "${TESTS[@]}" \
  "${CHECKPOINTS[@]}" \
  "${LMMSE_ARGS[@]}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --outdir "${OUT_ROOT}/comparison"

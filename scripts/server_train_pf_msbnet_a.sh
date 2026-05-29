#!/usr/bin/env bash
set -euo pipefail

# Train PF-MSBNet-A:
# PF-MSBNet with relative pilot attention, basis-pilot cross attention, and
# sparse basis-gate regularization. It writes to outputs/pXX/pf_msbnet_a by default.

USE_ATTENTION="${USE_ATTENTION:-1}" \
ATTENTION_HEADS="${ATTENTION_HEADS:-4}" \
ATTENTION_LAYERS="${ATTENTION_LAYERS:-1}" \
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.1}" \
GATE_DROPOUT="${GATE_DROPOUT:-0}" \
BASIS_DROPOUT="${BASIS_DROPOUT:-0}" \
PILOT_NOISE_STD="${PILOT_NOISE_STD:-0.01}" \
LAMBDA_GATE_ENTROPY="${LAMBDA_GATE_ENTROPY:-5e-4}" \
LAMBDA_ATTENTION_ENTROPY="${LAMBDA_ATTENTION_ENTROPY:-1e-4}" \
LAMBDA_GATE="${LAMBDA_GATE:-0}" \
OUTDIR="${OUTDIR:-}" \
bash scripts/server_train_pf_msbnet.sh

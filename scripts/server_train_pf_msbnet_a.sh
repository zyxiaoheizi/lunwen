#!/usr/bin/env bash
set -euo pipefail

# Train PF-MSBNet-A:
# PF-MSBNet with relative pilot attention, basis-pilot cross attention, and
# sparse basis-gate regularization. It writes to outputs/pXX/pf_msbnet_a by default.

USE_ATTENTION="${USE_ATTENTION:-1}" \
ATTENTION_HEADS="${ATTENTION_HEADS:-4}" \
ATTENTION_LAYERS="${ATTENTION_LAYERS:-1}" \
LAMBDA_GATE_ENTROPY="${LAMBDA_GATE_ENTROPY:-1e-3}" \
LAMBDA_GATE="${LAMBDA_GATE:-0}" \
OUTDIR="${OUTDIR:-}" \
bash scripts/server_train_pf_msbnet.sh

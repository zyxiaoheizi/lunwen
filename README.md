# Adapter MIMO-OFDM

Minimal Python baseline for MIMO-OFDM channel-estimation experiments.

中文说明见 [README_zh.md](README_zh.md).

Current scope:

- 2x2 MIMO-OFDM frequency-domain simulator
- Rayleigh/Rician tapped-delay channels
- Standard-like PDP profiles: LTE EPA/EVA/ETU and 3GPP TR 38.901-style TDL-A/B/C
- Orthogonal pilot observations
- Sparse-pilot LS with frequency interpolation
- Oracle frequency-domain LMMSE baseline
- Dataset export for later CNN/Adapter experiments

Run the first baseline curve:

```powershell
python scripts/run_baselines.py --frames 2000 --profile tdl-a --pilot-ratio 0.25
```

Generate a small dataset for later neural estimators:

```powershell
python scripts/generate_dataset.py --samples 5000 --profile tdl-a --out data/rayleigh_tdl_a_p025.npz
```

References used for baseline design:

- Sionna OFDM MIMO channel-estimation examples: https://nvlabs.github.io/sionna/phy/tutorials/OFDM_MIMO_Detection.html
- Sionna OFDM API: https://nvlabs.github.io/sionna/phy/api/ofdm.html
- MATLAB/GitHub LMMSE OFDM reference: https://github.com/vineel49/lmmse
- StructNet-CE motivation paper: https://arxiv.org/abs/2305.13487

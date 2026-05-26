# Phase 1 Baselines

## Goal

Build a trustworthy first-stage MIMO-OFDM channel-estimation simulator before
adding CNN or online Adapter tuning.

This phase verifies:

- frequency-domain 2x2 MIMO-OFDM channel generation
- sparse orthogonal pilot observation
- LS channel estimation with frequency interpolation
- oracle frequency-domain LMMSE interpolation
- dataset export for later neural estimators

## Data Standard

The simulator supports two levels of channel profiles.

1. Standard-like profiles for paper-facing experiments:
   - LTE EPA/EVA/ETU power-delay profiles
   - 3GPP TR 38.901-style TDL-A/B/C profiles

2. Simple controlled profiles for ablations:
   - exponential L-tap Rayleigh/Rician TDL

The default is `tdl-a` with RMS delay spread `300 ns`, subcarrier spacing
`15 kHz`, `64` subcarriers, and `2x2` MIMO. This matches the style of
modern link-level simulations while staying light enough for fast iteration.

## Channel Model

For each antenna pair, frequency response is generated from a tapped-delay
profile:

```text
H[k] = sum_l h_l exp(-j 2 pi f_k tau_l)
```

where `tau_l` and tap powers come from the selected delay profile. Taps are
normalized so the average channel power is one.

## Baselines

LS observes only pilot subcarriers using orthogonal pilots across transmit
antennas:

```text
Y_p[k] = H[k] X_p[k] + N[k]
```

With identity/orthogonal pilots, LS at pilot tones is:

```text
H_LS[k] = Y_p[k] X_p[k]^H
```

For all subcarriers, `LS + linear interp.` linearly interpolates the sparse
pilot estimates along frequency.

Oracle LMMSE uses the true delay profile and noise variance:

```text
R[k,m] = sum_l p_l exp(-j 2 pi (k-m) Delta_f tau_l)
H_LMMSE = R_hp (R_pp + sigma_n^2 I)^(-1) H_LS,p
```

It is an optimistic traditional baseline, useful as an upper reference for
later learning-based methods.

## Current Result

Command:

```powershell
python scripts/run_baselines.py --frames 2000 --profile tdl-a --pilot-ratio 0.25
```

Saved outputs:

- `outputs/baselines/rayleigh_tdl-a_p0.25_n2000.csv`
- `outputs/baselines/rayleigh_tdl-a_p0.25_n2000.png`

Observed NMSE:

| SNR (dB) | LS + linear (dB) | Oracle LMMSE (dB) |
| ---: | ---: | ---: |
| 0 | -1.526 | -9.477 |
| 5 | -6.303 | -13.289 |
| 10 | -10.778 | -17.533 |
| 15 | -14.305 | -21.882 |
| 20 | -16.564 | -26.099 |
| 25 | -17.540 | -30.420 |
| 30 | -18.006 | -35.044 |

Dataset command:

```powershell
python scripts/generate_dataset.py --samples 5000 --profile tdl-a --out data/rayleigh_tdl_a_p025_train_5k.npz
```

Saved arrays:

- `h_true`: `[samples, subcarrier, rx, tx]`, complex64
- `h_ls`: `[samples, subcarrier, rx, tx]`, complex64
- `h_pilot_ls`: `[samples, pilot_subcarrier, rx, tx]`, complex64
- `h_true_ri`: `[samples, 2*rx*tx, subcarrier]`, float32
- `h_ls_ri`: `[samples, 2*rx*tx, subcarrier]`, float32
- profile metadata: pilot indices, SNR, delays, tap powers

## References

- Sionna OFDM/channel-estimation API: https://nvlabs.github.io/sionna/phy/api/ofdm.html
- Sionna 3GPP wireless channel models: https://nvlabs.github.io/sionna/phy/api/channel.wireless.html
- StructNet-CE motivation: https://arxiv.org/abs/2305.13487

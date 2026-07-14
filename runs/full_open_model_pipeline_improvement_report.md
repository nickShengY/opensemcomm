# OpenSemCom Full-Open Model/Pipeline Improvements

Date: 2026-06-17

## Implemented

- Torch MLP receiver head for DINOv3 features.
- GPU-backed login-node short runs with `CUDA_VISIBLE_DEVICES=1`.
- Channel-corruption augmentation during receiver calibration.
- Unknown-aware detector with OpenMax-style and ViM-style evidence.
- Mixed-open calibration manifest with real `open-calibration` rows.
- Power-aware channel transmission: scheduled transmit power now scales symbols before interference/noise and de-scales at the receiver.
- Full-vector fallback: fallback now transmits the full real semantic feature vector instead of a truncated half-vector.
- Configurable accept/refine calibration quantiles are now actually used.
- Traces now include confidence and detector feature breakdowns.

## Best Current Full-Open Run

Run:

`/scratch/nickyun/new_study/opensemcom/runs/dinov3_torch_power_fallback_login_fullopen_results_login_20260617205745.csv`

Trace analysis:

`/scratch/nickyun/new_study/opensemcom/runs/analysis_dinov3_torch_power_fallback_login_login_20260617205745_action_rates.csv`

Mean over seeds 0, 1, 2:

| Variant | Accuracy | Open risk | AUROC | FPR95 | Goodput | OpenOut | Open-set F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unknown-aware safe | 0.0328 | 0.8117 | 0.7064 | 0.7587 | 0.0000 | 0.0000 | 0.6845 |
| torch MLP | 0.0440 | 0.8349 | 0.8000 | 0.6250 | 0.0001 | 0.3667 | 0.6964 |
| torch MLP + power-aware channel | 0.0413 | 0.8606 | 0.9184 | 0.4115 | 0.0002 | 0.4556 | 0.7182 |
| torch MLP + power-aware channel + full fallback | 0.0615 | 0.8347 | 0.9347 | 0.3351 | 0.0002 | 0.5722 | 0.7211 |

## Interpretation

The model/pipeline is now substantially stronger for full-open detection and recovery:

- AUROC improved from 0.7064 to 0.9347.
- FPR95 improved from 0.7587 to 0.3351.
- Full-open accuracy improved from 0.0328 to 0.0615.
- Known-ID full-open accuracy improved to roughly 0.47 in the best run's action analysis.

The remaining weak point is safe accepted throughput:

- Goodput is still very small.
- Accepted open outage remains unstable and too high.
- Risk curves only find tiny nonzero-goodput operating points under <= 0.05 selected-open-outage.

## Claim Status

Stronger defensible claim:

OpenSemCom with DINOv3, torch receiver calibration, power-aware transmission, and full-vector fallback substantially improves full-open OOD detection and known-ID recovery under combined open exposure.

Still not defensible:

OpenSemCom solves full-open semantic communication with reliable safe goodput.

## Next Required Model Step

The next improvement has to train the accept/reject policy directly for selective correctness, not only class prediction/OOD ranking:

- train a joint classifier plus selective risk head using held-out known calibration and open-calibration supervision;
- optimize a selective loss for accepted correctness and unknown rejection;
- report risk-coverage curves as the primary full-open result.

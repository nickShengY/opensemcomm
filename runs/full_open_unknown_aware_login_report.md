# OpenSemCom Unknown-Aware Full-Open Login Runs

Date: 2026-06-17

## Runtime Policy Update

Short feature-based jobs under one hour were run on the login node to avoid Slurm queue delay. All data, manifests, caches, logs, and outputs remained under scratch.

The login node exposes H100 GPUs, but these two runs use precomputed DINOv3 feature vectors and the current receiver path is NumPy/scikit-learn. No CUDA kernels are used by this stage unless the classifier/scorer is rewritten in torch.

## Implementation Added

- Mixed-open calibration manifest support using real `open-calibration` rows.
- Unknown-aware binary detector trained from real known/open DINOv3 calibration features.
- OpenMax-style class-distance evidence.
- ViM-style residual evidence.
- Receiver gating over unknown/OpenMax/ViM scores.
- Leakage-safe manifest splitter for reserving real open rows from eval into `open-calibration`.

## Real Manifest

`/scratch/nickyun/new_study/opensemcom/manifests/opensemcom_real_dinov3_mixed_open_calibration.csv`

- Rows: 11,392
- Calibration rows: 384
- Open-calibration rows: 1,024
- Eval rows: 9,984

## Runs

### Unknown-Aware With Conformal

Outputs:

- `/scratch/nickyun/new_study/opensemcom/runs/dinov3_unknown_fullopen_login_fullopen_results_login_20260617203153.csv`
- `/scratch/nickyun/new_study/opensemcom/runs/analysis_dinov3_unknown_fullopen_login_20260617203153_action_rates.csv`
- `/scratch/nickyun/new_study/opensemcom/runs/analysis_dinov3_unknown_fullopen_login_20260617203153_risk_curves.csv`

Mean over seeds 0, 1, 2:

| Accuracy | Open risk | AUROC | FPR95 | Goodput | Open outage | Open-set F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0328 | 0.8117 | 0.7064 | 0.7587 | 0.0000 | 0.0000 | 0.6845 |

### Unknown-Aware Without Conformal

Outputs:

- `/scratch/nickyun/new_study/opensemcom/runs/dinov3_unknown_noconformal_login_fullopen_results_login_20260617203514.csv`
- `/scratch/nickyun/new_study/opensemcom/runs/analysis_dinov3_unknown_noconformal_login_login_20260617203514_action_rates.csv`
- `/scratch/nickyun/new_study/opensemcom/runs/analysis_dinov3_unknown_noconformal_login_login_20260617203514_risk_curves.csv`

Mean over seeds 0, 1, 2:

| Accuracy | Open risk | AUROC | FPR95 | Goodput | Open outage | Open-set F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0328 | 1.0405 | 0.7064 | 0.7587 | 0.0000 | 0.0000 | 0.6845 |

## Diagnosis

The unknown-aware detector makes the full-open receiver safer, but it is now over-conservative: all full-open decisions are routed to fallback/refine/HARQ, with zero accepted decisions and therefore zero semantic goodput.

Disabling conformal prediction-set gating does not recover accepted decisions. That means the remaining bottleneck is not conformal set size alone; it is the full-open semantic channel/decoder path under combined interference and open exposure.

Risk-coverage analysis again found no threshold with nonzero goodput and selected open outage <= 0.05.

## Next Step

The next implementation step should target the semantic channel/decoder itself:

- train a torch receiver head on noisy full-open channel observations, not only clean/precomputed DINOv3 features;
- add channel-corruption augmentation during classifier training;
- evaluate a higher-repetition or stronger WITT/DeepJSCC-style image semantic codec;
- then rerun the same mixed-open calibration protocol.

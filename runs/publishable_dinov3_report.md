# OpenSemCom DINOv3 Real-Data Results

Date: 2026-06-17

## Scope

This report covers the strongest completed real-data run so far:

- Feature model: DINOv3 ViT-B/16, `facebook/dinov3-vitb16-pretrain-lvd1689m`.
- Model source: existing scratch HF cache at `/scratch/nickyun/hf_cache/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m`.
- Feature extraction Slurm job: `605052`, completed with exit code `0:0`.
- Strict multi-seed evaluation Slurm job: `605112`, completed with exit code `0:0`.
- Feature manifest: `/scratch/nickyun/new_study/opensemcom/manifests/opensemcom_real_facebook__dinov3-vitb16-pretrain-lvd1689m.csv`.
- Result table: `/scratch/nickyun/new_study/opensemcom/runs/dinov3_vitb16_strict_full_results_605112.csv`.
- JSON result table: `/scratch/nickyun/new_study/opensemcom/runs/dinov3_vitb16_strict_full_results_605112.json`.

The DINOv3 feature manifest contains 11,392 real rows. It skips 512 AG News text rows because DINOv3 is image-only. No synthetic or generated fallback rows are used.

## Final Metrics

Mean +/- standard deviation over seeds 0, 1, and 2.

| Regime | n | Accuracy | Semantic outage | Open risk | AUROC OOD | Open F1 | Goodput | Open outage | FPR95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed-id | 3 | 0.9019 +/- 0.0304 | 0.0981 +/- 0.0304 | 0.4860 +/- 0.0283 | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1555 +/- 0.0065 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| channel-open | 3 | 0.9288 +/- 0.0143 | 0.0712 +/- 0.0143 | 0.2347 +/- 0.0372 | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.2323 +/- 0.0074 | 0.0013 +/- 0.0023 | 1.0000 +/- 0.0000 |
| resource-open | 3 | 0.9062 +/- 0.0197 | 0.0938 +/- 0.0197 | 0.4313 +/- 0.0199 | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1628 +/- 0.0093 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| source-open | 3 | 0.4389 +/- 0.0722 | 0.5611 +/- 0.0722 | 0.7125 +/- 0.0127 | 1.0000 +/- 0.0000 | 0.7948 +/- 0.0000 | 0.0460 +/- 0.0019 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| task-open | 3 | 0.3154 +/- 0.0204 | 0.6846 +/- 0.0204 | 0.6886 +/- 0.0141 | 1.0000 +/- 0.0000 | 0.8123 +/- 0.0000 | 0.0508 +/- 0.0021 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| class-open | 3 | 0.0949 +/- 0.0032 | 0.9051 +/- 0.0032 | 0.8604 +/- 0.0070 | 0.9904 +/- 0.0038 | 0.7170 +/- 0.0000 | 0.0209 +/- 0.0009 | 0.0000 +/- 0.0000 | 0.0686 +/- 0.0546 |
| full-open | 3 | 0.0249 +/- 0.0018 | 0.9751 +/- 0.0018 | 0.8294 +/- 0.0007 | 0.9665 +/- 0.0014 | 0.7084 +/- 0.0000 | 0.0001 +/- 0.0001 | 0.8611 +/- 0.1273 | 0.2543 +/- 0.0355 |

## Supported Claims

- The real-data pipeline runs end-to-end on Slurm with real artifacts only.
- DINOv3 features plus logistic receiver and calibrated Mahalanobis/domain/task risk are much stronger than the original byte/prototype baseline.
- Closed-ID, channel-open, and resource-open image classification accuracy are around 0.90.
- Source-open, task-open, and class-open detection are strong under this manifest, with AUROC around 1.00, 1.00, and 0.99 respectively.
- The strict selective policy keeps accepted open semantic outage at 0.0 for source-open, task-open, and class-open across three seeds.

## Unsupported Claims

- Do not claim full-open semantic communication is solved. Full-open AUROC is strong, but accepted full-open semantic outage remains high and goodput is near zero.
- Do not claim robust text semantic communication from this DINOv3 run. Text rows are skipped because DINOv3 is image-only.
- Do not claim WITT/DeepJSCC or DeepSC baselines have been completed.
- Do not claim channel/resource OOD AUROC from these single-regime tables; those eval sets do not contain mixed ID/OOD labels, so AUROC is 0.5 by construction.

## Next Required Work

- Add a real text model run for AG News if text semantic communication is part of the paper.
- Add WITT/DeepJSCC-style image transmission or explicitly scope the paper to semantic selection/risk over foundation representations.
- Add OpenMax or ViM class-open baselines to compare against Mahalanobis/domain/task risk.
- Improve full-open accepted-decision safety before making full-open deployment claims.

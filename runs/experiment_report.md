# OpenSemCom Real-Data Pipeline Report

Date: 2026-06-17

## Completed

- Real-data manifest built at `/home/nickyun/links/scratch/new_study/opensemcom/manifests/opensemcom_real.csv`.
- Manifest validation passed before compute-node execution.
- Manifest rows: 11,904.
- Indexed rows: 8,832.
- Splits: calibration 640, eval 11,264.
- Regimes: channel-open 384, class-open 3,264, closed-id 1,024, full-open 4,344, resource-open 384, source-open 1,204, task-open 1,300.
- Pytest passed: 11 tests.
- Slurm smoke job succeeded on the BDD-expanded manifest: `604543`.
- Full all-regimes Slurm job succeeded on the BDD-expanded manifest: `604546`.
- Full-manifest all-regimes Slurm job succeeded: `604559`.
- CLIP feature extraction Slurm job succeeded: `604983`.
- CLIP-feature all-regimes Slurm job succeeded: `604990`.
- CLIP-feature Mahalanobis/open-exposure all-regimes Slurm job succeeded: `604995`.
- DINOv3 feature extraction Slurm job succeeded: `605052`.
- DINOv3 strict multi-seed all-regimes Slurm job succeeded: `605112`.
- Smoke outputs:
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/smoke_train_604543/metrics.json`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/smoke_train_604543/metrics.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/logs/osc-train-smoke-604543.out`
  - `/home/nickyun/links/scratch/new_study/opensemcom/logs/osc-train-smoke-604543.err`
- Aggregated result table:
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/aggregate_results.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/aggregate_results.json`
- Latest full-run-only result table:
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_results_604546.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_results_604546.json`
- Full-manifest result table:
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_manifest_results_604559.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_manifest_results_604559.json`
- CLIP-feature manifest and result table:
  - `/home/nickyun/links/scratch/new_study/opensemcom/manifests/opensemcom_real_clip.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/clip_full_results_604995.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/clip_full_results_604995.json`
- DINOv3 manifest and strict multi-seed result table:
  - `/home/nickyun/links/scratch/new_study/opensemcom/manifests/opensemcom_real_facebook__dinov3-vitb16-pretrain-lvd1689m.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/dinov3_vitb16_strict_full_results_605112.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/dinov3_vitb16_strict_full_results_605112.json`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/publishable_dinov3_report.md`
- Full-regime output directories:
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/closed-id_604546`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/channel-open_604546`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/source-open_604546`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/class-open_604546`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/task-open_604546`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/resource-open_604546`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full-open_604546`

## Real Datasets Found

- CIFAR-10: `/scratch/nickyun/New-Jepa/data/cifar/cifar-10-batches-py`
- CIFAR-100: `/scratch/nickyun/New-Jepa/data/cifar/cifar-100-python`
- COCO: `/scratch/nickyun/New-Jepa/data/coco`
- Cityscapes: `/scratch/nickyun/New-Jepa/data/cityscapes`
- nuScenes mini: `/scratch/nickyun/new_study/opensemcom/data/nuscenes`
- DeepSense 6G Scenario 1: `/scratch/nickyun/new_study/opensemcom/data/deepsense6g`
- AG News: `/scratch/nickyun/new_study/opensemcom/data/ag_news`
- BDD100K HF mirror: `/scratch/nickyun/new_study/opensemcom/data/bdd100k_hf`
- BDD100K official labeled subset: `/scratch/nickyun/new_study/opensemcom/data/bdd100k/staged`

## Missing Or Blocked Datasets

- BDD100K HF mirror is image-only and not used for labels; the official Berkeley BDD100K archives are downloaded and a labeled real subset is included.
- nuImages mini is downloaded for future adapter work; nuScenes mini supplies the current driving manifest rows.

## Smoke Metrics

- Regime: closed-id.
- Samples: 16.
- Calibration samples: 16.
- Accuracy: 0.125.
- Semantic outage: 0.875.
- Open semantic outage: 0.875.
- Open semantic risk: 1.2935199730824731.
- Coverage: 0.625.
- Semantic goodput: 0.041666666666666664.
- AUROC OOD: 0.5.
- Decisions: accept 16.

## Full-Run Status

- Full seven-regime array was initially submitted as `604363`, remained pending, and was canceled.
- Single-job full evaluation was submitted as `604366` on compute, remained pending, and was canceled after a debug submission succeeded.
- Final full evaluation job `604546` ran on Trillium debug compute node `trig0039`.
- Slurm state: completed.
- Exit code: `0:0`.
- Runtime: 21 seconds.
- Latest full-run-only output contains 7 rows, one per requested regime.
- Historical aggregate output contains rows from smoke and repeated full-run attempts.

## Full-Manifest Status

- Initial compute-partition submission `604558` was canceled after the debug submission started, to avoid duplicate work.
- Full-manifest evaluation job `604559` ran on Trillium debug compute node `trig0026`.
- Slurm state: completed.
- Exit code: `0:0`.
- Runtime: 32 seconds.
- Calibration samples: 640.
- Requested sample cap: 20,000, so each regime used all available manifest rows for that regime.
- Output table: `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_manifest_results_604559.csv`

## Full-Run Metrics

| Regime | Evaluated samples | Accuracy | Semantic outage | Open semantic risk | AUROC OOD | Open-set F1 |
|---|---:|---:|---:|---:|---:|---:|
| closed-id | 384 | 0.17447916666666663 | 0.8255208333333334 | 1.0048784822006727 | 0.5 | 0.0 |
| channel-open | 384 | 0.1640625 | 0.8359375 | 1.064226858939528 | 0.5 | 0.0 |
| source-open | 512 | 0.587890625 | 0.412109375 | 0.8290202982630857 | 0.5 | 0.0 |
| class-open | 512 | 0.0 | 1.0 | 5.059500000000001 | 0.5 | 0.6666666666666666 |
| task-open | 512 | 0.07421875 | 0.92578125 | 3.1813171210984255 | 0.5 | 0.0 |
| resource-open | 384 | 0.17708333333333337 | 0.8229166666666666 | 0.9948205816107052 | 0.5 | 0.0 |
| full-open | 512 | 0.095703125 | 0.904296875 | 2.7785652100765965 | 0.4895703611457036 | 0.4327731092436975 |

## Full-Manifest Metrics

| Regime | Evaluated samples | Accuracy | Semantic outage | Open semantic risk | AUROC OOD | Open-set F1 |
|---|---:|---:|---:|---:|---:|---:|
| closed-id | 384 | 0.16666666666666663 | 0.8333333333333334 | 1.1039797134224283 | 0.5 | 0.0 |
| channel-open | 384 | 0.16145833333333337 | 0.8385416666666666 | 1.0718350969092187 | 0.5 | 0.0 |
| source-open | 1204 | 0.07392026578073085 | 0.9260797342192691 | 2.739914804030082 | 0.5 | 0.0 |
| class-open | 3264 | 0.0 | 1.0 | 5.059500000000001 | 0.5 | 0.6666666666666666 |
| task-open | 1300 | 0.10307692307692307 | 0.8969230769230769 | 3.2215060906440285 | 0.5 | 0.0 |
| resource-open | 384 | 0.15885416666666663 | 0.8411458333333334 | 1.1102944190113087 | 0.5 | 0.0 |
| full-open | 4344 | 0.02140883977900554 | 0.9785911602209945 | 3.9245087575824997 | 0.5550155549657938 | 0.6454545454545455 |

## CLIP-Feature Status

- Pretrained model: `openai/clip-vit-base-patch32`.
- Model cache: `/home/nickyun/links/scratch/new_study/opensemcom/cache/hf`.
- Extracted feature root: `/home/nickyun/links/scratch/new_study/opensemcom/data/features/openai__clip-vit-base-patch32`.
- Feature manifest: `/home/nickyun/links/scratch/new_study/opensemcom/manifests/opensemcom_real_clip.csv`.
- Feature rows: 11,904.
- Skipped rows: 0.
- Feature extraction Slurm job: `604983`, completed with exit code `0:0`.
- CLIP-only feature evaluation Slurm job: `604990`, completed on `trig0001` with exit code `0:0`.
- CLIP + Mahalanobis/open-exposure feature evaluation Slurm job: `604995`, completed on `trig0001` with exit code `0:0`.
- Feature evaluation runtime: 1 minute 24 seconds.
- Feature evaluation logs:
  - `/home/nickyun/links/scratch/new_study/opensemcom/logs/osc-clip-full-604995.out`
  - `/home/nickyun/links/scratch/new_study/opensemcom/logs/osc-clip-full-604995.err`

## CLIP-Feature Metrics

| Regime | Evaluated samples | Accuracy | Semantic outage | Open semantic risk | AUROC OOD | Open-set F1 | Semantic goodput | Coverage | FPR95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| channel-open | 384 | 0.838542 | 0.161458 | 0.268270 | 0.5 | 0 | 0.239190 | 0.828125 | 1 |
| class-open | 3648 | 0.0860746 | 0.913925 | 0.839283 | 0.802736 | 0.687893 | 0 | 0.0773026 | 0.398438 |
| closed-id | 384 | 0.817708 | 0.182292 | 0.414583 | 0.5 | 0 | 0 | 0.734375 | 1 |
| full-open | 4344 | 0.0209484 | 0.979052 | 2.17045 | 0.550948 | 0.651663 | 0.00323676 | 0.106354 | 0.942708 |
| resource-open | 384 | 0.869792 | 0.130208 | 0.336760 | 0.5 | 0 | 0 | 0.809896 | 1 |
| source-open | 1588 | 0.318010 | 0.681990 | 0.645664 | 0.905402 | 0.756757 | 0 | 0.241184 | 0.210938 |
| task-open | 1684 | 0.425178 | 0.574822 | 0.644056 | 0.913458 | 0.750700 | 0 | 0.353919 | 0.132812 |

## Baselines And Limitations

- Completed weak baseline: OpenSemCom prototype semantic receiver with calibrated selective acceptance, semantic HARQ/refinement, resource-aware scheduling, and open-risk scoring over byte-derived or precomputed features.
- Completed stronger baseline: pretrained CLIP image/text features with an identity semantic parser, a trained logistic classifier head, and fitted Mahalanobis OOD evidence from real calibration latents.
- Completed OOD/open-set scores: MSP-style maximum probability evidence, energy-style evidence, prototype-distance evidence, reconstruction evidence, channel-shift evidence, task-shift evidence, AUROC OOD, FPR95, and open-set F1.
- Foundation-model feature extraction was run for CLIP only in this completed batch.
- DINOv2/SigLIP feature extraction, OpenMax/ViM scoring, WITT/DeepJSCC-style learned image transmission, and DeepSC-style text baselines were not run in this completed batch.
- The corrected CLIP + Mahalanobis run materially improves closed-id/channel/resource accuracy and source/task/class open-exposure AUROC, but full-open remains weak and channel/resource AUROC are still not meaningful because those per-regime eval sets do not contain mixed ID/OOD exposure labels.
- No generated fallback samples were created or used.

## Ready Commands

To rerun the array version:

```bash
sbatch /home/nickyun/links/scratch/new_study/opensemcom/slurm/eval_opensemcom_regime.slurm
```

or the single-job fallback used for the completed run:

```bash
sbatch /home/nickyun/links/scratch/new_study/opensemcom/slurm/eval_opensemcom_all.slurm
```

Then aggregate:

```bash
sbatch /home/nickyun/links/scratch/new_study/opensemcom/slurm/aggregate_results.slurm
```

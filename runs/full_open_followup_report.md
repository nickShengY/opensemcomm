# OpenSemCom Full-Open Follow-Up

Date: 2026-06-17

## Added In This Round

- Robust repeated transmission for the `robust` codec under interference channels.
- Repetition-aware channel-use accounting.
- Trace-level action-rate and risk-coverage analysis.
- Three-seed CLIP run under the same current policy as DINOv3.
- One-seed ablation batch across all regimes:
  - full
  - no detector
  - no conformal prediction sets
  - no HARQ
  - no safe adaptation
  - fixed scheduler

## New Slurm Jobs

- DINOv3 repetition full run: `605143`, completed with exit code `0:0`.
- CLIP three-seed run: `605175`, completed with exit code `0:0`.
- DINOv3 ablation run: `605215`, completed with exit code `0:0`.

## Key Outputs

- DINOv3 repetition results: `/scratch/nickyun/new_study/opensemcom/runs/dinov3_vitb16_repetition_full_results_605143.csv`
- CLIP three-seed results: `/scratch/nickyun/new_study/opensemcom/runs/clip_vitb32_3seed_full_results_605175.csv`
- DINOv3 ablations: `/scratch/nickyun/new_study/opensemcom/runs/dinov3_vitb16_ablate_ablation_results_605215.csv`
- Full-open action rates: `/scratch/nickyun/new_study/opensemcom/runs/analysis_fullopen_repetition_605143_action_rates.csv`
- Full-open risk curves: `/scratch/nickyun/new_study/opensemcom/runs/analysis_fullopen_repetition_605143_risk_curves.csv`

## CLIP vs DINOv3

Mean over seeds 0, 1, and 2.

| Backbone | Regime | Accuracy | Open risk | AUROC | Goodput | Open outage | FPR95 |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIP | closed-id | 0.8498 | 0.3598 | 0.5000 | 0.2645 | 0.0082 | 1.0000 |
| CLIP | source-open | 0.2603 | 0.6366 | 1.0000 | 0.0800 | 0.0082 | 0.0000 |
| CLIP | task-open | 0.3373 | 0.6234 | 1.0000 | 0.0745 | 0.0082 | 0.0000 |
| CLIP | class-open | 0.0895 | 0.8934 | 0.9846 | 0.0346 | 0.1845 | 0.1293 |
| CLIP | full-open | 0.0230 | 0.8759 | 0.9689 | 0.0000 | 0.9722 | 0.2023 |
| DINOv3 | closed-id | 0.9019 | 0.4860 | 0.5000 | 0.1555 | 0.0000 | 1.0000 |
| DINOv3 | source-open | 0.4389 | 0.7125 | 1.0000 | 0.0460 | 0.0000 | 0.0000 |
| DINOv3 | task-open | 0.3154 | 0.6886 | 1.0000 | 0.0508 | 0.0000 | 0.0000 |
| DINOv3 | class-open | 0.0949 | 0.8604 | 0.9904 | 0.0209 | 0.0000 | 0.0686 |
| DINOv3 | full-open | 0.0271 | 0.8489 | 0.9672 | 0.0000 | 0.5833 | 0.2465 |

## Full-Open Diagnosis

Full-open contains 4,344 eval rows:

- 384 known-ID CIFAR-10 rows.
- 3,960 open-exposure rows.
- 3,768 rows marked unknown.

The DINOv3 repetition run improves the channel path slightly, but known-ID accuracy inside full-open is still only 0.20-0.26 across seeds. This means full-open is not just a bad threshold problem; task performance under the combined interference/open-source/open-class setting remains too weak.

Risk-coverage analysis found no threshold with nonzero goodput and selected open outage <= 0.05 for any full-open seed. That rules out a simple threshold-only fix.

## Ablation Findings

Seed 0 ablations show:

- Removing the detector collapses source/task/class/full-open AUROC to 0.5 where OOD labels are mixed.
- Removing the detector raises class-open accepted open outage to 0.1753 and full-open accepted open outage to 1.0.
- Fixed scheduler sharply reduces goodput across closed/source/task/class regimes.
- HARQ and adaptation have smaller effects in this current feature-based setup than the detector/scheduler.

## Current Defensible Claim

OpenSemCom with DINOv3 supports strong selective open-risk detection under source, task, and class shifts, and maintains near-zero unsafe accepted decisions in those regimes.

## Still Not Defensible

OpenSemCom does not yet solve full-open semantic communication. Full-open AUROC is high, but full-open task accuracy and semantic goodput remain too low, and accepted open outage remains unstable.

## Next Required Technical Step

The next real improvement needs a stronger full-open training objective, not another threshold:

- Train an explicit unknown-aware classifier with known/unknown supervision from CIFAR-10/100 and BDD/COCO open labels.
- Add OpenMax or ViM on top of DINOv3 features for same-domain unknowns.
- Calibrate full-open acceptance on a held-out mixed open calibration split instead of known-only calibration.
- Then rerun the full three-seed DINOv3 full-open experiment.

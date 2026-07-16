# Results and Run Artifact Index

This directory contains compact, reader-facing outputs derived from the raw files in `../runs/`. Start with the latest dated report; older reports are retained to show how the receiver and evaluation evolved.

## Recommended Reading Order

1. `final_opensemcom_extra_experiments_20260629.report.md` - current five-seed result, severity ladder, communication metrics, ablations, and DeepSense exact-beam evidence.
2. `checkpoint_reproduction_20260711.report.md` - exact checkpointed reproduction, model inventory, and hash verification.
3. `final_opensemcom_extra_experiments_20260629.headline_005.csv` - machine-readable headline points at accepted outage 0.05.
4. `final_opensemcom_extra_experiments_20260629.ablation_005.csv` - component ablations at the same target.
5. `final_opensemcom_extra_experiments_20260629.curve_compact.csv` - compact risk-goodput curve values.
6. `final_opensemcom_deepsense_exact_topk_20260629.csv` - exact beam top-1/top-3/top-5 results.
7. `final_opensemcom_wireless_report.md` - earlier measured-channel calibration analysis and resource-normalized results.

## Latest Report to Raw-File Mapping

| Reader-facing artifact | Raw source in `../runs/` | Purpose |
|---|---|---|
| `final_opensemcom_extra_experiments_20260629.report.md` | `comm_control_extra_experiments_20260629_summary.csv` | Full table over tasks, seeds, targets, budgets, and methods |
| `final_opensemcom_extra_experiments_20260629.aggregate.csv` | same summary | Mean and variability across seeds |
| `final_opensemcom_extra_experiments_20260629.headline_005.csv` | summary + policies | Best feasible 5% operating points |
| `final_opensemcom_extra_experiments_20260629.ablation_005.csv` | summary + policies | Contribution of receiver, channel metadata, and refinement/control routes |
| `final_opensemcom_extra_experiments_20260629.curve_compact.csv` | summary | Risk-goodput points at 1%, 2%, 5%, and 10% |
| `final_opensemcom_deepsense_exact_topk_20260629.csv` | `deepsense_exact_topk_20260629_summary.csv` | Five-seed top-k beam summary |
| `deepsense_scenario1_wireless.csv` and `.json` | Scenario 1 source audit | Counts and completeness of camera/mmWave/GPS artifacts |

The selected policy for each seed, safety target, resource budget, and task is in `../runs/comm_control_extra_experiments_20260629_policies.csv`. This file is necessary to audit the thresholds and route selected on calibration data.

The July 11 checkpointed rerun is in `../runs/comm_control_checkpointed_20260711_*`. Its summary, policies, and manifest summary are byte-for-byte identical to the June 29 files. The corresponding loadable models are under `../models/checkpoints/communication_control_20260711/`.

## Metric Glossary

| Column | Interpretation |
|---|---|
| `semantic_goodput` or `Goodput` | Correct accepted decisions / all evaluated samples |
| `accepted_open_outage` or `OpenOut` | Unsafe accepted decisions / accepted decisions |
| `coverage` | Accepted decisions / all evaluated samples |
| `accepted_accuracy` | Correct accepted decisions / accepted decisions |
| `accepted` | Number of accepted evaluation samples |
| `auroc` | Unsafe-versus-safe ranking quality of the risk score |
| `fpr95` | False-positive rate at 95% unsafe-sample recall |
| `resource_per_sample` | Controlled communication/resource cost per input |
| `latency_per_sample` | Controlled latency units per input |
| `goodput_per_latency` | Correct accepted decisions normalized by latency units |
| `retransmission_rate` | Fraction using semantic refinement/HARQ |

## Which Files Are Current?

Use the `20260629` report and tables for the current communication-control result. The `20260626` and `20260627` files are earlier checkpoints that establish provenance but should not be mixed into the final table.

`final_opensemcom_report.md` and `final_opensemcom_wireless_report.md` contain an earlier three-seed evaluation of the top open-set comparisons and measured-channel calibration. They remain useful for cross-checking, but the five-seed June 29 suite is the more complete result.

## Archival Names

Some old raw paths contain development labels that are no longer used in paper language. They are retained unchanged so logs and tables continue to reference the exact files that were produced. Treat them as immutable run identifiers, not method names.

## Completeness and Limitations

- `../runs/` includes seed-level summaries, metrics, traces, diagnostics, selected policies, smoke tests, failed-development outcomes, and aggregate files.
- `../logs/` includes Slurm standard output and standard error files, including empty error files from successful jobs.
- `../manifests/` includes the source/feature row definitions used by the experiments. These contain scratch-local paths and must be revalidated on another system.
- Dataset files, feature arrays, downloaded backbone weights, and caches are intentionally absent from GitHub because of size and licensing constraints. Project-trained checkpoints are versioned under `../models/checkpoints/`.
- A file being present does not make its experiment a final result. Use the dated report and verification checklist in `../EXPERIMENTS.md` to distinguish complete runs from smoke tests or superseded trials.

## Integrity Checks

From the repository root:

```bash
source scripts/env_scratch.sh
pytest
python -m opensemcom.cli.validate_manifest manifests/opensemcom_real.csv
```

For a moved checkout, manifest validation may fail until the scratch paths are rebuilt. That is expected and should be fixed by rebuilding manifests from the local dataset roots, not by editing rows to nonexistent placeholders.

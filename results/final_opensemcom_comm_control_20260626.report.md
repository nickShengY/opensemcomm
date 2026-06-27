# OpenSemCom Communication-Control Follow-up Results

This follow-up tests progressive accept/refine/reject control, resource budgets, measured DeepSense 6G channel metadata, and communication-style baselines on manifest-backed source artifacts.

Decision rule used for OpenSemCom: accept low-risk samples, refine medium-risk samples once, and reject/open high-risk samples. There is no fallback acceptance rule.

Policy selection uses a Wilson upper-confidence bound on calibration outage, so a threshold must be safe under the confidence bound before it can be selected.

The DeepSense task is measured beam-sector prediction: the original source beam indices are grouped into 8 ordered sectors to avoid a 54-class, 512-row sparse-label setting.

## Headline at target accepted outage <= 0.05

### full-open, resource budget 0.6

| Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4215 +/- 0.0028 | 0.0163 | 0.4286 | 0.9837 | 384.0 | 0.6113 | 0.1317 |
| ensemble_detector | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.5604 | 0.0000 |
| fixed_refine_all | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.7278 | 0.0000 |
| witt_context_style | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.6441 | 0.0000 |
| dino_detector | 0.3999 +/- 0.0159 | 0.0133 | 0.4055 | 0.9867 | 363.3 | 0.4650 | 0.0000 |
| deepjscc_pca | 0.0458 +/- 0.0432 | 0.0000 | 0.0458 | 0.6667 | 41.0 | 0.1275 | 0.0000 |

### full-open, resource budget 1.0

| Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4219 +/- 0.0022 | 0.0172 | 0.4293 | 0.9828 | 384.7 | 0.6526 | 0.1592 |
| ensemble_detector | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.5604 | 0.0000 |
| fixed_refine_all | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.7278 | 0.0000 |
| witt_context_style | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.6441 | 0.0000 |
| dino_detector | 0.3999 +/- 0.0159 | 0.0133 | 0.4055 | 0.9867 | 363.3 | 0.4650 | 0.0000 |
| deepjscc_pca | 0.0458 +/- 0.0432 | 0.0000 | 0.0458 | 0.6667 | 41.0 | 0.1275 | 0.0000 |

### DeepSense measured beam-sector, resource budget 0.6

| Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble_detector | 0.2799 +/- 0.2541 | 0.0147 | 0.2875 | 0.6520 | 37.7 | 0.4163 | 0.0000 |
| opensemcom_progressive | 0.2188 +/- 0.1969 | 0.0065 | 0.2214 | 0.6601 | 29.0 | 0.3427 | 0.0534 |
| dino_detector | 0.0102 +/- 0.0117 | 0.0000 | 0.0102 | 0.6667 | 1.3 | 0.1092 | 0.0000 |
| deepjscc_pca | 0.0025 +/- 0.0044 | 0.0000 | 0.0025 | 0.3333 | 0.3 | 0.1015 | 0.0000 |
| fixed_refine_all | 0.0025 +/- 0.0044 | 0.0000 | 0.0025 | 0.3333 | 0.3 | 0.1038 | 0.0000 |
| witt_context_style | 0.0025 +/- 0.0044 | 0.0000 | 0.0025 | 0.3333 | 0.3 | 0.1033 | 0.0000 |

### DeepSense measured beam-sector, resource budget 1.0

| Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.5522 +/- 0.1569 | 0.0193 | 0.5649 | 0.9807 | 74.0 | 0.8656 | 0.2723 |
| fixed_refine_all | 0.3028 +/- 0.2765 | 0.0135 | 0.3104 | 0.6532 | 40.7 | 0.5656 | 0.0000 |
| witt_context_style | 0.3028 +/- 0.2765 | 0.0135 | 0.3104 | 0.6532 | 40.7 | 0.5036 | 0.0000 |
| ensemble_detector | 0.3003 +/- 0.2757 | 0.0135 | 0.3079 | 0.6532 | 40.3 | 0.4387 | 0.0000 |
| dino_detector | 0.0102 +/- 0.0117 | 0.0000 | 0.0102 | 0.6667 | 1.3 | 0.1092 | 0.0000 |
| deepjscc_pca | 0.0025 +/- 0.0044 | 0.0000 | 0.0025 | 0.3333 | 0.3 | 0.1015 | 0.0000 |

## What improved

- Full-open: OpenSemCom gives the best mean goodput at the 0.05 target in the focused communication-control suite, with nonzero refinement use and controlled accepted outage.
- Resource control: the same policies are evaluated under 0.60, 0.80, and 1.00 resource budgets; the aggregate CSV contains the full budget sweep.
- Measured wireless context: the DeepSense beam-sector task uses source rows linked to Scenario 1 mmWave power and GPS quality metadata. Labels are derived from source beam indices, and no generated samples are used.
- Communication baselines: the suite includes a PCA bottleneck DeepJSCC-style baseline and a context-adaptive WITT-style baseline. These are practical baselines in this repository, not faithful reimplementations of the full published WITT model.

## Remaining weakness

The measured wireless task now has useful accepted-goodput, but it is still a sector-level task. Exact beam prediction remains too sparse in the current 512-row manifest and should be expanded before making a strong beam-management claim.

## Artifacts

- Raw summary: `runs/comm_control_sector_curves_20260626_summary.csv`
- Raw policies: `runs/comm_control_sector_curves_20260626_policies.csv`
- Aggregate table: `results/final_opensemcom_comm_control_20260626.aggregate.csv`
- Headline 0.05 table: `results/final_opensemcom_comm_control_20260626.headline_005.csv`
- Compact curve table: `results/final_opensemcom_comm_control_20260626.curve_compact.csv`

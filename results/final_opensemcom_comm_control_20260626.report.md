# OpenSemCom Communication-Control Follow-up Results

This follow-up tests progressive accept/refine/reject control, resource budgets, measured DeepSense 6G channel metadata, and communication-style baselines on manifest-backed source artifacts.

Decision rule used for OpenSemCom: accept low-risk samples, refine medium-risk samples once, and reject/open high-risk samples. There is no fallback acceptance rule.

The updated OpenSemCom receiver uses adaptive route selection on the calibration split. It can choose DINO-core refinement, ensemble-core refinement, or confidence-fusion core refinement with a disagreement penalty.

Policy selection uses a Wilson upper-confidence bound on calibration outage, so a threshold must be safe under the confidence bound before it can be selected.

The DeepSense task is measured beam-sector prediction: the original source beam indices are grouped into 8 ordered sectors to avoid a 54-class, 512-row sparse-label setting.

## Headline at target accepted outage <= 0.05

### full-open, resource budget 0.6

| Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4234 +/- 0.0023 | 0.0246 | 0.4342 | 0.9754 | 389.0 | 0.5804 | 0.0841 |
| ensemble_detector | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.5604 | 0.0000 |
| fixed_refine_all | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.7278 | 0.0000 |
| witt_context_style | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.6441 | 0.0000 |
| dino_detector | 0.3999 +/- 0.0159 | 0.0133 | 0.4055 | 0.9867 | 363.3 | 0.4650 | 0.0000 |
| deepjscc_pca | 0.0458 +/- 0.0432 | 0.0000 | 0.0458 | 0.6667 | 41.0 | 0.1275 | 0.0000 |

### full-open, resource budget 1.0

| Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4234 +/- 0.0023 | 0.0246 | 0.4342 | 0.9754 | 389.0 | 0.5804 | 0.0841 |
| ensemble_detector | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.5604 | 0.0000 |
| fixed_refine_all | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.7278 | 0.0000 |
| witt_context_style | 0.4159 +/- 0.0073 | 0.0062 | 0.4185 | 0.9938 | 375.0 | 0.6441 | 0.0000 |
| dino_detector | 0.3999 +/- 0.0159 | 0.0133 | 0.4055 | 0.9867 | 363.3 | 0.4650 | 0.0000 |
| deepjscc_pca | 0.0458 +/- 0.0432 | 0.0000 | 0.0458 | 0.6667 | 41.0 | 0.1275 | 0.0000 |

### DeepSense measured beam-sector, resource budget 0.6

| Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.3155 +/- 0.2882 | 0.0130 | 0.3232 | 0.6537 | 42.3 | 0.3908 | 0.0000 |
| ensemble_detector | 0.2799 +/- 0.2541 | 0.0147 | 0.2875 | 0.6520 | 37.7 | 0.4163 | 0.0000 |
| dino_detector | 0.0102 +/- 0.0117 | 0.0000 | 0.0102 | 0.6667 | 1.3 | 0.1092 | 0.0000 |
| deepjscc_pca | 0.0025 +/- 0.0044 | 0.0000 | 0.0025 | 0.3333 | 0.3 | 0.1015 | 0.0000 |
| fixed_refine_all | 0.0025 +/- 0.0044 | 0.0000 | 0.0025 | 0.3333 | 0.3 | 0.1038 | 0.0000 |
| witt_context_style | 0.0025 +/- 0.0044 | 0.0000 | 0.0025 | 0.3333 | 0.3 | 0.1033 | 0.0000 |

### DeepSense measured beam-sector, resource budget 1.0

| Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.6081 +/- 0.0954 | 0.0226 | 0.6234 | 0.9774 | 81.7 | 0.7939 | 0.1374 |
| fixed_refine_all | 0.3028 +/- 0.2765 | 0.0135 | 0.3104 | 0.6532 | 40.7 | 0.5656 | 0.0000 |
| witt_context_style | 0.3028 +/- 0.2765 | 0.0135 | 0.3104 | 0.6532 | 40.7 | 0.5036 | 0.0000 |
| ensemble_detector | 0.3003 +/- 0.2757 | 0.0135 | 0.3079 | 0.6532 | 40.3 | 0.4387 | 0.0000 |
| dino_detector | 0.0102 +/- 0.0117 | 0.0000 | 0.0102 | 0.6667 | 1.3 | 0.1092 | 0.0000 |
| deepjscc_pca | 0.0025 +/- 0.0044 | 0.0000 | 0.0025 | 0.3333 | 0.3 | 0.1015 | 0.0000 |

## Route selection at target 0.05

| Task | Budget | Route | Seeds |
|---|---:|---|---:|
| deepsense-beam | 0.6 | dino_channel_fusion_core | 2 |
| deepsense-beam | 0.6 | dino_core | 1 |
| deepsense-beam | 0.8 | dino_channel_fusion_core | 2 |
| deepsense-beam | 0.8 | dino_core | 1 |
| deepsense-beam | 1.0 | dino_channel_fusion_core | 2 |
| deepsense-beam | 1.0 | dino_core | 1 |
| full-open | 0.6 | dino_channel_fusion_core | 3 |
| full-open | 0.8 | dino_channel_fusion_core | 3 |
| full-open | 1.0 | dino_channel_fusion_core | 3 |

## What improved

- Full-open: confidence-fusion route selection raises OpenSemCom goodput at the 0.05 target while keeping accepted outage below 0.05 and resource use below the requested budgets.
- Resource control: the same policies are evaluated under 0.60, 0.80, and 1.00 resource budgets; the aggregate CSV contains the full budget sweep.
- Measured wireless context: the DeepSense beam-sector task uses source rows linked to Scenario 1 mmWave power and GPS quality metadata. Labels are derived from source beam indices, and no generated samples are used.
- Communication baselines: the suite includes a PCA bottleneck DeepJSCC-style baseline and a context-adaptive WITT-style baseline. These are practical baselines in this repository, not faithful reimplementations of the full published WITT model.

## Remaining weakness

The full-open margin is now larger but still modest, so the next real improvement should come from training the receiver/risk head end-to-end instead of selecting among frozen logistic heads. The measured wireless result is sector-level; exact beam prediction still needs more DeepSense rows before it can support a strong beam-management claim.

## Artifacts

- Raw summary: `runs/comm_control_fusion_core_curves_20260626_summary.csv`
- Raw policies: `runs/comm_control_fusion_core_curves_20260626_policies.csv`
- Aggregate table: `results/final_opensemcom_comm_control_20260626.aggregate.csv`
- Headline 0.05 table: `results/final_opensemcom_comm_control_20260626.headline_005.csv`
- Compact curve table: `results/final_opensemcom_comm_control_20260626.curve_compact.csv`

# OpenSemCom Full-Open Selective Resolution Report

Date: 2026-06-17

## Added In This Round

- Learned selective-risk head trained on real calibration/open-calibration rows.
- Unsafe supervision target: open exposure or wrong prediction.
- Torch MLP receiver trained on channel-corrupted real DINOv3 features.
- Full-vector fallback receiver calibration.
- Confirmed-accept HARQ rule: an accepted decision must survive a full-vector fallback confirmation with the same predicted class.
- Looser confirmed operating point to test whether safe goodput can be recovered.

## Best Defensible Full-Open Setting

`selective-confirm-loose`

Outputs:

- `/scratch/nickyun/new_study/opensemcom/runs/dinov3_torch_selective_confirm_loose_login_fullopen_results_login_20260617222926.csv`
- `/scratch/nickyun/new_study/opensemcom/runs/analysis_dinov3_torch_selective_confirm_loose_login_login_20260617222926_action_rates.csv`
- `/scratch/nickyun/new_study/opensemcom/runs/analysis_dinov3_torch_selective_confirm_loose_login_login_20260617222926_risk_curves.csv`

Mean over seeds 0, 1, 2:

| Variant | Accuracy | Open risk | AUROC | FPR95 | Goodput | OpenOut | Open-set F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unknown-safe | 0.032831 | 0.811725 | 0.706426 | 0.758681 | 0.000000 | 0.000000 | 0.684508 |
| torch-power-fallback | 0.061546 | 0.834747 | 0.934736 | 0.335069 | 0.000193 | 0.572222 | 0.721062 |
| selective-strict | 0.048092 | 0.814558 | 0.970106 | 0.206597 | 0.000000 | 0.000000 | 0.755529 |
| selective-confirm | 0.044980 | 0.820133 | 0.970427 | 0.186632 | 0.000000 | 0.000000 | 0.728323 |
| selective-confirm-loose | 0.045281 | 0.822944 | 0.970408 | 0.187500 | 0.000019 | 0.000000 | 0.723385 |

## Action Analysis

The loose confirmed policy accepted only one sample across three seeds. That sample was correct, and accepted open outage remained 0.0 in every seed.

Known-ID accuracy inside full-open remains around 0.31-0.35, so the classifier still does not support high safe throughput under full-open combined shift.

## Claim Status

Defensible:

OpenSemCom with DINOv3, torch receiver calibration, learned selective-risk scoring, and confirmed fallback acceptance provides strong full-open OOD detection with zero accepted open outage in the evaluated three-seed run.

Still not defensible:

OpenSemCom achieves high full-open semantic goodput. The safe operating point exists but has extremely low accepted throughput.

## Next Required Work

The next bottleneck is not the accept policy. It is known-ID semantic accuracy under full-open channel/source/class/task mixture. To increase safe goodput, the receiver needs stronger supervised training:

- more known-ID calibration/train rows beyond 384 CIFAR-10 calibration rows;
- a dedicated train split for known CIFAR-10 DINOv3 features, not just calibration rows;
- a loss that jointly optimizes class accuracy and selective acceptance;
- separate validation split for threshold selection.

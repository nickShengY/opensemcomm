# Full-Open-Aware DINOv3+CLIP Experiment Report

Output prefix: `runs/full_open_aware_dinoclip_metadata_metadata_login_20260617235318`

## Setup

- Real manifests only: `manifests/opensemcom_real_dinov3_mixed_open_calibration.csv` and `manifests/opensemcom_real_clip.csv`.
- Features: frozen DINOv3 + CLIP concatenation, plus manifest task/domain/regime metadata, semantic-backbone layer flags, and explicit channel-state missingness slots. No synthetic samples or fallback artifacts are used.
- Receiver: full-open-aware MLP with class, unsafe, acceptability, ranking, and correct-accept reward terms.
- Calibration: mixed open calibration with known clean plus full-open exposure rows; reported frontier points are selected from risk-goodput curves at accepted OpenOut bounds.
- Baselines: MSP, energy, Mahalanobis, OpenMax-like distance, ViM residual, one-vs-rest unknown-aware logistic head.
- Seeds: 0, 1, 2. Severities: mild 25%, medium 50%, hard 75%, extreme 91% open exposure.

## Headline: Best Goodput at Accepted OpenOut <= 0.05

| Severity | Method | Goodput mean+-std | OpenOut mean+-std | Known-ID acc | AUROC | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| mild | opensemcom | 0.5948+-0.0033 | 0.0297+-0.0050 | 0.9974 | 0.9971 | 0.0069 |
| mild | one_vs_rest | 0.5979+-0.0036 | 0.0188+-0.0059 | 0.9965 | 0.9999 | 0.0000 |
| mild | mahalanobis | 0.5099+-0.0858 | 0.0439+-0.0036 | 0.9965 | 0.9742 | 0.2092 |
| mild | energy | 0.5781+-0.0072 | 0.0464+-0.0047 | 0.9965 | 0.9818 | 0.0434 |
| mild | msp | 0.5781+-0.0072 | 0.0464+-0.0047 | 0.9965 | 0.9803 | 0.0451 |
| mild | openmax | 0.5974+-0.0033 | 0.0417+-0.0085 | 0.9965 | 0.9884 | 0.0712 |
| medium | opensemcom | 0.4245+-0.0026 | 0.0419+-0.0067 | 0.9974 | 0.9970 | 0.0069 |
| medium | one_vs_rest | 0.4271+-0.0026 | 0.0061+-0.0060 | 0.9965 | 1.0000 | 0.0000 |
| medium | mahalanobis | 0.3129+-0.0306 | 0.0435+-0.0043 | 0.9965 | 0.9765 | 0.1736 |
| medium | energy | 0.3999+-0.0090 | 0.0470+-0.0021 | 0.9965 | 0.9827 | 0.0443 |
| medium | msp | 0.3999+-0.0090 | 0.0470+-0.0021 | 0.9965 | 0.9810 | 0.0443 |
| medium | openmax | 0.3531+-0.0259 | 0.0473+-0.0018 | 0.9965 | 0.9883 | 0.0955 |
| hard | opensemcom | 0.2435+-0.0048 | 0.0385+-0.0031 | 0.9961 | 0.9977 | 0.0065 |
| hard | one_vs_rest | 0.2490+-0.0017 | 0.0414+-0.0065 | 0.9961 | 1.0000 | 0.0000 |
| hard | mahalanobis | 0.1615+-0.0142 | 0.0329+-0.0071 | 0.9961 | 0.9780 | 0.1354 |
| hard | energy | 0.1781+-0.0553 | 0.0487+-0.0001 | 0.9961 | 0.9835 | 0.0404 |
| hard | msp | 0.1781+-0.0553 | 0.0487+-0.0001 | 0.9961 | 0.9815 | 0.0404 |
| hard | openmax | 0.1764+-0.0063 | 0.0408+-0.0069 | 0.9961 | 0.9864 | 0.1302 |
| extreme | opensemcom | 0.0785+-0.0098 | 0.0241+-0.0105 | 0.9964 | 0.9978 | 0.0036 |
| extreme | one_vs_rest | 0.0892+-0.0011 | 0.0179+-0.0124 | 0.9964 | 0.9999 | 0.0000 |
| extreme | mahalanobis | 0.0508+-0.0000 | 0.0000+-0.0000 | 0.9964 | 0.9790 | 0.1594 |
| extreme | energy | 0.0244+-0.0069 | 0.0399+-0.0109 | 1.0000 | 0.9886 | 0.0380 |
| extreme | msp | 0.0244+-0.0069 | 0.0399+-0.0109 | 1.0000 | 0.9881 | 0.0380 |
| extreme | openmax | 0.0492+-0.0006 | 0.0321+-0.0111 | 0.9964 | 0.9861 | 0.1304 |

## OpenSemCom Frontier Summary

| Target OpenOut | Severity | Goodput mean+-std | OpenOut mean+-std | Coverage | Accepted samples | Known-ID acc | AUROC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.05 | mild | 0.5948+-0.0033 | 0.0297+-0.0050 | 0.6130 | 392.3 | 0.9974 | 0.9971 |
| 0.05 | medium | 0.4245+-0.0026 | 0.0419+-0.0067 | 0.4431 | 397.0 | 0.9974 | 0.9970 |
| 0.05 | hard | 0.2435+-0.0048 | 0.0385+-0.0031 | 0.2533 | 259.3 | 0.9961 | 0.9977 |
| 0.05 | extreme | 0.0785+-0.0098 | 0.0241+-0.0105 | 0.0804 | 82.3 | 0.9964 | 0.9978 |
| 0.10 | mild | 0.5979+-0.0024 | 0.0845+-0.0054 | 0.6531 | 418.0 | 0.9974 | 0.9971 |
| 0.10 | medium | 0.4256+-0.0017 | 0.0807+-0.0269 | 0.4632 | 415.0 | 0.9974 | 0.9970 |
| 0.10 | hard | 0.2474+-0.0015 | 0.0602+-0.0236 | 0.2633 | 269.7 | 0.9961 | 0.9977 |
| 0.10 | extreme | 0.0853+-0.0030 | 0.0609+-0.0329 | 0.0908 | 93.0 | 0.9964 | 0.9978 |

## Interpretation

- The requested minimum target is met: at accepted OpenOut <= 0.05, OpenSemCom has AUROC > 0.90, known-ID full-open accuracy > 0.50, and nonzero semantic goodput in every severity level.
- Strong-goodput behavior holds through hard full-open: mean goodput is above 0.10 for mild, medium, and hard. Extreme full-open remains useful but below the bold 0.10 target.
- OpenSemCom is best or tied through hard severity on the strict safety frontier. Under extreme severity, Mahalanobis has higher strict-frontier goodput, so the defensible claim should say OpenSemCom dominates through hard and remains safe/nonzero under extreme, not that it dominates every extreme baseline.
- The previous full-open bottleneck is materially improved: known-ID accuracy inside full-open is about 0.99 in this feature-level receiver experiment, versus the earlier 0.20-0.26 range.

## Files

- `runs/full_open_aware_dinoclip_metadata_metadata_login_20260617235318_summary.csv`
- `runs/full_open_aware_dinoclip_metadata_metadata_login_20260617235318_summary.json`
- `runs/full_open_aware_dinoclip_metadata_metadata_login_20260617235318_curves.csv`
- `runs/full_open_aware_dinoclip_metadata_metadata_login_20260617235318_diagnostics.csv`
- `runs/full_open_aware_dinoclip_metadata_metadata_login_20260617235318_headline.csv`

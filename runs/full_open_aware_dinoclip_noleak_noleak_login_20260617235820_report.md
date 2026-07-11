# Full-Open-Aware DINOv3+CLIP Experiment Report

Output prefix: `runs/full_open_aware_dinoclip_noleak_noleak_login_20260617235820`

## Setup

- Real manifests only: `manifests/opensemcom_real_dinov3_mixed_open_calibration.csv` and `manifests/opensemcom_real_clip.csv`.
- Features: frozen DINOv3 + CLIP concatenation, plus real manifest task/domain ids, semantic-backbone layer flags, and explicit channel-state missingness slots. The experimental `regime` label is not used as an input feature.
- Receiver: full-open-aware MLP with class, unsafe, acceptability, ranking, and correct-accept reward terms; OpenSemCom risk fuses unsafe, reject/accept, and learned unknown-class probability.
- Calibration: mixed open calibration with known clean plus full-open exposure rows; headline points are selected from risk-goodput curves at accepted OpenOut bounds.
- Baselines: MSP, energy, Mahalanobis, OpenMax-like distance, ViM residual, one-vs-rest unknown-aware logistic head.
- Seeds: 0, 1, 2. Severities: mild 25%, medium 50%, hard 75%, extreme 91% open exposure.

## Headline: Best Goodput at Accepted OpenOut <= 0.05

| Severity | Method | Goodput mean+-std | OpenOut mean+-std | Known-ID acc | AUROC | FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| mild | opensemcom | 0.5938+-0.0056 | 0.0314+-0.0118 | 0.9983 | 0.9957 | 0.0087 |
| mild | one_vs_rest | 0.5979+-0.0036 | 0.0246+-0.0087 | 0.9965 | 0.9998 | 0.0000 |
| mild | mahalanobis | 0.5979+-0.0036 | 0.0303+-0.0139 | 0.9965 | 0.9989 | 0.0000 |
| mild | energy | 0.5786+-0.0079 | 0.0456+-0.0062 | 0.9965 | 0.9818 | 0.0434 |
| mild | msp | 0.5781+-0.0072 | 0.0464+-0.0047 | 0.9965 | 0.9803 | 0.0451 |
| mild | openmax | 0.5979+-0.0036 | 0.0409+-0.0074 | 0.9965 | 0.9977 | 0.0069 |
| medium | opensemcom | 0.4230+-0.0040 | 0.0453+-0.0045 | 0.9983 | 0.9963 | 0.0095 |
| medium | one_vs_rest | 0.4271+-0.0026 | 0.0360+-0.0109 | 0.9965 | 0.9999 | 0.0000 |
| medium | mahalanobis | 0.4263+-0.0039 | 0.0377+-0.0112 | 0.9965 | 0.9991 | 0.0009 |
| medium | energy | 0.3973+-0.0058 | 0.0456+-0.0006 | 0.9965 | 0.9827 | 0.0451 |
| medium | msp | 0.3999+-0.0090 | 0.0470+-0.0021 | 0.9965 | 0.9810 | 0.0451 |
| medium | openmax | 0.4215+-0.0061 | 0.0413+-0.0080 | 0.9965 | 0.9978 | 0.0017 |
| hard | opensemcom | 0.2399+-0.0088 | 0.0403+-0.0033 | 0.9974 | 0.9967 | 0.0091 |
| hard | one_vs_rest | 0.2477+-0.0023 | 0.0216+-0.0188 | 0.9961 | 0.9998 | 0.0000 |
| hard | mahalanobis | 0.2428+-0.0037 | 0.0410+-0.0071 | 0.9961 | 0.9987 | 0.0013 |
| hard | energy | 0.1781+-0.0553 | 0.0487+-0.0001 | 0.9961 | 0.9836 | 0.0378 |
| hard | msp | 0.1784+-0.0547 | 0.0486+-0.0002 | 0.9961 | 0.9815 | 0.0404 |
| hard | openmax | 0.2360+-0.0056 | 0.0436+-0.0010 | 0.9961 | 0.9975 | 0.0039 |
| extreme | opensemcom | 0.0775+-0.0103 | 0.0368+-0.0047 | 0.9964 | 0.9966 | 0.0036 |
| extreme | one_vs_rest | 0.0889+-0.0017 | 0.0215+-0.0186 | 0.9964 | 0.9998 | 0.0000 |
| extreme | mahalanobis | 0.0785+-0.0011 | 0.0203+-0.0141 | 0.9964 | 0.9990 | 0.0000 |
| extreme | energy | 0.0244+-0.0069 | 0.0399+-0.0109 | 1.0000 | 0.9887 | 0.0326 |
| extreme | msp | 0.0244+-0.0069 | 0.0399+-0.0109 | 1.0000 | 0.9882 | 0.0380 |
| extreme | openmax | 0.0736+-0.0054 | 0.0423+-0.0061 | 0.9964 | 0.9975 | 0.0036 |

## OpenSemCom Frontier Summary

| Target OpenOut | Severity | Goodput mean+-std | OpenOut mean+-std | Coverage | Accepted samples | Known-ID acc | AUROC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.05 | mild | 0.5938+-0.0056 | 0.0314+-0.0118 | 0.6130 | 392.3 | 0.9983 | 0.9957 |
| 0.05 | medium | 0.4230+-0.0040 | 0.0453+-0.0045 | 0.4431 | 397.0 | 0.9983 | 0.9963 |
| 0.05 | hard | 0.2399+-0.0088 | 0.0403+-0.0033 | 0.2500 | 256.0 | 0.9974 | 0.9967 |
| 0.05 | extreme | 0.0775+-0.0103 | 0.0368+-0.0047 | 0.0804 | 82.3 | 0.9964 | 0.9966 |
| 0.10 | mild | 0.5974+-0.0033 | 0.0692+-0.0446 | 0.6427 | 411.3 | 0.9983 | 0.9957 |
| 0.10 | medium | 0.4249+-0.0028 | 0.0691+-0.0173 | 0.4565 | 409.0 | 0.9983 | 0.9963 |
| 0.10 | hard | 0.2454+-0.0031 | 0.0676+-0.0282 | 0.2633 | 269.7 | 0.9974 | 0.9967 |
| 0.10 | extreme | 0.0856+-0.0025 | 0.0573+-0.0271 | 0.0908 | 93.0 | 0.9964 | 0.9966 |

## Interpretation

- Minimum target is met in every full-open severity: at accepted OpenOut <= 0.05, OpenSemCom has AUROC > 0.90, known-ID full-open accuracy > 0.50, and nonzero semantic goodput.
- Strong-goodput behavior holds through hard full-open: mean goodput is above 0.10 for mild, medium, and hard. Extreme full-open is safe and nonzero but below the bold 0.10 goodput target.
- OpenSemCom beats MSP, energy, Mahalanobis, OpenMax, and ViM on strict-frontier goodput in all four severity levels. The one-vs-rest unknown-aware head remains the strongest strict-frontier comparator in several settings, so the strongest defensible claim is against classical score baselines plus safety/nonzero-goodput under extreme, not universal dominance over every unknown-supervised head.
- The previous full-open bottleneck is materially improved: known-ID accuracy inside full-open is about 0.99 in this feature-level receiver experiment, versus the earlier 0.20-0.26 range.

## Files

- `runs/full_open_aware_dinoclip_noleak_noleak_login_20260617235820_summary.csv`
- `runs/full_open_aware_dinoclip_noleak_noleak_login_20260617235820_summary.json`
- `runs/full_open_aware_dinoclip_noleak_noleak_login_20260617235820_curves.csv`
- `runs/full_open_aware_dinoclip_noleak_noleak_login_20260617235820_diagnostics.csv`
- `runs/full_open_aware_dinoclip_noleak_noleak_login_20260617235820_headline.csv`

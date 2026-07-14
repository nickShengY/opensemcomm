# SOTA Full-Open Comparison Report

Output prefix: `runs/sota_full_open_sota_login_20260618030256`

## Setup

- Real rows: 11,392 common rows across DINOv3, SigLIP2, and OpenCLIP DFN5B feature manifests.
- Main method: OpenSemCom with DINOv3 + SigLIP2 + OpenCLIP DFN5B ensemble, learned selective-risk gate, and confirmed fallback acceptance.
- Compared methods: DINOv3 MSP, Energy, Mahalanobis, ViM, ReAct+Energy, ASH+Energy, one-vs-rest unknown head, SigLIP2 best OOD head, OpenCLIP DFN5B best OOD head, and DeepJSCC-style feature bottleneck.
- Calibration uses only real held-out known/open calibration rows from the same manifest intersection.

## Best Goodput at Accepted OpenOut <= 0.05

| Severity | Method | Backbone | Detector/control | Goodput | OpenOut | AUROC | FPR95 | Coverage | Accepted Known Acc | Accepted | Seeds |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mild | opensemcom | dino+siglip2+openclip | learned_selective_risk+confirmed_fallback | 0.5990+-0.0009 | 0.0177+-0.0176 | 0.9995 | 0.0009 | 0.6099 | 0.9823 | 390.3 | 3 |
| mild | dino_energy | DINOv3 | energy | 0.5901+-0.0033 | 0.0316+-0.0053 | 0.9931 | 0.0156 | 0.6094 | 0.9684 | 390.0 | 3 |
| mild | dino_msp | DINOv3 | msp | 0.5849+-0.0018 | 0.0352+-0.0105 | 0.9901 | 0.0286 | 0.6063 | 0.9648 | 388.0 | 3 |
| mild | dino_mahalanobis | DINOv3 | mahalanobis | 0.5687+-0.0056 | 0.0421+-0.0027 | 0.9899 | 0.0530 | 0.5938 | 0.9579 | 380.0 | 3 |
| mild | dino_vim | DINOv3 | vim | 0.5687+-0.0159 | 0.0463+-0.0025 | 0.9897 | 0.0668 | 0.5964 | 0.9537 | 381.7 | 3 |
| mild | dino_react_energy | DINOv3 | react_energy | 0.5901+-0.0033 | 0.0266+-0.0089 | 0.9932 | 0.0148 | 0.6063 | 0.9734 | 388.0 | 3 |
| mild | dino_ash_energy | DINOv3 | ash_energy | 0.5859+-0.0041 | 0.0392+-0.0097 | 0.9892 | 0.0269 | 0.6099 | 0.9608 | 390.3 | 3 |
| mild | dino_one_vs_rest | DINOv3 | one_vs_rest | 0.5969+-0.0016 | 0.0103+-0.0111 | 1.0000 | 0.0000 | 0.6031 | 0.9897 | 386.0 | 3 |
| mild | siglip2_best_ood | SigLIP2 | best_ood:one_vs_rest | 0.5953+-0.0041 | 0.0239+-0.0059 | 0.9999 | 0.0000 | 0.6099 | 0.9761 | 390.3 | 3 |
| mild | openclip_best_ood | OpenCLIP-DFN5B | best_ood:one_vs_rest | 0.5995+-0.0009 | 0.0111+-0.0097 | 0.9993 | 0.0009 | 0.6063 | 0.9889 | 388.0 | 3 |
| mild | deepjscc_style | DINOv3 | deepjscc | 0.5630+-0.0070 | 0.0458+-0.0048 | 0.9858 | 0.0781 | 0.5901 | 0.9542 | 377.7 | 3 |
| medium | opensemcom | dino+siglip2+openclip | learned_selective_risk+confirmed_fallback | 0.4278+-0.0006 | 0.0343+-0.0133 | 0.9995 | 0.0009 | 0.4431 | 0.9657 | 397.0 | 3 |
| medium | dino_energy | DINOv3 | energy | 0.4148+-0.0072 | 0.0429+-0.0033 | 0.9907 | 0.0165 | 0.4334 | 0.9571 | 388.3 | 3 |
| medium | dino_msp | DINOv3 | msp | 0.4129+-0.0051 | 0.0464+-0.0042 | 0.9879 | 0.0295 | 0.4330 | 0.9536 | 388.0 | 3 |
| medium | dino_mahalanobis | DINOv3 | mahalanobis | 0.3832+-0.0084 | 0.0427+-0.0024 | 0.9889 | 0.0703 | 0.4003 | 0.9573 | 358.7 | 3 |
| medium | dino_vim | DINOv3 | vim | 0.3899+-0.0141 | 0.0420+-0.0062 | 0.9896 | 0.0668 | 0.4070 | 0.9580 | 364.7 | 3 |
| medium | dino_react_energy | DINOv3 | react_energy | 0.4163+-0.0059 | 0.0460+-0.0026 | 0.9909 | 0.0165 | 0.4364 | 0.9540 | 391.0 | 3 |
| medium | dino_ash_energy | DINOv3 | ash_energy | 0.4096+-0.0011 | 0.0468+-0.0026 | 0.9864 | 0.0278 | 0.4297 | 0.9532 | 385.0 | 3 |
| medium | dino_one_vs_rest | DINOv3 | one_vs_rest | 0.4260+-0.0017 | 0.0162+-0.0168 | 0.9999 | 0.0000 | 0.4330 | 0.9838 | 388.0 | 3 |
| medium | siglip2_best_ood | SigLIP2 | best_ood:one_vs_rest | 0.4249+-0.0036 | 0.0336+-0.0147 | 0.9997 | 0.0000 | 0.4397 | 0.9664 | 394.0 | 3 |
| medium | openclip_best_ood | OpenCLIP-DFN5B | best_ood:one_vs_rest | 0.4282+-0.0006 | 0.0262+-0.0015 | 0.9997 | 0.0009 | 0.4397 | 0.9738 | 394.0 | 3 |
| medium | deepjscc_style | DINOv3 | deepjscc | 0.3668+-0.0063 | 0.0446+-0.0046 | 0.9850 | 0.0920 | 0.3839 | 0.9554 | 344.0 | 3 |
| hard | opensemcom | dino+siglip2+openclip | learned_selective_risk+confirmed_fallback | 0.2487+-0.0015 | 0.0426+-0.0057 | 0.9994 | 0.0000 | 0.2598 | 0.9574 | 266.0 | 3 |
| hard | dino_energy | DINOv3 | energy | 0.2197+-0.0152 | 0.0449+-0.0067 | 0.9923 | 0.0156 | 0.2301 | 0.9551 | 235.7 | 3 |
| hard | dino_msp | DINOv3 | msp | 0.2015+-0.0172 | 0.0434+-0.0032 | 0.9889 | 0.0286 | 0.2106 | 0.9566 | 215.7 | 3 |
| hard | dino_mahalanobis | DINOv3 | mahalanobis | 0.2210+-0.0098 | 0.0423+-0.0037 | 0.9901 | 0.0508 | 0.2308 | 0.9577 | 236.3 | 3 |
| hard | dino_vim | DINOv3 | vim | 0.2214+-0.0088 | 0.0395+-0.0036 | 0.9911 | 0.0586 | 0.2305 | 0.9605 | 236.0 | 3 |
| hard | dino_react_energy | DINOv3 | react_energy | 0.2161+-0.0060 | 0.0363+-0.0070 | 0.9925 | 0.0156 | 0.2243 | 0.9637 | 229.7 | 3 |
| hard | dino_ash_energy | DINOv3 | ash_energy | 0.1982+-0.0220 | 0.0456+-0.0024 | 0.9883 | 0.0339 | 0.2077 | 0.9544 | 212.7 | 3 |
| hard | dino_one_vs_rest | DINOv3 | one_vs_rest | 0.2487+-0.0015 | 0.0301+-0.0234 | 0.9999 | 0.0000 | 0.2565 | 0.9699 | 262.7 | 3 |
| hard | siglip2_best_ood | SigLIP2 | best_ood:one_vs_rest | 0.2477+-0.0006 | 0.0464+-0.0022 | 0.9998 | 0.0000 | 0.2598 | 0.9536 | 266.0 | 3 |
| hard | openclip_best_ood | OpenCLIP-DFN5B | best_ood:one_vs_rest | 0.2497+-0.0006 | 0.0388+-0.0022 | 0.9998 | 0.0000 | 0.2598 | 0.9612 | 266.0 | 3 |
| hard | deepjscc_style | DINOv3 | deepjscc | 0.1963+-0.0049 | 0.0382+-0.0125 | 0.9870 | 0.0690 | 0.2041 | 0.9618 | 209.0 | 3 |
| extreme | opensemcom | dino+siglip2+openclip | learned_selective_risk+confirmed_fallback | 0.0885+-0.0011 | 0.0251+-0.0124 | 0.9999 | 0.0000 | 0.0908 | 0.9749 | 93.0 | 3 |
| extreme | dino_energy | DINOv3 | energy | 0.0482+-0.0098 | 0.0452+-0.0058 | 0.9923 | 0.0181 | 0.0505 | 0.9548 | 51.7 | 3 |
| extreme | dino_msp | DINOv3 | msp | 0.0511+-0.0113 | 0.0485+-0.0002 | 0.9896 | 0.0290 | 0.0537 | 0.9515 | 55.0 | 3 |
| extreme | dino_mahalanobis | DINOv3 | mahalanobis | 0.0732+-0.0051 | 0.0464+-0.0041 | 0.9876 | 0.0797 | 0.0768 | 0.9536 | 78.7 | 3 |
| extreme | dino_vim | DINOv3 | vim | 0.0713+-0.0043 | 0.0301+-0.0176 | 0.9900 | 0.0688 | 0.0736 | 0.9699 | 75.3 | 3 |
| extreme | dino_react_energy | DINOv3 | react_energy | 0.0452+-0.0145 | 0.0397+-0.0081 | 0.9921 | 0.0217 | 0.0472 | 0.9603 | 48.3 | 3 |
| extreme | dino_ash_energy | DINOv3 | ash_energy | 0.0319+-0.0107 | 0.0484+-0.0007 | 0.9886 | 0.0362 | 0.0335 | 0.9516 | 34.3 | 3 |
| extreme | dino_one_vs_rest | DINOv3 | one_vs_rest | 0.0885+-0.0006 | 0.0251+-0.0062 | 1.0000 | 0.0000 | 0.0908 | 0.9749 | 93.0 | 3 |
| extreme | siglip2_best_ood | SigLIP2 | best_ood:one_vs_rest | 0.0895+-0.0006 | 0.0143+-0.0062 | 0.9998 | 0.0000 | 0.0908 | 0.9857 | 93.0 | 3 |
| extreme | openclip_best_ood | OpenCLIP-DFN5B | best_ood:one_vs_rest | 0.0889+-0.0000 | 0.0215+-0.0000 | 0.9999 | 0.0000 | 0.0908 | 0.9785 | 93.0 | 3 |
| extreme | deepjscc_style | DINOv3 | deepjscc | 0.0615+-0.0059 | 0.0361+-0.0108 | 0.9827 | 0.1123 | 0.0638 | 0.9639 | 65.3 | 3 |

## Best Goodput at Accepted OpenOut <= 0.10

| Severity | Method | Backbone | Detector/control | Goodput | OpenOut | AUROC | FPR95 | Coverage | Accepted Known Acc | Accepted | Seeds |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mild | opensemcom | dino+siglip2+openclip | learned_selective_risk+confirmed_fallback | 0.5990+-0.0009 | 0.0177+-0.0176 | 0.9995 | 0.0009 | 0.6099 | 0.9823 | 390.3 | 3 |
| mild | dino_energy | DINOv3 | energy | 0.5938+-0.0016 | 0.0758+-0.0226 | 0.9931 | 0.0156 | 0.6427 | 0.9242 | 411.3 | 3 |
| mild | dino_msp | DINOv3 | msp | 0.5906+-0.0000 | 0.0862+-0.0090 | 0.9901 | 0.0286 | 0.6464 | 0.9138 | 413.7 | 3 |
| mild | dino_mahalanobis | DINOv3 | mahalanobis | 0.5880+-0.0055 | 0.0946+-0.0059 | 0.9899 | 0.0530 | 0.6495 | 0.9054 | 415.7 | 3 |
| mild | dino_vim | DINOv3 | vim | 0.5870+-0.0080 | 0.0910+-0.0113 | 0.9897 | 0.0668 | 0.6458 | 0.9090 | 413.3 | 3 |
| mild | dino_react_energy | DINOv3 | react_energy | 0.5943+-0.0009 | 0.0759+-0.0184 | 0.9932 | 0.0148 | 0.6432 | 0.9241 | 411.7 | 3 |
| mild | dino_ash_energy | DINOv3 | ash_energy | 0.5917+-0.0033 | 0.0701+-0.0205 | 0.9892 | 0.0269 | 0.6365 | 0.9299 | 407.3 | 3 |
| mild | dino_one_vs_rest | DINOv3 | one_vs_rest | 0.5969+-0.0016 | 0.0103+-0.0111 | 1.0000 | 0.0000 | 0.6031 | 0.9897 | 386.0 | 3 |
| mild | siglip2_best_ood | SigLIP2 | best_ood:one_vs_rest | 0.5953+-0.0041 | 0.0239+-0.0059 | 0.9999 | 0.0000 | 0.6099 | 0.9761 | 390.3 | 3 |
| mild | openclip_best_ood | OpenCLIP-DFN5B | best_ood:one_vs_rest | 0.5995+-0.0009 | 0.0111+-0.0097 | 0.9993 | 0.0009 | 0.6063 | 0.9889 | 388.0 | 3 |
| mild | deepjscc_style | DINOv3 | deepjscc | 0.5859+-0.0078 | 0.0927+-0.0085 | 0.9858 | 0.0781 | 0.6458 | 0.9073 | 413.3 | 3 |
| medium | opensemcom | dino+siglip2+openclip | learned_selective_risk+confirmed_fallback | 0.4278+-0.0006 | 0.0343+-0.0133 | 0.9995 | 0.0009 | 0.4431 | 0.9657 | 397.0 | 3 |
| medium | dino_energy | DINOv3 | energy | 0.4222+-0.0026 | 0.0815+-0.0155 | 0.9907 | 0.0165 | 0.4598 | 0.9185 | 412.0 | 3 |
| medium | dino_msp | DINOv3 | msp | 0.4185+-0.0000 | 0.0759+-0.0240 | 0.9879 | 0.0295 | 0.4531 | 0.9241 | 406.0 | 3 |
| medium | dino_mahalanobis | DINOv3 | mahalanobis | 0.4051+-0.0062 | 0.0925+-0.0051 | 0.9889 | 0.0703 | 0.4464 | 0.9075 | 400.0 | 3 |
| medium | dino_vim | DINOv3 | vim | 0.4077+-0.0096 | 0.0935+-0.0037 | 0.9896 | 0.0668 | 0.4498 | 0.9065 | 403.0 | 3 |
| medium | dino_react_energy | DINOv3 | react_energy | 0.4222+-0.0026 | 0.0815+-0.0155 | 0.9909 | 0.0165 | 0.4598 | 0.9185 | 412.0 | 3 |
| medium | dino_ash_energy | DINOv3 | ash_energy | 0.4200+-0.0036 | 0.0931+-0.0039 | 0.9864 | 0.0278 | 0.4632 | 0.9069 | 415.0 | 3 |
| medium | dino_one_vs_rest | DINOv3 | one_vs_rest | 0.4263+-0.0011 | 0.0225+-0.0278 | 0.9999 | 0.0000 | 0.4364 | 0.9775 | 391.0 | 3 |
| medium | siglip2_best_ood | SigLIP2 | best_ood:one_vs_rest | 0.4252+-0.0030 | 0.0474+-0.0159 | 0.9997 | 0.0000 | 0.4464 | 0.9526 | 400.0 | 3 |
| medium | openclip_best_ood | OpenCLIP-DFN5B | best_ood:one_vs_rest | 0.4282+-0.0006 | 0.0262+-0.0015 | 0.9997 | 0.0009 | 0.4397 | 0.9738 | 394.0 | 3 |
| medium | deepjscc_style | DINOv3 | deepjscc | 0.4018+-0.0110 | 0.0940+-0.0051 | 0.9850 | 0.0920 | 0.4435 | 0.9060 | 397.3 | 3 |
| hard | opensemcom | dino+siglip2+openclip | learned_selective_risk+confirmed_fallback | 0.2493+-0.0011 | 0.0655+-0.0244 | 0.9994 | 0.0000 | 0.2669 | 0.9345 | 273.3 | 3 |
| hard | dino_energy | DINOv3 | energy | 0.2432+-0.0045 | 0.0900+-0.0074 | 0.9923 | 0.0156 | 0.2673 | 0.9100 | 273.7 | 3 |
| hard | dino_msp | DINOv3 | msp | 0.2386+-0.0015 | 0.0815+-0.0057 | 0.9889 | 0.0286 | 0.2598 | 0.9185 | 266.0 | 3 |
| hard | dino_mahalanobis | DINOv3 | mahalanobis | 0.2282+-0.0073 | 0.0871+-0.0064 | 0.9901 | 0.0508 | 0.2500 | 0.9129 | 256.0 | 3 |
| hard | dino_vim | DINOv3 | vim | 0.2308+-0.0096 | 0.0886+-0.0047 | 0.9911 | 0.0586 | 0.2533 | 0.9114 | 259.3 | 3 |
| hard | dino_react_energy | DINOv3 | react_energy | 0.2422+-0.0034 | 0.0926+-0.0085 | 0.9925 | 0.0156 | 0.2669 | 0.9074 | 273.3 | 3 |
| hard | dino_ash_energy | DINOv3 | ash_energy | 0.2360+-0.0020 | 0.0915+-0.0078 | 0.9883 | 0.0339 | 0.2598 | 0.9085 | 266.0 | 3 |
| hard | dino_one_vs_rest | DINOv3 | one_vs_rest | 0.2487+-0.0015 | 0.0301+-0.0234 | 0.9999 | 0.0000 | 0.2565 | 0.9699 | 262.7 | 3 |
| hard | siglip2_best_ood | SigLIP2 | best_ood:one_vs_rest | 0.2477+-0.0006 | 0.0464+-0.0022 | 0.9998 | 0.0000 | 0.2598 | 0.9536 | 266.0 | 3 |
| hard | openclip_best_ood | OpenCLIP-DFN5B | best_ood:one_vs_rest | 0.2500+-0.0000 | 0.0503+-0.0221 | 0.9998 | 0.0000 | 0.2633 | 0.9497 | 269.7 | 3 |
| hard | deepjscc_style | DINOv3 | deepjscc | 0.2210+-0.0088 | 0.0921+-0.0066 | 0.9870 | 0.0690 | 0.2435 | 0.9079 | 249.3 | 3 |
| extreme | opensemcom | dino+siglip2+openclip | learned_selective_risk+confirmed_fallback | 0.0885+-0.0011 | 0.0251+-0.0124 | 0.9999 | 0.0000 | 0.0908 | 0.9749 | 93.0 | 3 |
| extreme | dino_energy | DINOv3 | energy | 0.0664+-0.0101 | 0.0973+-0.0005 | 0.9923 | 0.0181 | 0.0736 | 0.9027 | 75.3 | 3 |
| extreme | dino_msp | DINOv3 | msp | 0.0648+-0.0176 | 0.0810+-0.0141 | 0.9896 | 0.0290 | 0.0706 | 0.9190 | 72.3 | 3 |
| extreme | dino_mahalanobis | DINOv3 | mahalanobis | 0.0732+-0.0051 | 0.0464+-0.0041 | 0.9876 | 0.0797 | 0.0768 | 0.9536 | 78.7 | 3 |
| extreme | dino_vim | DINOv3 | vim | 0.0726+-0.0034 | 0.0534+-0.0420 | 0.9900 | 0.0688 | 0.0768 | 0.9466 | 78.7 | 3 |
| extreme | dino_react_energy | DINOv3 | react_energy | 0.0687+-0.0105 | 0.0774+-0.0113 | 0.9921 | 0.0217 | 0.0745 | 0.9226 | 76.3 | 3 |
| extreme | dino_ash_energy | DINOv3 | ash_energy | 0.0550+-0.0093 | 0.0921+-0.0076 | 0.9886 | 0.0362 | 0.0605 | 0.9079 | 62.0 | 3 |
| extreme | dino_one_vs_rest | DINOv3 | one_vs_rest | 0.0885+-0.0006 | 0.0251+-0.0062 | 1.0000 | 0.0000 | 0.0908 | 0.9749 | 93.0 | 3 |
| extreme | siglip2_best_ood | SigLIP2 | best_ood:one_vs_rest | 0.0895+-0.0006 | 0.0143+-0.0062 | 0.9998 | 0.0000 | 0.0908 | 0.9857 | 93.0 | 3 |
| extreme | openclip_best_ood | OpenCLIP-DFN5B | best_ood:one_vs_rest | 0.0889+-0.0000 | 0.0215+-0.0000 | 0.9999 | 0.0000 | 0.0908 | 0.9785 | 93.0 | 3 |
| extreme | deepjscc_style | DINOv3 | deepjscc | 0.0654+-0.0068 | 0.0664+-0.0328 | 0.9827 | 0.1123 | 0.0703 | 0.9336 | 72.0 | 3 |

## Files

- `runs/sota_full_open_sota_login_20260618030256_summary.csv`
- `runs/sota_full_open_sota_login_20260618030256_summary.json`
- `runs/sota_full_open_sota_login_20260618030256_curves.csv`
- `runs/sota_full_open_sota_login_20260618030256_diagnostics.csv`
- `runs/sota_full_open_sota_login_20260618030256_headline.csv`
- `runs/sota_full_open_sota_login_20260618030256_manifest_summary.json`

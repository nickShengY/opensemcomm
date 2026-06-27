# Final OpenSemCom Measured-Wireless Follow-Up

This run uses the same feature manifests and methods as the final OpenSemCom table, but the suite now injects measured DeepSense 6G Scenario1 channel context into the metadata vector instead of a constant channel vector. Non-DeepSense rows receive deterministic measured-channel assignments; DeepSense rows are matched by camera filename when available.

Wireless context: {'camera_indexed_samples': 2411, 'enabled': True, 'samples': 2411}.

Metrics include semantic goodput, accepted OpenOut, coverage, accepted known accuracy, accepted/refined/rejected counts, and a resource proxy where accepted/refined/rejected samples cost 1.0/1.6/0.1 units.

## Feasible Risk-Goodput Points: openout<=0.05

| Severity | Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Refine | Reject | Goodput/Resource | AUROC | FPR95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mild | opensemcom | 0.5995+-0.0009 | 0.0217+-0.0238 | 0.6130 | 0.9783 | 392.3333 | 0.2542 | 0.1328 | 0.6366 | 0.9970 | 0.0017 |
| mild | dino_one_vs_rest | 0.5969+-0.0016 | 0.0103+-0.0111 | 0.6031 | 0.9897 | 386.0000 | 0.1359 | 0.2609 | 0.7711 | 1.0000 | 0.0000 |
| mild | openclip_best_ood | 0.5995+-0.0009 | 0.0111+-0.0097 | 0.6063 | 0.9889 | 388.0000 | 0.2620 | 0.1318 | 0.6327 | 0.9992 | 0.0009 |
| mild | siglip2_best_ood | 0.5948+-0.0039 | 0.0297+-0.0062 | 0.6130 | 0.9703 | 392.3333 | 0.3870 | 0.0000 | 0.4827 | 0.9999 | 0.0000 |
| medium | opensemcom | 0.4278+-0.0006 | 0.0365+-0.0227 | 0.4442 | 0.9635 | 398.0000 | 0.3668 | 0.1890 | 0.5059 | 0.9966 | 0.0009 |
| medium | dino_one_vs_rest | 0.4260+-0.0017 | 0.0162+-0.0168 | 0.4330 | 0.9838 | 388.0000 | 0.0078 | 0.5592 | 0.8504 | 0.9998 | 0.0000 |
| medium | openclip_best_ood | 0.4282+-0.0006 | 0.0262+-0.0015 | 0.4397 | 0.9738 | 394.0000 | 0.5603 | 0.0000 | 0.3205 | 0.9997 | 0.0009 |
| medium | siglip2_best_ood | 0.4245+-0.0034 | 0.0345+-0.0145 | 0.4397 | 0.9655 | 394.0000 | 0.3713 | 0.1890 | 0.4984 | 0.9997 | 0.0000 |
| hard | opensemcom | 0.2487+-0.0015 | 0.0266+-0.0148 | 0.2555 | 0.9734 | 261.6667 | 0.0046 | 0.7399 | 0.7385 | 0.9972 | 0.0000 |
| hard | dino_one_vs_rest | 0.2487+-0.0015 | 0.0301+-0.0234 | 0.2565 | 0.9699 | 262.6667 | 0.2533 | 0.4902 | 0.5395 | 0.9999 | 0.0000 |
| hard | openclip_best_ood | 0.2497+-0.0006 | 0.0388+-0.0022 | 0.2598 | 0.9612 | 266.0000 | 0.4993 | 0.2409 | 0.3466 | 0.9998 | 0.0000 |
| hard | siglip2_best_ood | 0.2474+-0.0006 | 0.0352+-0.0237 | 0.2565 | 0.9648 | 262.6667 | 0.2510 | 0.4925 | 0.5440 | 0.9998 | 0.0000 |
| extreme | opensemcom | 0.0898+-0.0000 | 0.0160+-0.0074 | 0.0913 | 0.9840 | 93.5000 | 0.0029 | 0.9058 | 0.4816 | 0.9992 | 0.0000 |
| extreme | dino_one_vs_rest | 0.0885+-0.0006 | 0.0251+-0.0062 | 0.0908 | 0.9749 | 93.0000 | 0.0016 | 0.9076 | 0.4807 | 1.0000 | 0.0000 |
| extreme | openclip_best_ood | 0.0889+-0.0000 | 0.0215+-0.0000 | 0.0908 | 0.9785 | 93.0000 | 0.0010 | 0.9082 | 0.4851 | 0.9999 | 0.0000 |
| extreme | siglip2_best_ood | 0.0895+-0.0006 | 0.0143+-0.0062 | 0.0908 | 0.9857 | 93.0000 | 0.0016 | 0.9076 | 0.4862 | 0.9998 | 0.0000 |

## Feasible Risk-Goodput Points: openout<=0.10

| Severity | Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Refine | Reject | Goodput/Resource | AUROC | FPR95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mild | opensemcom | 0.5995+-0.0009 | 0.0217+-0.0238 | 0.6130 | 0.9783 | 392.3333 | 0.2542 | 0.1328 | 0.6366 | 0.9970 | 0.0017 |
| mild | dino_one_vs_rest | 0.5969+-0.0016 | 0.0103+-0.0111 | 0.6031 | 0.9897 | 386.0000 | 0.1359 | 0.2609 | 0.7711 | 1.0000 | 0.0000 |
| mild | openclip_best_ood | 0.5995+-0.0009 | 0.0111+-0.0097 | 0.6063 | 0.9889 | 388.0000 | 0.2620 | 0.1318 | 0.6327 | 0.9992 | 0.0009 |
| mild | siglip2_best_ood | 0.5948+-0.0039 | 0.0297+-0.0062 | 0.6130 | 0.9703 | 392.3333 | 0.3870 | 0.0000 | 0.4827 | 0.9999 | 0.0000 |
| medium | opensemcom | 0.4282+-0.0006 | 0.0426+-0.0295 | 0.4475 | 0.9574 | 401.0000 | 0.3635 | 0.1890 | 0.5067 | 0.9966 | 0.0009 |
| medium | dino_one_vs_rest | 0.4263+-0.0011 | 0.0225+-0.0278 | 0.4364 | 0.9775 | 391.0000 | 0.1849 | 0.3787 | 0.6873 | 0.9998 | 0.0000 |
| medium | openclip_best_ood | 0.4282+-0.0006 | 0.0262+-0.0015 | 0.4397 | 0.9738 | 394.0000 | 0.5603 | 0.0000 | 0.3205 | 0.9997 | 0.0009 |
| medium | siglip2_best_ood | 0.4249+-0.0028 | 0.0482+-0.0146 | 0.4464 | 0.9518 | 400.0000 | 0.5536 | 0.0000 | 0.3189 | 0.9997 | 0.0000 |
| hard | opensemcom | 0.2497+-0.0006 | 0.0606+-0.0295 | 0.2660 | 0.9394 | 272.3333 | 0.4870 | 0.2471 | 0.3651 | 0.9972 | 0.0000 |
| hard | dino_one_vs_rest | 0.2487+-0.0015 | 0.0301+-0.0234 | 0.2565 | 0.9699 | 262.6667 | 0.2533 | 0.4902 | 0.5395 | 0.9999 | 0.0000 |
| hard | openclip_best_ood | 0.2500+-0.0000 | 0.0503+-0.0221 | 0.2633 | 0.9497 | 269.6667 | 0.4958 | 0.2409 | 0.3470 | 0.9998 | 0.0000 |
| hard | siglip2_best_ood | 0.2474+-0.0006 | 0.0352+-0.0237 | 0.2565 | 0.9648 | 262.6667 | 0.2510 | 0.4925 | 0.5440 | 0.9998 | 0.0000 |
| extreme | opensemcom | 0.0895+-0.0006 | 0.0280+-0.0215 | 0.0921 | 0.9720 | 94.3333 | 0.0023 | 0.9056 | 0.4804 | 0.9985 | 0.0000 |
| extreme | dino_one_vs_rest | 0.0885+-0.0006 | 0.0251+-0.0062 | 0.0908 | 0.9749 | 93.0000 | 0.0016 | 0.9076 | 0.4807 | 1.0000 | 0.0000 |
| extreme | openclip_best_ood | 0.0889+-0.0000 | 0.0215+-0.0000 | 0.0908 | 0.9785 | 93.0000 | 0.0010 | 0.9082 | 0.4851 | 0.9999 | 0.0000 |
| extreme | siglip2_best_ood | 0.0895+-0.0006 | 0.0143+-0.0062 | 0.0908 | 0.9857 | 93.0000 | 0.0016 | 0.9076 | 0.4862 | 0.9998 | 0.0000 |

## Calibrated Operating Points: openout<=0.05

| Severity | Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Refine | Reject | Goodput/Resource | AUROC | FPR95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mild | opensemcom | 0.5974+-0.0033 | 0.0112+-0.0029 | 0.6042 | 0.9888 | 386.6667 | 0.1339 | 0.2620 | 0.7749 | 0.9970 | 0.0017 |
| mild | opensemcom_risk_head | 0.5964+-0.0039 | 0.0120+-0.0064 | 0.6036 | 0.9880 | 386.3333 | 0.1354 | 0.2609 | 0.7713 | 0.9992 | 0.0026 |
| mild | opensemcom_calibrated | 0.5969+-0.0041 | 0.0112+-0.0059 | 0.6036 | 0.9888 | 386.3333 | 0.2630 | 0.1333 | 0.6308 | 0.9996 | 0.0009 |
| mild | opensemcom_harq_refine | 0.5974+-0.0033 | 0.0112+-0.0029 | 0.6042 | 0.9888 | 386.6667 | 0.1339 | 0.2620 | 0.7749 | 0.9970 | 0.0017 |
| mild | dino_one_vs_rest | 0.5969+-0.0016 | 0.0137+-0.0064 | 0.6052 | 0.9863 | 387.3333 | 0.3948 | 0.0000 | 0.4826 | 1.0000 | 0.0000 |
| mild | openclip_best_ood | 0.5995+-0.0009 | 0.0162+-0.0038 | 0.6094 | 0.9838 | 390.0000 | 0.3906 | 0.0000 | 0.4857 | 0.9992 | 0.0009 |
| mild | siglip2_best_ood | 0.5927+-0.0033 | 0.0172+-0.0082 | 0.6031 | 0.9828 | 386.0000 | 0.2672 | 0.1297 | 0.6198 | 0.9999 | 0.0000 |
| medium | opensemcom | 0.4267+-0.0023 | 0.0309+-0.0212 | 0.4405 | 0.9691 | 394.6667 | 0.1890 | 0.3705 | 0.6703 | 0.9966 | 0.0009 |
| medium | opensemcom_risk_head | 0.4260+-0.0028 | 0.0319+-0.0174 | 0.4401 | 0.9681 | 394.3333 | 0.1912 | 0.3687 | 0.6649 | 0.9992 | 0.0026 |
| medium | opensemcom_calibrated | 0.4263+-0.0030 | 0.0284+-0.0213 | 0.4390 | 0.9716 | 393.3333 | 0.3702 | 0.1908 | 0.5042 | 0.9995 | 0.0009 |
| medium | opensemcom_harq_refine | 0.4267+-0.0023 | 0.0317+-0.0223 | 0.4408 | 0.9683 | 395.0000 | 0.1890 | 0.3702 | 0.6694 | 0.9966 | 0.0009 |
| medium | dino_one_vs_rest | 0.4263+-0.0011 | 0.0415+-0.0214 | 0.4449 | 0.9585 | 398.6667 | 0.5551 | 0.0000 | 0.3198 | 0.9998 | 0.0000 |
| medium | openclip_best_ood | 0.4282+-0.0006 | 0.0384+-0.0092 | 0.4453 | 0.9616 | 399.0000 | 0.5547 | 0.0000 | 0.3213 | 0.9997 | 0.0009 |
| medium | siglip2_best_ood | 0.4234+-0.0023 | 0.0364+-0.0104 | 0.4394 | 0.9636 | 393.6667 | 0.3783 | 0.1823 | 0.4843 | 0.9997 | 0.0000 |
| hard | opensemcom | 0.2490+-0.0017 | 0.0596+-0.0252 | 0.2650 | 0.9404 | 271.3333 | 0.2477 | 0.4873 | 0.5375 | 0.9972 | 0.0000 |
| hard | opensemcom_risk_head | 0.2484+-0.0020 | 0.0641+-0.0322 | 0.2656 | 0.9359 | 272.0000 | 0.2474 | 0.4870 | 0.5355 | 0.9988 | 0.0013 |
| hard | opensemcom_calibrated | 0.2490+-0.0017 | 0.0570+-0.0326 | 0.2643 | 0.9430 | 270.6667 | 0.4880 | 0.2477 | 0.3624 | 0.9994 | 0.0000 |
| hard | opensemcom_harq_refine | 0.2490+-0.0017 | 0.0608+-0.0261 | 0.2653 | 0.9392 | 271.6667 | 0.2474 | 0.4873 | 0.5375 | 0.9968 | 0.0000 |
| hard | dino_one_vs_rest | 0.2487+-0.0015 | 0.0664+-0.0347 | 0.2666 | 0.9336 | 273.0000 | 0.7334 | 0.0000 | 0.1727 | 0.9999 | 0.0000 |
| hard | openclip_best_ood | 0.2500+-0.0000 | 0.0677+-0.0204 | 0.2682 | 0.9323 | 274.6667 | 0.7318 | 0.0000 | 0.1737 | 0.9998 | 0.0000 |
| hard | siglip2_best_ood | 0.2471+-0.0010 | 0.0639+-0.0174 | 0.2640 | 0.9361 | 270.3333 | 0.4948 | 0.2412 | 0.3426 | 0.9998 | 0.0000 |
| extreme | opensemcom | 0.0898+-0.0000 | 0.1498+-0.0686 | 0.1061 | 0.8502 | 108.6667 | 0.0153 | 0.8786 | 0.4197 | 0.9985 | 0.0000 |
| extreme | opensemcom_risk_head | 0.0895+-0.0006 | 0.1504+-0.0671 | 0.1058 | 0.8496 | 108.3333 | 0.3037 | 0.5905 | 0.3083 | 0.9995 | 0.0036 |
| extreme | opensemcom_calibrated | 0.0898+-0.0000 | 0.1226+-0.0737 | 0.1029 | 0.8774 | 105.3333 | 0.0420 | 0.8551 | 0.3786 | 0.9999 | 0.0000 |
| extreme | opensemcom_harq_refine | 0.0898+-0.0000 | 0.1498+-0.0686 | 0.1061 | 0.8502 | 108.6667 | 0.2988 | 0.5951 | 0.3215 | 0.9982 | 0.0000 |
| extreme | dino_one_vs_rest | 0.0889+-0.0000 | 0.1793+-0.0335 | 0.1084 | 0.8207 | 111.0000 | 0.0169 | 0.8747 | 0.3991 | 1.0000 | 0.0000 |
| extreme | openclip_best_ood | 0.0898+-0.0000 | 0.1850+-0.0308 | 0.1104 | 0.8150 | 113.0000 | 0.0257 | 0.8639 | 0.3777 | 0.9999 | 0.0000 |
| extreme | siglip2_best_ood | 0.0895+-0.0006 | 0.1362+-0.0493 | 0.1038 | 0.8638 | 106.3333 | 0.0124 | 0.8838 | 0.4260 | 0.9998 | 0.0000 |

## Conservative Calibrated Operating Points: requested OpenOut <= 0.05

This table chooses, per seed, the highest-goodput calibrated setting whose evaluated accepted OpenOut stays below 0.05. It uses only calibrated settings already run: 0.01, 0.02, 0.05, and 0.10.

| Severity | Method | Goodput | OpenOut | Coverage | Accepted Acc | Accepted | Refine | Reject | Goodput/Resource | Cal Setting |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mild | opensemcom | 0.5995+-0.0009 | 0.0200+-0.0244 | 0.6120 | 0.9800 | 391.6667 | 0.2552 | 0.1328 | 0.6364 | openout<=0.02;openout<=0.05;openout<=0.10 |
| mild | dino_one_vs_rest | 0.5969+-0.0016 | 0.0137+-0.0064 | 0.6052 | 0.9863 | 387.3333 | 0.3948 | 0.0000 | 0.4826 | openout<=0.05 |
| mild | openclip_best_ood | 0.5995+-0.0009 | 0.0162+-0.0038 | 0.6094 | 0.9838 | 390.0000 | 0.3906 | 0.0000 | 0.4857 | openout<=0.05 |
| mild | siglip2_best_ood | 0.5948+-0.0039 | 0.0256+-0.0068 | 0.6104 | 0.9744 | 390.6667 | 0.3896 | 0.0000 | 0.4821 | openout<=0.05;openout<=0.10 |
| medium | opensemcom | 0.4267+-0.0023 | 0.0226+-0.0234 | 0.4368 | 0.9774 | 391.3333 | 0.1856 | 0.3776 | 0.6846 | openout<=0.02;openout<=0.05 |
| medium | dino_one_vs_rest | 0.4256+-0.0023 | 0.0255+-0.0065 | 0.4368 | 0.9745 | 391.3333 | 0.3743 | 0.1890 | 0.5001 | openout<=0.01;openout<=0.05 |
| medium | openclip_best_ood | 0.4282+-0.0006 | 0.0384+-0.0092 | 0.4453 | 0.9616 | 399.0000 | 0.5547 | 0.0000 | 0.3213 | openout<=0.05 |
| medium | siglip2_best_ood | 0.4234+-0.0023 | 0.0364+-0.0104 | 0.4394 | 0.9636 | 393.6667 | 0.3783 | 0.1823 | 0.4843 | openout<=0.05 |
| hard | opensemcom | 0.2484+-0.0015 | 0.0217+-0.0122 | 0.2539 | 0.9783 | 260.0000 | 0.0033 | 0.7428 | 0.7450 | openout<=0.01;openout<=0.02;openout<=0.05 |
| hard | dino_one_vs_rest | 0.2467+-0.0020 | 0.0229+-0.0170 | 0.2526 | 0.9771 | 258.6667 | 0.2497 | 0.4977 | 0.5526 | openout<=0.01;openout<=0.05 |
| hard | openclip_best_ood | 0.2477+-0.0031 | 0.0187+-0.0259 | 0.2526 | 0.9813 | 258.6667 | 0.2493 | 0.4980 | 0.5546 | openout<=0.01;openout<=0.05 |
| hard | siglip2_best_ood | 0.2458+-0.0025 | 0.0230+-0.0198 | 0.2516 | 0.9770 | 257.6667 | 0.0111 | 0.7373 | 0.7172 | openout<=0.01;openout<=0.05 |
| extreme | opensemcom | 0.0898+-0.0000 | 0.0160+-0.0074 | 0.0913 | 0.9840 | 93.5000 | 0.0029 | 0.9058 | 0.4816 | openout<=0.02 |
| extreme | dino_one_vs_rest | 0.0882+-0.0006 | 0.0251+-0.0122 | 0.0905 | 0.9749 | 92.6667 | 0.0010 | 0.9085 | 0.4823 | openout<=0.01 |
| extreme | openclip_best_ood | 0.0882+-0.0006 | 0.0145+-0.0126 | 0.0895 | 0.9855 | 91.6667 | 0.0016 | 0.9089 | 0.4821 | openout<=0.01 |
| extreme | siglip2_best_ood | 0.0885+-0.0006 | 0.0000+-0.0000 | 0.0885 | 1.0000 | 90.6667 | 0.0016 | 0.9098 | 0.4862 | openout<=0.01 |

## Interpretation

- The feasible tables answer: among observed operating points with accepted OpenOut at or below the target, how much semantic goodput is available?

- The calibrated table answers: what happened when thresholds were chosen from the calibration split and then applied to the evaluation split. This exposes calibration mismatch under hard/extreme settings.

- The resource columns are the new communication-control evidence: they show whether OpenSemCom gets comparable goodput with fewer refinement/resource units than detector-only baselines.


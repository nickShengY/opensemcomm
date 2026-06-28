# OpenSemCom Trained Receiver Communication-Control Results

This run adds a trained OpenSemCom receiver/risk head on top of the DINOv3 + SigLIP2 + OpenCLIP feature ensemble and measured DeepSense channel metadata. The receiver is trained on manifest rows with class, unsafe/open, acceptability, ranking, and correct-accept reward terms.

Decision rule: accept low-risk samples, refine medium-risk samples once, and reject/open high-risk samples. There is no fallback acceptance rule.

Policy selection uses a Wilson upper-confidence bound on calibration outage. Final rows report AUROC and FPR95 against the actual unsafe condition, meaning open exposure or wrong semantic prediction.

Run scope: 5 seeds, targets 0.01/0.02/0.05/0.10, resource budgets 0.30/0.45/0.60/0.80/1.00, full-open and DeepSense measured beam-sector tasks.

## Headline at target accepted outage <= 0.05

### full-open, resource budget 0.30

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.2844 +/- 0.0105 | 0.0024 | 0.2850 | 0.9976 | 0.9968 | 0.0078 | 255.4 | 0.3565 | 0.0000 |
| dino_detector | 0.2694 +/- 0.0148 | 0.0064 | 0.2712 | 0.9936 | 0.9921 | 0.0175 | 243.0 | 0.3441 | 0.0000 |
| opensemcom_receiver_only | 0.2188 +/- 0.0142 | 0.0020 | 0.2192 | 0.9980 | 0.9984 | 0.0016 | 196.4 | 0.3411 | 0.0000 |
| opensemcom_no_channel | 0.2170 +/- 0.0130 | 0.0031 | 0.2176 | 0.9969 | 0.9986 | 0.0010 | 195.0 | 0.3394 | 0.0000 |
| ensemble_detector | 0.2116 +/- 0.0157 | 0.0010 | 0.2118 | 0.9990 | 0.9988 | 0.0005 | 189.8 | 0.3330 | 0.0000 |
| fixed_refine_all | 0.1585 +/- 0.0148 | 0.0014 | 0.1587 | 0.9986 | 0.9989 | 0.0005 | 142.2 | 0.3381 | 0.0000 |
| witt_context_style | 0.1585 +/- 0.0148 | 0.0014 | 0.1587 | 0.9986 | 0.9989 | 0.0005 | 142.2 | 0.3063 | 0.0000 |
| deepjscc_pca | 0.0190 +/- 0.0424 | 0.0046 | 0.0194 | 0.1954 | 0.9125 | 0.3884 | 17.4 | 0.1117 | 0.0000 |

### full-open, resource budget 0.45

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4246 +/- 0.0028 | 0.0190 | 0.4328 | 0.9810 | 0.9980 | 0.0037 | 387.8 | 0.4927 | 0.0042 |
| opensemcom_no_channel | 0.3777 +/- 0.0053 | 0.0030 | 0.3788 | 0.9970 | 0.9986 | 0.0010 | 339.4 | 0.5167 | 0.0000 |
| opensemcom_receiver_only | 0.3775 +/- 0.0089 | 0.0029 | 0.3786 | 0.9971 | 0.9984 | 0.0016 | 339.2 | 0.5164 | 0.0000 |
| ensemble_detector | 0.3748 +/- 0.0169 | 0.0030 | 0.3759 | 0.9970 | 0.9988 | 0.0005 | 336.8 | 0.5135 | 0.0000 |
| dino_detector | 0.3723 +/- 0.0111 | 0.0088 | 0.3757 | 0.9912 | 0.9921 | 0.0175 | 336.6 | 0.4381 | 0.0000 |
| witt_context_style | 0.3214 +/- 0.0112 | 0.0014 | 0.3219 | 0.9986 | 0.9989 | 0.0005 | 288.4 | 0.5184 | 0.0000 |
| fixed_refine_all | 0.2732 +/- 0.0196 | 0.0008 | 0.2734 | 0.9992 | 0.9989 | 0.0005 | 245.0 | 0.5102 | 0.0000 |
| deepjscc_pca | 0.0190 +/- 0.0424 | 0.0046 | 0.0194 | 0.1954 | 0.9125 | 0.3884 | 17.4 | 0.1117 | 0.0000 |

### full-open, resource budget 0.60

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4246 +/- 0.0028 | 0.0190 | 0.4328 | 0.9810 | 0.9980 | 0.0037 | 387.8 | 0.4927 | 0.0042 |
| opensemcom_no_channel | 0.4228 +/- 0.0024 | 0.0135 | 0.4286 | 0.9865 | 0.9986 | 0.0010 | 384.0 | 0.5714 | 0.0000 |
| opensemcom_receiver_only | 0.4219 +/- 0.0032 | 0.0161 | 0.4288 | 0.9839 | 0.9984 | 0.0016 | 384.2 | 0.5717 | 0.0000 |
| ensemble_detector | 0.4217 +/- 0.0021 | 0.0165 | 0.4288 | 0.9835 | 0.9988 | 0.0005 | 384.2 | 0.5717 | 0.0000 |
| witt_context_style | 0.4217 +/- 0.0021 | 0.0170 | 0.4290 | 0.9830 | 0.9989 | 0.0005 | 384.4 | 0.6577 | 0.0000 |
| fixed_refine_all | 0.3754 +/- 0.0103 | 0.0024 | 0.3763 | 0.9976 | 0.9989 | 0.0005 | 337.2 | 0.6645 | 0.0000 |
| dino_detector | 0.3723 +/- 0.0111 | 0.0088 | 0.3757 | 0.9912 | 0.9921 | 0.0175 | 336.6 | 0.4381 | 0.0000 |
| deepjscc_pca | 0.0190 +/- 0.0424 | 0.0046 | 0.0194 | 0.1954 | 0.9125 | 0.3884 | 17.4 | 0.1117 | 0.0000 |

### full-open, resource budget 1.00

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4246 +/- 0.0028 | 0.0190 | 0.4328 | 0.9810 | 0.9980 | 0.0037 | 387.8 | 0.4927 | 0.0042 |
| opensemcom_no_channel | 0.4228 +/- 0.0024 | 0.0135 | 0.4286 | 0.9865 | 0.9986 | 0.0010 | 384.0 | 0.5714 | 0.0000 |
| opensemcom_receiver_only | 0.4219 +/- 0.0032 | 0.0161 | 0.4288 | 0.9839 | 0.9984 | 0.0016 | 384.2 | 0.5717 | 0.0000 |
| ensemble_detector | 0.4217 +/- 0.0021 | 0.0165 | 0.4288 | 0.9835 | 0.9988 | 0.0005 | 384.2 | 0.5717 | 0.0000 |
| fixed_refine_all | 0.4217 +/- 0.0021 | 0.0170 | 0.4290 | 0.9830 | 0.9989 | 0.0005 | 384.4 | 0.7435 | 0.0000 |
| witt_context_style | 0.4217 +/- 0.0021 | 0.0170 | 0.4290 | 0.9830 | 0.9989 | 0.0005 | 384.4 | 0.6577 | 0.0000 |
| dino_detector | 0.3723 +/- 0.0111 | 0.0088 | 0.3757 | 0.9912 | 0.9921 | 0.0175 | 336.6 | 0.4381 | 0.0000 |
| deepjscc_pca | 0.0190 +/- 0.0424 | 0.0046 | 0.0194 | 0.1954 | 0.9125 | 0.3884 | 17.4 | 0.1117 | 0.0000 |

### DeepSense measured beam-sector, resource budget 0.30

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_no_channel | 0.0092 +/- 0.0084 | 0.0000 | 0.0092 | 0.8000 | 0.8305 | 0.5203 | 1.2 | 0.1101 | 0.0000 |
| dino_detector | 0.0061 +/- 0.0100 | 0.0000 | 0.0061 | 0.4000 | 0.8337 | 0.5722 | 0.8 | 0.1055 | 0.0000 |
| fixed_refine_all | 0.0046 +/- 0.0068 | 0.0000 | 0.0046 | 0.4000 | 0.8301 | 0.5744 | 0.6 | 0.1069 | 0.0000 |
| opensemcom_progressive | 0.0046 +/- 0.0102 | 0.0000 | 0.0046 | 0.2000 | 0.8337 | 0.5722 | 0.6 | 0.1041 | 0.0000 |
| witt_context_style | 0.0046 +/- 0.0068 | 0.0000 | 0.0046 | 0.4000 | 0.8301 | 0.5744 | 0.6 | 0.1060 | 0.0000 |
| deepjscc_pca | 0.0031 +/- 0.0042 | 0.0000 | 0.0031 | 0.4000 | 0.6470 | 0.8102 | 0.4 | 0.1018 | 0.0000 |
| ensemble_detector | 0.0031 +/- 0.0068 | 0.0000 | 0.0031 | 0.2000 | 0.8367 | 0.5601 | 0.4 | 0.1034 | 0.0000 |
| opensemcom_receiver_only | 0.0015 +/- 0.0034 | 0.0000 | 0.0015 | 0.2000 | 0.8455 | 0.4164 | 0.2 | 0.1017 | 0.0000 |

### DeepSense measured beam-sector, resource budget 0.45

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_no_channel | 0.0092 +/- 0.0084 | 0.0000 | 0.0092 | 0.8000 | 0.8305 | 0.5203 | 1.2 | 0.1101 | 0.0000 |
| dino_detector | 0.0061 +/- 0.0100 | 0.0000 | 0.0061 | 0.4000 | 0.8337 | 0.5722 | 0.8 | 0.1055 | 0.0000 |
| fixed_refine_all | 0.0046 +/- 0.0068 | 0.0000 | 0.0046 | 0.4000 | 0.8301 | 0.5744 | 0.6 | 0.1069 | 0.0000 |
| opensemcom_progressive | 0.0046 +/- 0.0102 | 0.0000 | 0.0046 | 0.2000 | 0.8337 | 0.5722 | 0.6 | 0.1041 | 0.0000 |
| witt_context_style | 0.0046 +/- 0.0068 | 0.0000 | 0.0046 | 0.4000 | 0.8301 | 0.5744 | 0.6 | 0.1060 | 0.0000 |
| deepjscc_pca | 0.0031 +/- 0.0042 | 0.0000 | 0.0031 | 0.4000 | 0.6470 | 0.8102 | 0.4 | 0.1018 | 0.0000 |
| ensemble_detector | 0.0031 +/- 0.0068 | 0.0000 | 0.0031 | 0.2000 | 0.8367 | 0.5601 | 0.4 | 0.1034 | 0.0000 |
| opensemcom_receiver_only | 0.0015 +/- 0.0034 | 0.0000 | 0.0015 | 0.2000 | 0.8455 | 0.4164 | 0.2 | 0.1017 | 0.0000 |

### DeepSense measured beam-sector, resource budget 0.60

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4031 +/- 0.2301 | 0.0108 | 0.4092 | 0.7892 | 0.8102 | 0.5487 | 53.6 | 0.4682 | 0.0000 |
| ensemble_detector | 0.2901 +/- 0.1716 | 0.0138 | 0.2962 | 0.7862 | 0.8367 | 0.5601 | 38.8 | 0.4258 | 0.0000 |
| dino_detector | 0.2733 +/- 0.2397 | 0.0072 | 0.2763 | 0.9928 | 0.8337 | 0.5722 | 36.2 | 0.3487 | 0.0000 |
| opensemcom_no_channel | 0.2580 +/- 0.2290 | 0.0134 | 0.2641 | 0.7866 | 0.8305 | 0.5203 | 34.6 | 0.3905 | 0.0000 |
| opensemcom_receiver_only | 0.2427 +/- 0.2217 | 0.0032 | 0.2443 | 0.7968 | 0.8455 | 0.4164 | 32.0 | 0.3687 | 0.0000 |
| fixed_refine_all | 0.0046 +/- 0.0068 | 0.0000 | 0.0046 | 0.4000 | 0.8301 | 0.5744 | 0.6 | 0.1069 | 0.0000 |
| witt_context_style | 0.0046 +/- 0.0068 | 0.0000 | 0.0046 | 0.4000 | 0.8301 | 0.5744 | 0.6 | 0.1060 | 0.0000 |
| deepjscc_pca | 0.0031 +/- 0.0042 | 0.0000 | 0.0031 | 0.4000 | 0.6470 | 0.8102 | 0.4 | 0.1018 | 0.0000 |

### DeepSense measured beam-sector, resource budget 1.00

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Accepted | Resource/sample | Refine rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.6489 +/- 0.1335 | 0.0261 | 0.6687 | 0.9739 | 0.8264 | 0.4997 | 87.6 | 0.8507 | 0.1313 |
| ensemble_detector | 0.4076 +/- 0.2487 | 0.0196 | 0.4198 | 0.7804 | 0.8367 | 0.5601 | 55.0 | 0.5618 | 0.0000 |
| fixed_refine_all | 0.3878 +/- 0.2277 | 0.0168 | 0.3969 | 0.7832 | 0.8301 | 0.5744 | 52.0 | 0.6954 | 0.0000 |
| witt_context_style | 0.3878 +/- 0.2277 | 0.0168 | 0.3969 | 0.7832 | 0.8301 | 0.5744 | 52.0 | 0.6160 | 0.0000 |
| dino_detector | 0.3511 +/- 0.3277 | 0.0145 | 0.3618 | 0.9855 | 0.8337 | 0.5722 | 47.4 | 0.4256 | 0.0000 |
| opensemcom_no_channel | 0.3496 +/- 0.3152 | 0.0183 | 0.3603 | 0.7817 | 0.8305 | 0.5203 | 47.2 | 0.4963 | 0.0000 |
| opensemcom_receiver_only | 0.3405 +/- 0.3194 | 0.0135 | 0.3496 | 0.7865 | 0.8455 | 0.4164 | 45.8 | 0.4846 | 0.0000 |
| deepjscc_pca | 0.0031 +/- 0.0042 | 0.0000 | 0.0031 | 0.4000 | 0.6470 | 0.8102 | 0.4 | 0.1018 | 0.0000 |

## Ablation Summary

| Task | Budget | Variant | Goodput | OpenOut | Coverage | AUROC | FPR95 |
|---|---:|---|---:|---:|---:|---:|---:|
| full-open | 0.30 | opensemcom_no_channel | 0.2170 | 0.0031 | 0.2176 | 0.9986 | 0.0010 |
| full-open | 0.30 | opensemcom_progressive | 0.2844 | 0.0024 | 0.2850 | 0.9968 | 0.0078 |
| full-open | 0.30 | opensemcom_receiver_only | 0.2188 | 0.0020 | 0.2192 | 0.9984 | 0.0016 |
| full-open | 0.30 | dino_detector | 0.2694 | 0.0064 | 0.2712 | 0.9921 | 0.0175 |
| full-open | 0.45 | opensemcom_no_channel | 0.3777 | 0.0030 | 0.3788 | 0.9986 | 0.0010 |
| full-open | 0.45 | opensemcom_progressive | 0.4246 | 0.0190 | 0.4328 | 0.9980 | 0.0037 |
| full-open | 0.45 | opensemcom_receiver_only | 0.3775 | 0.0029 | 0.3786 | 0.9984 | 0.0016 |
| full-open | 0.45 | ensemble_detector | 0.3748 | 0.0030 | 0.3759 | 0.9988 | 0.0005 |
| full-open | 0.60 | opensemcom_no_channel | 0.4228 | 0.0135 | 0.4286 | 0.9986 | 0.0010 |
| full-open | 0.60 | opensemcom_progressive | 0.4246 | 0.0190 | 0.4328 | 0.9980 | 0.0037 |
| full-open | 0.60 | opensemcom_receiver_only | 0.4219 | 0.0161 | 0.4288 | 0.9984 | 0.0016 |
| full-open | 0.60 | ensemble_detector | 0.4217 | 0.0165 | 0.4288 | 0.9988 | 0.0005 |
| full-open | 0.80 | opensemcom_no_channel | 0.4228 | 0.0135 | 0.4286 | 0.9986 | 0.0010 |
| full-open | 0.80 | opensemcom_progressive | 0.4246 | 0.0190 | 0.4328 | 0.9980 | 0.0037 |
| full-open | 0.80 | opensemcom_receiver_only | 0.4219 | 0.0161 | 0.4288 | 0.9984 | 0.0016 |
| full-open | 0.80 | ensemble_detector | 0.4217 | 0.0165 | 0.4288 | 0.9988 | 0.0005 |
| full-open | 1.00 | opensemcom_no_channel | 0.4228 | 0.0135 | 0.4286 | 0.9986 | 0.0010 |
| full-open | 1.00 | opensemcom_progressive | 0.4246 | 0.0190 | 0.4328 | 0.9980 | 0.0037 |
| full-open | 1.00 | opensemcom_receiver_only | 0.4219 | 0.0161 | 0.4288 | 0.9984 | 0.0016 |
| full-open | 1.00 | ensemble_detector | 0.4217 | 0.0165 | 0.4288 | 0.9988 | 0.0005 |
| deepsense-beam | 0.30 | opensemcom_no_channel | 0.0092 | 0.0000 | 0.0092 | 0.8305 | 0.5203 |
| deepsense-beam | 0.30 | opensemcom_progressive | 0.0046 | 0.0000 | 0.0046 | 0.8337 | 0.5722 |
| deepsense-beam | 0.30 | opensemcom_receiver_only | 0.0015 | 0.0000 | 0.0015 | 0.8455 | 0.4164 |
| deepsense-beam | 0.30 | dino_detector | 0.0061 | 0.0000 | 0.0061 | 0.8337 | 0.5722 |
| deepsense-beam | 0.45 | opensemcom_no_channel | 0.0092 | 0.0000 | 0.0092 | 0.8305 | 0.5203 |
| deepsense-beam | 0.45 | opensemcom_progressive | 0.0046 | 0.0000 | 0.0046 | 0.8337 | 0.5722 |
| deepsense-beam | 0.45 | opensemcom_receiver_only | 0.0015 | 0.0000 | 0.0015 | 0.8455 | 0.4164 |
| deepsense-beam | 0.45 | dino_detector | 0.0061 | 0.0000 | 0.0061 | 0.8337 | 0.5722 |
| deepsense-beam | 0.60 | opensemcom_no_channel | 0.2580 | 0.0134 | 0.2641 | 0.8305 | 0.5203 |
| deepsense-beam | 0.60 | opensemcom_progressive | 0.4031 | 0.0108 | 0.4092 | 0.8102 | 0.5487 |
| deepsense-beam | 0.60 | opensemcom_receiver_only | 0.2427 | 0.0032 | 0.2443 | 0.8455 | 0.4164 |
| deepsense-beam | 0.60 | ensemble_detector | 0.2901 | 0.0138 | 0.2962 | 0.8367 | 0.5601 |
| deepsense-beam | 0.80 | opensemcom_no_channel | 0.3115 | 0.0126 | 0.3176 | 0.8305 | 0.5203 |
| deepsense-beam | 0.80 | opensemcom_progressive | 0.5023 | 0.0185 | 0.5160 | 0.8194 | 0.5135 |
| deepsense-beam | 0.80 | opensemcom_receiver_only | 0.3008 | 0.0032 | 0.3023 | 0.8455 | 0.4164 |
| deepsense-beam | 0.80 | ensemble_detector | 0.3634 | 0.0118 | 0.3695 | 0.8367 | 0.5601 |
| deepsense-beam | 1.00 | opensemcom_no_channel | 0.3496 | 0.0183 | 0.3603 | 0.8305 | 0.5203 |
| deepsense-beam | 1.00 | opensemcom_progressive | 0.6489 | 0.0261 | 0.6687 | 0.8264 | 0.4997 |
| deepsense-beam | 1.00 | opensemcom_receiver_only | 0.3405 | 0.0135 | 0.3496 | 0.8455 | 0.4164 |
| deepsense-beam | 1.00 | ensemble_detector | 0.4076 | 0.0196 | 0.4198 | 0.8367 | 0.5601 |

## Route Selection at target 0.05

| Task | Budget | Route | Seeds |
|---|---:|---|---:|
| deepsense-beam | 0.30 | dino_core | 5 |
| deepsense-beam | 0.45 | dino_core | 5 |
| deepsense-beam | 0.60 | dino_channel_fusion_core | 1 |
| deepsense-beam | 0.60 | dino_core | 1 |
| deepsense-beam | 0.60 | ensemble_core | 1 |
| deepsense-beam | 0.60 | trained_receiver_dino_fusion_core | 2 |
| deepsense-beam | 0.80 | dino_core | 2 |
| deepsense-beam | 0.80 | trained_receiver_dino_fusion_core | 3 |
| deepsense-beam | 1.00 | dino_core | 2 |
| deepsense-beam | 1.00 | trained_receiver_dino_fusion_core | 3 |
| full-open | 0.30 | dino_channel_fusion_core | 4 |
| full-open | 0.30 | trained_receiver_core | 1 |
| full-open | 0.45 | trained_receiver_channel_fusion_core | 1 |
| full-open | 0.45 | trained_receiver_dino_fusion_core | 3 |
| full-open | 0.45 | trained_receiver_no_channel_core | 1 |
| full-open | 0.60 | trained_receiver_channel_fusion_core | 1 |
| full-open | 0.60 | trained_receiver_dino_fusion_core | 3 |
| full-open | 0.60 | trained_receiver_no_channel_core | 1 |
| full-open | 0.80 | trained_receiver_channel_fusion_core | 1 |
| full-open | 0.80 | trained_receiver_dino_fusion_core | 3 |
| full-open | 0.80 | trained_receiver_no_channel_core | 1 |
| full-open | 1.00 | trained_receiver_channel_fusion_core | 1 |
| full-open | 1.00 | trained_receiver_dino_fusion_core | 3 |
| full-open | 1.00 | trained_receiver_no_channel_core | 1 |

## What improved

- Full-open under tight resource is stronger: at budget 0.30 OpenSemCom goodput is 0.2844 versus the best comparison at 0.2694; at budget 0.45 it is 0.4246 versus 0.3748.
- At standard budgets 0.60-1.00, OpenSemCom remains the best full-open row, but the margin is modest: 0.4246 versus about 0.4217.
- DeepSense measured beam-sector improves materially at useful budgets: at budget 0.60 OpenSemCom reaches 0.4031 versus 0.2901; at 1.00 it reaches 0.6489 versus 0.4076.
- The trained receiver ablations show the learned head contributes, but the best result usually comes from the complete adaptive/refinement controller rather than receiver-only or no-channel rows.

## Remaining weakness

The full-open high-resource margin is still small, so the strongest next step is larger-scale receiver training or task-specific fine-tuning instead of frozen-feature heads. The wireless result is still beam-sector-level; exact beam prediction needs more DeepSense rows before it can support a strong beam-management claim.

## Artifacts

- Raw summary: `runs/comm_control_trained_receiver_curves_20260627_summary.csv`
- Raw policies: `runs/comm_control_trained_receiver_curves_20260627_policies.csv`
- Aggregate table: `results/final_opensemcom_comm_control_20260627.aggregate.csv`
- Headline 0.05 table: `results/final_opensemcom_comm_control_20260627.headline_005.csv`
- Ablation 0.05 table: `results/final_opensemcom_comm_control_20260627.ablation_005.csv`
- Compact risk-goodput table: `results/final_opensemcom_comm_control_20260627.curve_compact.csv`

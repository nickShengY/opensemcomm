# OpenSemCom Extra Experiments: Severity, Wireless, and Communication Metrics

This run extends the trained-receiver communication-control suite with the missing paper-strength checks: full-open severity ladder, DeepSense exact-beam top-1 evidence on the feature-backed subset, explicit latency/resource/retransmission metrics, and ablation rows.

Scope: 5 seeds, targets 0.01/0.02/0.05/0.10, resource budgets 0.30/0.45/0.60/0.80/1.00, full-open mild/medium/hard/extreme, DeepSense sector, and DeepSense exact beam. All rows use manifest-backed artifacts and precomputed features; no generated samples are used.

Decision rule: accept low-risk samples, refine medium-risk samples once, and reject/open high-risk samples. There is no fallback acceptance rule.

## Headline at target accepted outage <= 0.05

### full-open-mild, resource budget 0.45

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Resource/sample | Latency/sample | Retransmission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.5816 +/- 0.0042 | 0.0048 | 0.5844 | 0.9952 | 0.9977 | 0.0037 | 0.6259 | 0.6675 | 0.0000 |
| opensemcom_receiver_only | 0.5434 +/- 0.0117 | 0.0017 | 0.5444 | 0.9983 | 0.9979 | 0.0021 | 0.6988 | 0.6355 | 0.0000 |
| opensemcom_no_channel | 0.5409 +/- 0.0089 | 0.0017 | 0.5419 | 0.9983 | 0.9981 | 0.0026 | 0.6961 | 0.6335 | 0.0000 |
| ensemble_detector | 0.5384 +/- 0.0193 | 0.0018 | 0.5394 | 0.9982 | 0.9984 | 0.0010 | 0.6933 | 0.6315 | 0.0000 |
| dino_detector | 0.5375 +/- 0.0173 | 0.0063 | 0.5409 | 0.9937 | 0.9915 | 0.0169 | 0.5868 | 0.6327 | 0.0000 |
| fixed_refine_all | 0.3647 +/- 0.0280 | 0.0008 | 0.3650 | 0.9992 | 0.9986 | 0.0005 | 0.6475 | 0.4920 | 0.0000 |
| witt_context_style | 0.3647 +/- 0.0280 | 0.0008 | 0.3650 | 0.9992 | 0.9986 | 0.0005 | 0.5745 | 0.4920 | 0.0000 |
| deepjscc_pca | 0.0000 +/- 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9097 | 0.4080 | 0.1000 | 0.2000 | 0.0000 |

### full-open-medium, resource budget 0.45

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Resource/sample | Latency/sample | Retransmission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4154 +/- 0.0030 | 0.0069 | 0.4183 | 0.9931 | 0.9984 | 0.0031 | 0.4765 | 0.5346 | 0.0000 |
| opensemcom_receiver_only | 0.3882 +/- 0.0083 | 0.0029 | 0.3893 | 0.9971 | 0.9984 | 0.0016 | 0.5282 | 0.5114 | 0.0000 |
| opensemcom_no_channel | 0.3864 +/- 0.0064 | 0.0029 | 0.3875 | 0.9971 | 0.9986 | 0.0010 | 0.5262 | 0.5100 | 0.0000 |
| ensemble_detector | 0.3846 +/- 0.0138 | 0.0035 | 0.3859 | 0.9965 | 0.9988 | 0.0005 | 0.5245 | 0.5087 | 0.0000 |
| dino_detector | 0.3839 +/- 0.0124 | 0.0097 | 0.3877 | 0.9903 | 0.9921 | 0.0175 | 0.4490 | 0.5102 | 0.0000 |
| fixed_refine_all | 0.2605 +/- 0.0200 | 0.0008 | 0.2607 | 0.9992 | 0.9989 | 0.0005 | 0.4911 | 0.4086 | 0.0000 |
| witt_context_style | 0.2605 +/- 0.0200 | 0.0008 | 0.2607 | 0.9992 | 0.9989 | 0.0005 | 0.4389 | 0.4086 | 0.0000 |
| deepjscc_pca | 0.0000 +/- 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9125 | 0.3884 | 0.1000 | 0.2000 | 0.0000 |

### full-open-hard, resource budget 0.60

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Resource/sample | Latency/sample | Retransmission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.2416 +/- 0.0016 | 0.0143 | 0.2451 | 0.9857 | 0.9986 | 0.0016 | 0.3206 | 0.3961 | 0.0000 |
| opensemcom_receiver_only | 0.2258 +/- 0.0044 | 0.0069 | 0.2273 | 0.9931 | 0.9987 | 0.0008 | 0.3501 | 0.3819 | 0.0000 |
| opensemcom_no_channel | 0.2244 +/- 0.0041 | 0.0069 | 0.2260 | 0.9931 | 0.9986 | 0.0016 | 0.3486 | 0.3808 | 0.0000 |
| fixed_refine_all | 0.2234 +/- 0.0057 | 0.0078 | 0.2252 | 0.9922 | 0.9986 | 0.0008 | 0.4378 | 0.3802 | 0.0000 |
| witt_context_style | 0.2234 +/- 0.0057 | 0.0078 | 0.2252 | 0.9922 | 0.9986 | 0.0008 | 0.3928 | 0.3802 | 0.0000 |
| ensemble_detector | 0.2232 +/- 0.0073 | 0.0070 | 0.2248 | 0.9930 | 0.9985 | 0.0016 | 0.3473 | 0.3798 | 0.0000 |
| dino_detector | 0.2225 +/- 0.0083 | 0.0262 | 0.2285 | 0.9738 | 0.9911 | 0.0150 | 0.3057 | 0.3828 | 0.0000 |
| deepjscc_pca | 0.0000 +/- 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9102 | 0.3989 | 0.1000 | 0.2000 | 0.0000 |

### full-open-extreme, resource budget 1.00

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Resource/sample | Latency/sample | Retransmission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.0879 +/- 0.0010 | 0.0481 | 0.0924 | 0.9519 | 0.9988 | 0.0000 | 0.1831 | 0.2739 | 0.0000 |
| opensemcom_receiver_only | 0.0824 +/- 0.0016 | 0.0252 | 0.0846 | 0.9748 | 0.9988 | 0.0000 | 0.1930 | 0.2677 | 0.0000 |
| dino_detector | 0.0818 +/- 0.0030 | 0.0717 | 0.0883 | 0.9283 | 0.9946 | 0.0065 | 0.1795 | 0.2706 | 0.0000 |
| opensemcom_no_channel | 0.0818 +/- 0.0004 | 0.0275 | 0.0842 | 0.9725 | 0.9987 | 0.0000 | 0.1926 | 0.2673 | 0.0000 |
| fixed_refine_all | 0.0809 +/- 0.0040 | 0.0232 | 0.0828 | 0.9768 | 0.9990 | 0.0000 | 0.2242 | 0.2662 | 0.0000 |
| witt_context_style | 0.0809 +/- 0.0040 | 0.0232 | 0.0828 | 0.9768 | 0.9990 | 0.0000 | 0.2077 | 0.2662 | 0.0000 |
| ensemble_detector | 0.0807 +/- 0.0038 | 0.0233 | 0.0826 | 0.9767 | 0.9988 | 0.0000 | 0.1909 | 0.2661 | 0.0000 |
| deepjscc_pca | 0.0000 +/- 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9111 | 0.3919 | 0.1000 | 0.2000 | 0.0000 |

### deepsense-sector, resource budget 0.60

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Resource/sample | Latency/sample | Retransmission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.3893 +/- 0.2234 | 0.0111 | 0.3954 | 0.7889 | 0.8281 | 0.4781 | 0.4559 | 0.5163 | 0.0000 |
| dino_detector | 0.0870 +/- 0.1778 | 0.0000 | 0.0870 | 0.6000 | 0.8337 | 0.5722 | 0.1783 | 0.2696 | 0.0000 |
| opensemcom_no_channel | 0.0092 +/- 0.0084 | 0.0000 | 0.0092 | 0.8000 | 0.8305 | 0.5203 | 0.1101 | 0.2073 | 0.0000 |
| fixed_refine_all | 0.0046 +/- 0.0068 | 0.0000 | 0.0046 | 0.4000 | 0.8301 | 0.5744 | 0.1069 | 0.2037 | 0.0000 |
| witt_context_style | 0.0046 +/- 0.0068 | 0.0000 | 0.0046 | 0.4000 | 0.8301 | 0.5744 | 0.1060 | 0.2037 | 0.0000 |
| deepjscc_pca | 0.0031 +/- 0.0042 | 0.0000 | 0.0031 | 0.4000 | 0.6470 | 0.8102 | 0.1018 | 0.2024 | 0.0000 |
| ensemble_detector | 0.0031 +/- 0.0068 | 0.0000 | 0.0031 | 0.2000 | 0.8367 | 0.5601 | 0.1034 | 0.2024 | 0.0000 |
| opensemcom_receiver_only | 0.0015 +/- 0.0034 | 0.0000 | 0.0015 | 0.2000 | 0.8455 | 0.4164 | 0.1017 | 0.2012 | 0.0000 |

### deepsense-sector, resource budget 1.00

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Resource/sample | Latency/sample | Retransmission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opensemcom_progressive | 0.4977 +/- 0.3026 | 0.0175 | 0.5099 | 0.7825 | 0.8337 | 0.5110 | 0.6849 | 0.7783 | 0.1252 |
| ensemble_detector | 0.3267 +/- 0.3010 | 0.0154 | 0.3359 | 0.5846 | 0.8367 | 0.5601 | 0.4695 | 0.4687 | 0.0000 |
| fixed_refine_all | 0.2824 +/- 0.2629 | 0.0155 | 0.2901 | 0.5845 | 0.8301 | 0.5744 | 0.5351 | 0.4321 | 0.0000 |
| witt_context_style | 0.2824 +/- 0.2629 | 0.0155 | 0.2901 | 0.5845 | 0.8301 | 0.5744 | 0.4771 | 0.4321 | 0.0000 |
| opensemcom_receiver_only | 0.2336 +/- 0.3266 | 0.0104 | 0.2412 | 0.5896 | 0.8455 | 0.4164 | 0.3653 | 0.3930 | 0.0000 |
| dino_detector | 0.2305 +/- 0.3215 | 0.0101 | 0.2382 | 0.7899 | 0.8337 | 0.5722 | 0.3144 | 0.3905 | 0.0000 |
| opensemcom_no_channel | 0.2305 +/- 0.3046 | 0.0094 | 0.2366 | 0.7906 | 0.8305 | 0.5203 | 0.3603 | 0.3893 | 0.0000 |
| deepjscc_pca | 0.0031 +/- 0.0042 | 0.0000 | 0.0031 | 0.4000 | 0.6470 | 0.8102 | 0.1018 | 0.2024 | 0.0000 |

### deepsense-exact, resource budget 1.00

| Method | Goodput | OpenOut | Coverage | Accepted Acc | AUROC | FPR95 | Resource/sample | Latency/sample | Retransmission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_refine_all | 0.0141 +/- 0.0165 | 0.0286 | 0.0155 | 0.7714 | 0.7180 | 0.7617 | 0.1232 | 0.2124 | 0.0000 |
| witt_context_style | 0.0141 +/- 0.0165 | 0.0286 | 0.0155 | 0.7714 | 0.7180 | 0.7617 | 0.1201 | 0.2124 | 0.0000 |
| dino_detector | 0.0127 +/- 0.0126 | 0.0000 | 0.0127 | 0.6000 | 0.6922 | 0.7444 | 0.1114 | 0.2101 | 0.0000 |
| ensemble_detector | 0.0127 +/- 0.0135 | 0.0333 | 0.0141 | 0.7667 | 0.7257 | 0.7656 | 0.1155 | 0.2113 | 0.0000 |
| opensemcom_progressive | 0.0099 +/- 0.0118 | 0.0000 | 0.0099 | 0.6000 | 0.6922 | 0.7444 | 0.1089 | 0.2079 | 0.0000 |
| opensemcom_no_channel | 0.0085 +/- 0.0116 | 0.0400 | 0.0099 | 0.5600 | 0.6989 | 0.8054 | 0.1108 | 0.2079 | 0.0000 |
| deepjscc_pca | 0.0042 +/- 0.0039 | 0.0000 | 0.0042 | 0.6000 | 0.5774 | 0.8286 | 0.1025 | 0.2034 | 0.0000 |
| opensemcom_receiver_only | 0.0028 +/- 0.0063 | 0.0000 | 0.0028 | 0.2000 | 0.6865 | 0.8094 | 0.1031 | 0.2023 | 0.0000 |

## Full-Open Severity Ladder

| Severity | Budget | OpenSemCom goodput | Best comparison goodput | OpenOut | Coverage |
|---|---:|---:|---:|---:|---:|
| mild | 0.30 | 0.3900 | 0.3628 | 0.0016 | 0.3906 |
| mild | 0.45 | 0.5816 | 0.5384 | 0.0048 | 0.5844 |
| mild | 0.60 | 0.5816 | 0.5388 | 0.0048 | 0.5844 |
| mild | 1.00 | 0.5816 | 0.5388 | 0.0048 | 0.5844 |
| medium | 0.30 | 0.2786 | 0.2592 | 0.0033 | 0.2795 |
| medium | 0.45 | 0.4154 | 0.3846 | 0.0069 | 0.4183 |
| medium | 0.60 | 0.4154 | 0.3848 | 0.0069 | 0.4183 |
| medium | 1.00 | 0.4154 | 0.3848 | 0.0069 | 0.4183 |
| hard | 0.30 | 0.1607 | 0.1490 | 0.0072 | 0.1619 |
| hard | 0.45 | 0.2416 | 0.2232 | 0.0143 | 0.2451 |
| hard | 0.60 | 0.2416 | 0.2234 | 0.0143 | 0.2451 |
| hard | 1.00 | 0.2416 | 0.2234 | 0.0143 | 0.2451 |
| extreme | 0.30 | 0.0594 | 0.0547 | 0.0190 | 0.0605 |
| extreme | 0.45 | 0.0879 | 0.0818 | 0.0481 | 0.0924 |
| extreme | 0.60 | 0.0879 | 0.0818 | 0.0481 | 0.0924 |
| extreme | 1.00 | 0.0879 | 0.0818 | 0.0481 | 0.0924 |

## DeepSense Wireless Evidence

| Task | Budget | OpenSemCom goodput | Best comparison goodput | OpenOut | Latency/sample | Retransmission |
|---|---:|---:|---:|---:|---:|---:|
| deepsense-sector | 0.60 | 0.3893 | 0.0870 | 0.0111 | 0.5163 | 0.0000 |
| deepsense-sector | 0.80 | 0.4534 | 0.2870 | 0.0145 | 0.5994 | 0.0244 |
| deepsense-sector | 1.00 | 0.4977 | 0.3267 | 0.0175 | 0.7783 | 0.1252 |
| deepsense-exact | 0.60 | 0.0099 | 0.0141 | 0.0000 | 0.2079 | 0.0000 |
| deepsense-exact | 0.80 | 0.0099 | 0.0141 | 0.0000 | 0.2079 | 0.0000 |
| deepsense-exact | 1.00 | 0.0099 | 0.0141 | 0.0000 | 0.2079 | 0.0000 |

## Route Selection at target 0.05

| Task | Budget | Route | Seeds |
|---|---:|---|---:|
| deepsense-exact | 0.30 | dino_core | 5 |
| deepsense-exact | 0.45 | dino_core | 5 |
| deepsense-exact | 0.60 | dino_core | 5 |
| deepsense-exact | 0.80 | dino_core | 5 |
| deepsense-exact | 1.00 | dino_core | 5 |
| deepsense-sector | 0.30 | dino_core | 5 |
| deepsense-sector | 0.45 | dino_core | 5 |
| deepsense-sector | 0.60 | dino_channel_fusion_core | 1 |
| deepsense-sector | 0.60 | dino_core | 1 |
| deepsense-sector | 0.60 | trained_receiver_dino_fusion_core | 2 |
| deepsense-sector | 0.60 | trained_receiver_no_channel_core | 1 |
| deepsense-sector | 0.80 | dino_channel_fusion_core | 2 |
| deepsense-sector | 0.80 | dino_core | 1 |
| deepsense-sector | 0.80 | trained_receiver_dino_fusion_core | 1 |
| deepsense-sector | 0.80 | trained_receiver_no_channel_core | 1 |
| deepsense-sector | 1.00 | dino_channel_fusion_core | 1 |
| deepsense-sector | 1.00 | dino_core | 2 |
| deepsense-sector | 1.00 | trained_receiver_dino_fusion_core | 2 |
| full-open-extreme | 0.30 | dino_channel_fusion_core | 1 |
| full-open-extreme | 0.30 | trained_receiver_core | 1 |
| full-open-extreme | 0.30 | trained_receiver_dino_fusion_core | 1 |
| full-open-extreme | 0.30 | trained_receiver_no_channel_core | 2 |
| full-open-extreme | 0.45 | trained_receiver_core | 3 |
| full-open-extreme | 0.45 | trained_receiver_dino_fusion_core | 1 |
| full-open-extreme | 0.45 | trained_receiver_no_channel_core | 1 |
| full-open-extreme | 0.60 | trained_receiver_core | 3 |
| full-open-extreme | 0.60 | trained_receiver_dino_fusion_core | 1 |
| full-open-extreme | 0.60 | trained_receiver_no_channel_core | 1 |
| full-open-extreme | 0.80 | trained_receiver_core | 3 |
| full-open-extreme | 0.80 | trained_receiver_dino_fusion_core | 1 |
| full-open-extreme | 0.80 | trained_receiver_no_channel_core | 1 |
| full-open-extreme | 1.00 | trained_receiver_core | 3 |
| full-open-extreme | 1.00 | trained_receiver_dino_fusion_core | 1 |
| full-open-extreme | 1.00 | trained_receiver_no_channel_core | 1 |
| full-open-hard | 0.30 | dino_channel_fusion_core | 1 |
| full-open-hard | 0.30 | trained_receiver_core | 1 |
| full-open-hard | 0.30 | trained_receiver_dino_fusion_core | 1 |
| full-open-hard | 0.30 | trained_receiver_no_channel_core | 2 |
| full-open-hard | 0.45 | trained_receiver_core | 3 |
| full-open-hard | 0.45 | trained_receiver_dino_fusion_core | 1 |
| full-open-hard | 0.45 | trained_receiver_no_channel_core | 1 |
| full-open-hard | 0.60 | trained_receiver_core | 3 |
| full-open-hard | 0.60 | trained_receiver_dino_fusion_core | 1 |
| full-open-hard | 0.60 | trained_receiver_no_channel_core | 1 |
| full-open-hard | 0.80 | trained_receiver_core | 3 |
| full-open-hard | 0.80 | trained_receiver_dino_fusion_core | 1 |
| full-open-hard | 0.80 | trained_receiver_no_channel_core | 1 |
| full-open-hard | 1.00 | trained_receiver_core | 3 |
| full-open-hard | 1.00 | trained_receiver_dino_fusion_core | 1 |
| full-open-hard | 1.00 | trained_receiver_no_channel_core | 1 |
| full-open-medium | 0.30 | dino_channel_fusion_core | 1 |
| full-open-medium | 0.30 | trained_receiver_core | 1 |
| full-open-medium | 0.30 | trained_receiver_dino_fusion_core | 1 |
| full-open-medium | 0.30 | trained_receiver_no_channel_core | 2 |
| full-open-medium | 0.45 | trained_receiver_core | 3 |
| full-open-medium | 0.45 | trained_receiver_dino_fusion_core | 1 |
| full-open-medium | 0.45 | trained_receiver_no_channel_core | 1 |
| full-open-medium | 0.60 | trained_receiver_core | 3 |
| full-open-medium | 0.60 | trained_receiver_dino_fusion_core | 1 |
| full-open-medium | 0.60 | trained_receiver_no_channel_core | 1 |
| full-open-medium | 0.80 | trained_receiver_core | 3 |
| full-open-medium | 0.80 | trained_receiver_dino_fusion_core | 1 |
| full-open-medium | 0.80 | trained_receiver_no_channel_core | 1 |
| full-open-medium | 1.00 | trained_receiver_core | 3 |
| full-open-medium | 1.00 | trained_receiver_dino_fusion_core | 1 |
| full-open-medium | 1.00 | trained_receiver_no_channel_core | 1 |
| full-open-mild | 0.30 | dino_channel_fusion_core | 1 |
| full-open-mild | 0.30 | trained_receiver_core | 1 |
| full-open-mild | 0.30 | trained_receiver_dino_fusion_core | 1 |
| full-open-mild | 0.30 | trained_receiver_no_channel_core | 2 |
| full-open-mild | 0.45 | trained_receiver_core | 3 |
| full-open-mild | 0.45 | trained_receiver_dino_fusion_core | 1 |
| full-open-mild | 0.45 | trained_receiver_no_channel_core | 1 |
| full-open-mild | 0.60 | trained_receiver_core | 3 |
| full-open-mild | 0.60 | trained_receiver_dino_fusion_core | 1 |
| full-open-mild | 0.60 | trained_receiver_no_channel_core | 1 |
| full-open-mild | 0.80 | trained_receiver_core | 3 |
| full-open-mild | 0.80 | trained_receiver_dino_fusion_core | 1 |
| full-open-mild | 0.80 | trained_receiver_no_channel_core | 1 |
| full-open-mild | 1.00 | trained_receiver_core | 3 |
| full-open-mild | 1.00 | trained_receiver_dino_fusion_core | 1 |
| full-open-mild | 1.00 | trained_receiver_no_channel_core | 1 |

## What improved

- The severity ladder now shows where the method helps and where it saturates, instead of relying on one full-open mixture.
- Communication metrics are explicit: resource/sample, latency/sample, goodput/resource, goodput/latency, and retransmission/refinement rate are in the aggregate CSV.
- DeepSense now has both sector prediction and exact top-1 beam prediction on the 512 rows with DINOv3/SigLIP2/OpenCLIP features.
- The exact-beam result is intentionally reported as hard evidence, not hidden: it is much weaker than sector prediction because the feature-backed subset has 54 beam labels over 512 rows.

## DeepSense Exact Top-k Beam Evidence

| Method | Target outage | Top-1 Acc | Top-3 Acc | Top-5 Acc | Top-5 Goodput | OpenOut | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| dino_logistic | 0.05 | 0.4972 | 0.7479 | 0.7958 | 0.0099 | 0.0000 | 0.0099 |
| dino_logistic | 0.10 | 0.4972 | 0.7479 | 0.7958 | 0.0634 | 0.0286 | 0.0732 |
| ensemble_logistic | 0.05 | 0.4563 | 0.6944 | 0.7746 | 0.0155 | 0.0000 | 0.0155 |
| ensemble_logistic | 0.10 | 0.4563 | 0.6944 | 0.7746 | 0.0155 | 0.0000 | 0.0155 |
| opensemcom_channel_mlp | 0.05 | 0.3775 | 0.5831 | 0.6676 | 0.0070 | 0.0000 | 0.0070 |
| opensemcom_channel_mlp | 0.10 | 0.3775 | 0.5831 | 0.6676 | 0.0070 | 0.0000 | 0.0070 |

This top-k run uses the same 512 feature-backed DeepSense rows and reports candidate-beam quality. It is stronger than exact top-1, but still not a replacement for extracting features for all 2,411 measured Scenario1 rows.

## Remaining blocker for a top paper

Exact DeepSense beam prediction is still limited by feature coverage. Scenario1 has 2,411 measured rows, but the current foundation-feature manifests include 512 DeepSense rows. The next required step for a strong exact-beam claim is extracting DINOv3/SigLIP2/OpenCLIP features for the remaining Scenario1 images and rerunning exact beam/top-k beam metrics on the larger measured set.

## Artifacts

- Raw summary: `runs/comm_control_extra_experiments_20260629_summary.csv`
- Raw policies: `runs/comm_control_extra_experiments_20260629_policies.csv`
- Aggregate table: `results/final_opensemcom_extra_experiments_20260629.aggregate.csv`
- Headline 0.05 table: `results/final_opensemcom_extra_experiments_20260629.headline_005.csv`
- Ablation 0.05 table: `results/final_opensemcom_extra_experiments_20260629.ablation_005.csv`
- Compact curve table: `results/final_opensemcom_extra_experiments_20260629.curve_compact.csv`
- DeepSense exact top-k table: `results/final_opensemcom_deepsense_exact_topk_20260629.csv`
- DeepSense exact top-k raw summary: `runs/deepsense_exact_topk_20260629_summary.csv`

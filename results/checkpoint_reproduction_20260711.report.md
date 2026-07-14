# Checkpointed Communication-Control Reproduction

## Outcome

The five-seed communication-control suite was trained again on July 11, 2026 with checkpoint persistence enabled. It produced 30 task/seed model bundles and exactly reproduced the previously published summary, policy selections, and manifest summary.

| Artifact | Earlier run | Checkpointed rerun | Verification |
|---|---|---|---|
| Summary rows | 4,800 | 4,800 | Byte-for-byte identical |
| Policy rows | 4,800 | 4,800 | Byte-for-byte identical |
| Summary SHA-256 | `e839c1ee...b1f4931` | `e839c1ee...b1f4931` | Match |
| Policies SHA-256 | `13834cc0...ae1a2` | `13834cc0...ae1a2` | Match |
| Manifest summary | June 29 file | July 11 file | Byte-for-byte identical |
| Task/seed bundles | Not saved | 30 | All loaded successfully |
| Indexed checkpoint files | Not saved | 121 | Size and SHA-256 verified |
| Checkpoint bytes | 0 | 204,997,018 | Indexed |

The matching files are:

```text
runs/comm_control_extra_experiments_20260629_summary.csv
runs/comm_control_checkpointed_20260711_summary.csv

runs/comm_control_extra_experiments_20260629_policies.csv
runs/comm_control_checkpointed_20260711_policies.csv
```

Because the rerun output is identical, the headline result remains:

| Condition | Open exposure | OpenSemCom goodput | Accepted outage | Coverage | Accepted accuracy | Best comparison goodput |
|---|---:|---:|---:|---:|---:|---:|
| Mild | 25% | 0.5816 | 0.0048 | 0.5844 | 0.9952 | 0.5384 |
| Medium | 50% | 0.4154 | 0.0069 | 0.4183 | 0.9931 | 0.3846 |
| Hard | 75% | 0.2416 | 0.0143 | 0.2451 | 0.9857 | 0.2234 |
| Extreme | 91% | 0.0879 | 0.0481 | 0.0924 | 0.9519 | 0.0818 |

## Saved State

For every severity/task and seed, the rerun saves:

- the channel-aware OpenSemCom receiver encoder and its class, unsafe, and acceptance heads;
- the receiver normalization mean and standard deviation;
- the receiver architecture and training hyperparameters;
- the corresponding receiver trained without channel inputs;
- fitted DINOv3, ensemble, channel-aware ensemble, and DeepJSCC-style classical components;
- exact train/calibration/evaluation split-key hashes;
- feature-manifest paths, sizes, and SHA-256 values;
- per-file sizes and SHA-256 values.

The root inventory is `models/checkpoints/communication_control_20260711/checkpoint_index.json`. The human-readable loading guide is beside it in `README.md`.

## Verification Performed

1. Loaded all 60 PyTorch receiver files with safe tensor-only checkpoint loading.
2. Loaded all 30 classical joblib bundles and verified the expected four components.
3. Recomputed all 121 hashes in the checkpoint index.
4. Compared the new and old summary CSV files with `cmp` and SHA-256.
5. Compared the new and old policy CSV files with `cmp` and SHA-256.
6. Compared the new and old manifest summaries byte-for-byte.
7. Ran the complete Python test suite after adding checkpoint round-trip tests.

## Scope

The archive contains project-trained heads and fitted comparison components. It does not contain pretrained foundation-model weights, feature arrays, or dataset files. Those inputs are downloaded or extracted under scratch storage and remain governed by their original licenses.

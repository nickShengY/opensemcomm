# OpenSemCom Communication-Control Checkpoints

This directory contains the project-trained model state from the five-seed communication-control rerun completed on July 11, 2026. The rerun uses the same manifests, split construction, severity ladder, safety targets, resource budgets, and training configuration as the June 29 result.

The regenerated summary and selected-policy CSV files are byte-for-byte identical to the published files:

```text
summary SHA-256:  e839c1ee9a8cbc167750fde1726596f2df8836bcea0d678488d47dc21b1f4931
policies SHA-256: 13834cc043bb3dbc6b1dcff8e634e2ce9f942570fd4adf25673b9a95783ae1a2
```

## Contents

There are 30 task/seed bundles:

```text
full-open-mild/{seed_0,...,seed_4}
full-open-medium/{seed_0,...,seed_4}
full-open-hard/{seed_0,...,seed_4}
full-open-extreme/{seed_0,...,seed_4}
deepsense-sector/{seed_0,...,seed_4}
deepsense-exact/{seed_0,...,seed_4}
```

Each bundle contains:

| File | Contents |
|---|---|
| `receiver_channel.pt` | Trained OpenSemCom encoder, class head, unsafe head, accept head, normalization state, architecture, and training parameters using measured channel context |
| `receiver_no_channel.pt` | Corresponding receiver trained without channel-context inputs for ablation/comparison |
| `classical_models.joblib` | Fitted DINOv3, feature-ensemble, channel-aware ensemble, and DeepJSCC-style PCA/classifier components |
| `bundle.json` | Task, seed, split counts, split-key hashes, feature-manifest hashes, model sizes, and model SHA-256 values |

`checkpoint_index.json` inventories every bundle and file. It records the complete run configuration, software versions, feature-manifest provenance, model hashes, and total checkpoint bytes.

## Load a Receiver

Run from the repository root with the project environment active:

```python
from pathlib import Path

from opensemcom.cli.communication_control_suite import TrainedReceiver

path = Path(
    "models/checkpoints/communication_control_20260711/"
    "full-open-hard/seed_0/receiver_channel.pt"
)
receiver = TrainedReceiver.load_checkpoint(path, device="cpu")
scores = receiver.score(feature_matrix)
```

Use `device="cuda"` to load onto an available GPU. `feature_matrix` must use the same feature ordering and channel-context construction recorded by the suite.

## Load Classical Components

```python
from pathlib import Path

from opensemcom.cli.communication_control_suite import load_classical_checkpoint

path = Path(
    "models/checkpoints/communication_control_20260711/"
    "full-open-hard/seed_0/classical_models.joblib"
)
models = load_classical_checkpoint(path)
dino_model = models["dino"]
```

Joblib files use Python object serialization. Only load checkpoint files from this repository or another trusted source, and verify them against `checkpoint_index.json` first.

## What Is Not Included

These are trained task heads and fitted experiment components. Pretrained DINOv3, SigLIP2, and OpenCLIP DFN5B backbone weights are not duplicated here. They remain scratch-local downloads and must be obtained from their original model sources using the repository setup instructions.

Feature arrays and dataset files are also excluded. Their paths and manifest hashes are retained for provenance, but applicable dataset and model licenses still govern access.

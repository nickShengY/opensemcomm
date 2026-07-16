# OpenSemCom

OpenSemCom is an experimental semantic communication receiver for deployment conditions that can differ from training. It combines foundation-model image representations, a learned receiver and risk score, calibrated accept/refine/reject control, measured wireless context, and resource-aware evaluation.

The practical question is simple:

> At the same limit on unsafe accepted decisions, can the receiver deliver more correct semantic decisions than strong open-set alternatives?

OpenSemCom does not accept every prediction. It estimates the risk of each decision and uses two calibrated thresholds:

```text
risk <= q1          accept now
q1 < risk <= q2     request semantic refinement, then decide again
risk > q2           reject as open or unsafe
```

The current implementation never turns a high-risk rejection into an acceptance through a separate recovery rule. Historical filenames may contain terms from earlier development; those names are archival run identifiers and are not names for the current method.

## Current Result in Plain Language

The latest five-seed communication-control experiment evaluates four levels of full-open exposure. At an accepted-outage target of 5%, the best OpenSemCom operating point produced:

| Condition | Open exposure | Goodput | Accepted outage | Coverage | Accepted accuracy | Best comparison goodput |
|---|---:|---:|---:|---:|---:|---:|
| Mild | 25% | 0.5816 | 0.0048 | 0.5844 | 0.9952 | 0.5384 |
| Medium | 50% | 0.4154 | 0.0069 | 0.4183 | 0.9931 | 0.3846 |
| Hard | 75% | 0.2416 | 0.0143 | 0.2451 | 0.9857 | 0.2234 |
| Extreme | 91% | 0.0879 | 0.0481 | 0.0924 | 0.9519 | 0.0818 |

Goodput is the fraction of all evaluated samples that were both accepted and semantically correct. Coverage is the fraction accepted, whether correct or incorrect. Accepted outage is the error fraction among accepted samples. Therefore, high accuracy with tiny coverage is not enough: the central result is the goodput obtained while staying below the safety target.

These figures come from [the latest experiment report](results/final_opensemcom_extra_experiments_20260629.report.md). The report also contains the 1%, 2%, 5%, and 10% risk-goodput operating points, resource budgets, ablations, DeepSense beam-sector results, and exact-beam top-k results.

The complete suite was trained again with checkpoint persistence on July 11, 2026. The regenerated summaries and selected policies are byte-for-byte identical to the published run. All 30 task/seed bundles are versioned under `models/checkpoints/communication_control_20260711/`; see [the reproduction report](results/checkpoint_reproduction_20260711.report.md) for hashes and verification details.

The measured DeepSense evidence is useful but not complete. Scenario 1 contains 2,411 camera/mmWave/GPS rows, while the common DINOv3, SigLIP2, and OpenCLIP feature set currently covers 512 of those rows. Exact top-5 beam accuracy reaches 0.7958 with the DINOv3 logistic model, but safety-calibrated exact-beam goodput is still low. Expanding foundation-feature coverage to all 2,411 rows is the next required wireless experiment.

## What the System Contains

- DINOv3 as the primary image representation.
- SigLIP2 and OpenCLIP DFN5B representations for ensemble and comparison experiments.
- A trained receiver and risk head using image features, task/domain information, semantic-layer information, and measured DeepSense channel metadata.
- Open-set comparisons including one-vs-rest, MSP, energy, Mahalanobis, ViM, ReAct, and ASH in the full experiment archive.
- Communication comparisons using DeepJSCC-style and WITT-context-style controls.
- Mixed calibration using known and open-exposure rows.
- Accept, semantic refinement/HARQ, and reject/open actions.
- Full-open severity levels with 25%, 50%, 75%, and 91% open exposure.
- Risk-goodput targets at accepted outage limits of 0.01, 0.02, 0.05, and 0.10.
- Measured DeepSense 6G Scenario 1 camera, mmWave, GPS, and beam labels.
- Slurm scripts for feature extraction, training, evaluation, and aggregation.
- Strict manifest validation that fails when a row references a missing or empty artifact.

## Data Integrity Rules

Every experiment starts from a CSV manifest. Each row must resolve to a non-empty source artifact or precomputed feature derived from one. Required fields are:

```csv
source_path,label,task,domain,is_unknown,split,regime
```

The pipeline has no generated-sample path and no placeholder-data path. A run must fail when its manifest is absent, malformed, or points to missing files. Dataset licenses still apply: datasets and downloaded foundation-model weights remain in scratch storage and are not committed to GitHub. Project-trained heads are versioned separately under `models/checkpoints/`.

Validate a manifest before any training or evaluation:

```bash
source scripts/env_scratch.sh
python -m opensemcom.cli.validate_manifest manifests/opensemcom_real.csv
pytest
```

Feature manifests retain the semantic metadata of the source row and point `source_path` to a `.npy` or `.npz` feature artifact. They may also retain `raw_source_path` so the feature can be traced back to the source image.

## Scratch-Only Setup on Trillium

The expected project root is:

```text
/home/nickyun/links/scratch/new_study/opensemcom
```

Start every setup, download, preprocessing, training, or evaluation shell with:

```bash
cd /home/nickyun/links/scratch/new_study/opensemcom
source scripts/env_scratch.sh
```

The environment script creates scratch directories and exports:

```text
HF_HOME
HF_DATASETS_CACHE
TRANSFORMERS_CACHE
TORCH_HOME
XDG_CACHE_HOME
WANDB_DIR
TMPDIR
```

All point under the project scratch tree. Dataset archives, extracted datasets, model weights, feature arrays, checkpoints, logs, and experiment outputs must remain there. Do not place them in `$HOME`.

## Installation

Dependency installation and downloads are login-node-safe operations:

```bash
cd /home/nickyun/links/scratch/new_study/opensemcom
source scripts/env_scratch.sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

Foundation-model extraction requires the corresponding PyTorch, Transformers, timm/open-clip dependencies used by the selected extractor. Install them into the scratch-local virtual environment, then submit extraction through Slurm.

## Reproduce the Latest Suite

The complete command, input mapping, output mapping, and cluster workflow are documented in [EXPERIMENTS.md](EXPERIMENTS.md). The core command is:

```bash
python -m opensemcom.cli.communication_control_suite \
  --feature-manifest dino=manifests/opensemcom_real_dinov3_mixed_open_calibration.csv \
  --feature-manifest siglip2=manifests/opensemcom_real_siglip2_base.csv \
  --feature-manifest openclip=manifests/opensemcom_real_openclip_dfn5b.csv \
  --output-prefix runs/comm_control_extra_experiments_20260629 \
  --checkpoint-dir models/checkpoints/communication_control_20260711 \
  --seeds 0,1,2,3,4 \
  --targets 0.01,0.02,0.05,0.10 \
  --resource-budgets 0.30,0.45,0.60,0.80,1.00 \
  --eval-size 1024 \
  --train-known-per-class 192 \
  --train-open 1024 \
  --cal-known-per-class 64 \
  --cal-open 768 \
  --full-open-severity mild:0.25,medium:0.50,hard:0.75,extreme:0.91
```

This command consumes precomputed features. Feature extraction and other GPU-heavy work must be submitted to a compute node. Example:

```bash
export OPENSEMCOM_FEATURE_MODEL_ID=<model-id>
export OPENSEMCOM_FEATURE_SLUG=<short-name>
sbatch slurm/extract_foundation_features.slurm
```

## How to Read the Metrics

| Metric | Meaning | Better direction |
|---|---|---|
| Semantic goodput | Correct accepted decisions divided by all evaluated samples | Higher |
| Coverage | Accepted decisions divided by all evaluated samples | Higher, subject to safety |
| Accepted OpenOut | Unsafe accepted decisions divided by accepted decisions | Lower |
| Accepted accuracy | Correct decisions divided by accepted decisions | Higher |
| AUROC | Ability of risk score to rank unsafe samples above safe samples | Higher |
| FPR95 | Safe samples incorrectly flagged when unsafe recall is 95% | Lower |
| Accepted count | Number of accepted evaluation samples | Context for coverage and variance |
| Resource/sample | Mean communication-control cost proxy per evaluated sample | Lower at equal goodput |
| Latency/sample | Mean latency-unit proxy per evaluated sample | Lower at equal goodput |
| Retransmission rate | Fraction sent through semantic refinement/HARQ | Depends on safety/goodput tradeoff |

An accepted-outage target is a constraint, not a score. For example, a row labeled `target = 0.05` is useful only if its evaluated accepted outage is actually at or below 0.05. Among feasible rows, compare goodput first, then resource and latency use.

## Results and Run Archive

[results/README.md](results/README.md) is the index for report-ready artifacts, aggregate tables, curves, diagnostics, and raw run files. In short:

- `results/` contains compact reports, final tables, plots, diagnostics, and manifest audits.
- `runs/` contains seed-level metrics, selected policies, traces, developmental outcomes, and aggregate JSON/CSV files.
- `logs/` contains Slurm standard output and error logs.
- `manifests/` contains experiment row definitions and feature-manifest metadata. Paths are Trillium-specific and must be validated after moving the scratch tree.
- `models/checkpoints/` contains project-trained receiver heads and fitted comparison components. Downloaded backbone weights remain excluded.

The Git repository intentionally excludes datasets, downloaded foundation-model weights, feature arrays, caches, virtual environments, and temporary files. Project-trained checkpoints under `models/checkpoints/` are versioned for reproducibility.

## Repository Layout

```text
configs/              experiment configurations by open condition
logs/                 Slurm stdout/stderr retained with the run record
manifests/             source and feature manifests
models/checkpoints/    project-trained receiver and comparison model state
results/               reader-facing reports, tables, plots, and audits
runs/                  raw and seed-level experiment outputs
scripts/               setup, download, audit, and exact-beam utilities
slurm/                 compute-node extraction, training, and evaluation jobs
src/opensemcom/        Python package
tests/                 manifest, channel, calibration, risk, and simulation tests
EXPERIMENTS.md         reproducibility and cluster runbook
OpenSemCom_Research_Plan.md  research formulation and design history
```

## Current Limitations

- Exact DeepSense beam experiments use 512 common feature rows rather than all 2,411 Scenario 1 rows.
- Exact-beam safety-constrained goodput remains low even though unconstrained top-5 accuracy is substantially higher.
- Some communication baselines are implementation-style approximations rather than official pretrained WITT or DeepJSCC checkpoints; the reports identify them explicitly.
- Resource and latency values in the communication-control suite are controlled cost units, not measured wall-clock latency or radio energy.
- Older run directories are retained for provenance. Use the dated June 29 report and its linked raw files as the current result set.

These limitations are experimental facts, not hidden failure modes. They define the next work: complete DeepSense foundation-feature extraction, rerun exact beam prediction on the full measured set, and evaluate official communication-model checkpoints under the same manifests and safety targets.

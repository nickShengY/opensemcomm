# OpenSemCom Experiment Runbook

This document explains how to inspect, validate, rerun, and interpret the OpenSemCom experiments on Trillium. It complements the reader-facing overview in `README.md` and the artifact catalog in `results/README.md`.

## 1. Execution Policy

Use the login node for repository inspection, dependency setup, dataset/model downloads, manifest inspection, and Slurm submission. Use compute nodes for foundation-feature extraction, training, evaluation, and heavy preprocessing. All persistent files must stay under the scratch project tree.

```bash
cd /home/nickyun/links/scratch/new_study/opensemcom
source scripts/env_scratch.sh
```

Confirm that cache variables resolve to scratch:

```bash
for name in HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE TORCH_HOME XDG_CACHE_HOME WANDB_DIR TMPDIR; do
  printf '%s=%s\n' "$name" "${!name}"
done
```

## 2. Artifact Flow

```text
licensed/public dataset in scratch
        |
        v
source manifest --validate--> source files exist and are non-empty
        |
        v
foundation feature extraction on a compute node
        |
        v
DINOv3 / SigLIP2 / OpenCLIP feature manifests --validate-->
        |
        v
train + calibration + evaluation splits
        |
        v
seed-level summaries and policy selections in runs/
        |
        v
aggregate CSV/JSON, curves, diagnostics, and report in results/
```

No stage is permitted to create substitute samples when an input is unavailable. A missing dataset, model, feature, or manifest is a blocked input and should be recorded as such.

## 3. Manifest Contract

The minimum schema is:

```csv
source_path,label,task,domain,is_unknown,split,regime
```

Important meanings:

| Field | Meaning |
|---|---|
| `source_path` | Existing source artifact or extracted feature file |
| `label` | Dataset-derived class or task target |
| `task` | Classification, detection, segmentation, driving, beam prediction, or text task |
| `domain` | Source dataset/domain identifier |
| `is_unknown` | Whether the semantic class is outside the known training class set |
| `split` | `train`, `calibration`, or `eval` |
| `regime` | Closed-ID or the applicable open condition |

Feature manifests can add `dataset`, `raw_source_path`, feature model identifiers, and traceability metadata. Validate every input manifest immediately before a run:

```bash
python -m opensemcom.cli.validate_manifest manifests/opensemcom_real.csv
python -m opensemcom.cli.validate_manifest manifests/opensemcom_real_dinov3_mixed_open_calibration.csv
python -m opensemcom.cli.validate_manifest manifests/opensemcom_real_siglip2_base.csv
python -m opensemcom.cli.validate_manifest manifests/opensemcom_real_openclip_dfn5b.csv
```

## 4. Dataset and Model Preparation

Build and validate the source manifest using scratch-resident datasets:

```bash
source scripts/env_scratch.sh
bash scripts/download_models_and_data.sh
```

Optional model prefetching is also scratch-local:

```bash
OPENSEMCOM_PREFETCH_MODELS=1 bash scripts/download_models_and_data.sh
```

Licensed datasets are not downloaded automatically when credentials or agreement are required. Place authorized archives or extracted files under `data/`, rerun the manifest builder, and validate the result. Never replace unavailable rows with generated content.

## 5. Foundation Feature Extraction

Submit each heavy extractor to Slurm. Set the model identifier and a stable feature slug so reruns reuse the same output tree.

```bash
export OPENSEMCOM_FEATURE_MODEL_ID=<model-id>
export OPENSEMCOM_FEATURE_SLUG=dinov3
export OPENSEMCOM_FEATURE_OUTPUT_MANIFEST="$PWD/manifests/opensemcom_real_dinov3.csv"
sbatch slurm/extract_foundation_features.slurm
```

Repeat for SigLIP2 and OpenCLIP DFN5B. The extraction script writes features under `data/features/` and validates the output manifest before the job exits. Inspect both the Slurm exit state and its log:

```bash
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
sed -n '1,240p' "logs/osc-found-feats-<job-id>.out"
sed -n '1,240p' "logs/osc-found-feats-<job-id>.err"
```

## 6. Latest Communication-Control Suite

Inputs:

- DINOv3 mixed-open calibration manifest.
- SigLIP2 feature manifest.
- OpenCLIP DFN5B feature manifest.
- DeepSense Scenario 1 metadata under `data/deepsense6g/Scenario1`.

Run command used for the June 29 result:

```bash
source scripts/env_scratch.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m opensemcom.cli.communication_control_suite \
  --feature-manifest dino=manifests/opensemcom_real_dinov3_mixed_open_calibration.csv \
  --feature-manifest siglip2=manifests/opensemcom_real_siglip2_base.csv \
  --feature-manifest openclip=manifests/opensemcom_real_openclip_dfn5b.csv \
  --output-prefix runs/comm_control_extra_experiments_20260629 \
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

Expected raw outputs:

```text
runs/comm_control_extra_experiments_20260629_summary.csv
runs/comm_control_extra_experiments_20260629_policies.csv
runs/comm_control_extra_experiments_20260629_manifest_summary.json
```

The summary contains 4,800 rows: five seeds, six tasks/conditions, four outage targets, five resource budgets, and eight evaluated method/control variants. The policy file records the selected route and thresholds so every reported operating point can be audited.

## 7. DeepSense Exact-Beam Top-k Run

```bash
source scripts/env_scratch.sh
python scripts/deepsense_exact_topk.py \
  --feature-manifest dino=manifests/opensemcom_real_dinov3_mixed_open_calibration.csv \
  --feature-manifest siglip2=manifests/opensemcom_real_siglip2_base.csv \
  --feature-manifest openclip=manifests/opensemcom_real_openclip_dfn5b.csv \
  --output-prefix runs/deepsense_exact_topk_20260629 \
  --seeds 0,1,2,3,4 \
  --targets 0.05,0.10
```

Expected outputs:

```text
runs/deepsense_exact_topk_20260629_summary.csv
runs/deepsense_exact_topk_20260629_manifest_summary.json
results/final_opensemcom_deepsense_exact_topk_20260629.csv
```

This run uses the 512 DeepSense rows shared by all three feature manifests. It is not the final full-Scenario-1 experiment.

## 8. Severity Ladder

Severity changes the fraction of evaluation rows carrying at least one open exposure:

| Name | Open fraction | Interpretation |
|---|---:|---|
| Mild | 0.25 | Mostly known deployment with occasional open exposure |
| Medium | 0.50 | Equal known and open exposure |
| Hard | 0.75 | Open exposure dominates |
| Extreme | 0.91 | Nearly all evaluation rows are exposed to a shift |

The same model families, split construction, seed set, safety targets, and resource budgets are used at each level. This makes the decline in coverage and goodput interpretable as exposure severity increases.

## 9. Decision and Metric Definitions

For a calibrated risk score `r`:

```text
r <= q1       accept
q1 < r <= q2  refine once, then apply the final decision
r > q2        reject/open
```

For `N` evaluation samples:

```text
coverage = accepted / N
accepted outage = accepted unsafe / accepted
semantic goodput = accepted correct / N
accepted accuracy = accepted correct / accepted
```

An unsafe decision is an accepted open-exposure sample or an accepted incorrect semantic prediction. AUROC and FPR95 evaluate whether the risk score separates unsafe and safe decisions before thresholding.

When comparing methods at a target such as 0.05:

1. Discard operating points whose evaluated accepted outage exceeds 0.05.
2. Compare semantic goodput among the feasible points.
3. Use resource/sample, latency/sample, and retransmission rate to compare communication cost at similar goodput.
4. Inspect accepted count and seed variance; tiny accepted sets can make outage estimates unstable.

## 10. Verification Checklist

Before declaring a run complete, verify all of the following:

- Source and feature manifests pass `opensemcom-validate-manifest`.
- Manifest summary JSON exists and reports the expected datasets and common-row count.
- Slurm reports `COMPLETED` with exit code `0:0` for heavy jobs.
- Standard output and error logs exist under `logs/`.
- Every requested seed exists in the raw summary.
- Every requested severity, outage target, and resource budget exists.
- Accepted-outage constraints are checked on evaluation values, not inferred from the requested target name.
- Aggregate CSV/JSON and report files exist under `results/`.
- `pytest` passes after code changes.

Useful checks:

```bash
pytest
python -m opensemcom.cli.validate_manifest manifests/opensemcom_real.csv
python -m opensemcom.cli.aggregate_results \
  --runs-dir runs \
  --output-csv runs/aggregate_results.csv \
  --output-json runs/aggregate_results.json
```

## 11. Resume and Idempotency

- Use stable output prefixes and model slugs.
- Do not delete completed feature files when rerunning a downstream experiment.
- Validate existing feature manifests before scheduling extraction again.
- Give new scientific runs a new dated prefix instead of overwriting the published raw output.
- Keep seed-level output and selected-policy files together.
- Aggregate only after all expected seeds complete.
- Record failed jobs in `logs/`; do not turn a partial run into a final table.

## 12. Next Wireless Experiment

Scenario 1 has 2,411 measured rows, while the common foundation-feature subset currently has 512. The next run should:

1. Build a DeepSense-only source manifest covering all authorized Scenario 1 camera rows.
2. Validate all camera, mmWave, GPS, and beam-label references.
3. Extract DINOv3, SigLIP2, and OpenCLIP features on compute nodes.
4. Intersect the resulting manifests and report the retained row count.
5. Rerun exact top-1, top-3, and top-5 beam evaluation over five seeds.
6. Report unconstrained accuracy and goodput at outage targets 0.01, 0.02, 0.05, and 0.10.
7. Compare OpenSemCom against the strongest single-backbone model at exactly the same splits and accepted-outage target.

Until that run exists, the sector task and 512-row exact-beam task should be described as measured wireless evidence, not a complete beam-management evaluation.

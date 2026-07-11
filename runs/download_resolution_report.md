# Dataset Block Resolution Report

Date: 2026-06-17

## Scratch-Only Download Roots

- Project: `/home/nickyun/links/scratch/new_study/opensemcom`
- Data root: `/home/nickyun/links/scratch/new_study/opensemcom/data`
- HF cache: `/home/nickyun/links/scratch/new_study/opensemcom/cache/hf`
- Dataset cache: `/home/nickyun/links/scratch/new_study/opensemcom/cache/datasets`
- Transformer cache: `/home/nickyun/links/scratch/new_study/opensemcom/cache/transformers`
- Torch cache: `/home/nickyun/links/scratch/new_study/opensemcom/cache/torch`

## Downloaded Datasets

- nuScenes mini:
  - Archive: `/home/nickyun/links/scratch/new_study/opensemcom/data/nuscenes/v1.0-mini.tgz`
  - Extracted root: `/home/nickyun/links/scratch/new_study/opensemcom/data/nuscenes`
  - Manifest rows: 512
- nuImages mini:
  - Archive: `/home/nickyun/links/scratch/new_study/opensemcom/data/nuimages/nuimages-v1.0-mini.tgz`
  - Extracted root: `/home/nickyun/links/scratch/new_study/opensemcom/data/nuimages`
  - Manifest rows: 0. It is stored for future adapter work; nuScenes mini supplies the current driving rows.
- DeepSense 6G Scenario 1:
  - Archive: `/home/nickyun/links/scratch/new_study/opensemcom/data/deepsense6g/Scenario1.zip`
  - Extracted root: `/home/nickyun/links/scratch/new_study/opensemcom/data/deepsense6g/Scenario1`
  - Manifest rows: 512
- BDD100K Hugging Face mirror:
  - Parquet root: `/home/nickyun/links/scratch/new_study/opensemcom/data/bdd100k_hf/parquet`
  - Image subset root: `/home/nickyun/links/scratch/new_study/opensemcom/data/bdd100k_hf/images`
  - Downloaded images: 256
  - Manifest rows: 0. The accessible mirror is image-only, so it was not added to the manifest to avoid fake labels.
- BDD100K official Berkeley archives:
  - Labels: `/home/nickyun/links/scratch/new_study/opensemcom/data/bdd100k/bdd100k_labels.zip`
  - Images: `/home/nickyun/links/scratch/new_study/opensemcom/data/bdd100k/bdd100k_images_100k.zip`
  - Staged real subset: `/home/nickyun/links/scratch/new_study/opensemcom/data/bdd100k/staged`
  - Manifest rows: 512
- AG News:
  - Parquet root: `/home/nickyun/links/scratch/new_study/opensemcom/data/ag_news/parquet`
  - Materialized text root: `/home/nickyun/links/scratch/new_study/opensemcom/data/ag_news/samples`
  - Manifest rows: 512

## Updated Manifest

- Manifest: `/home/nickyun/links/scratch/new_study/opensemcom/manifests/opensemcom_real.csv`
- Rows: 11,904
- Calibration rows: 640
- Eval rows: 11,264
- Indexed rows: 8,832

Dataset row counts:

- ag_news: 512
- bdd100k: 512
- cifar10: 2,432
- cifar100: 6,400
- cityscapes: 512
- coco: 512
- deepsense6g: 512
- nuscenes: 512

## Validation

- `pytest`: 11 passed.
- Manifest validation passed with all paths under scratch.
- CLIP feature manifest validation passed with all feature paths under scratch.
- Updated smoke Slurm job: `604543`
- Updated full Slurm job: `604546`
- Full-manifest Slurm job: `604559`
- CLIP feature extraction Slurm job: `604983`
- CLIP feature evaluation Slurm job: `604990`
- CLIP Mahalanobis/open-exposure feature evaluation Slurm job: `604995`
- Slurm state: completed.
- Exit code: `0:0`.
- Latest full results:
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_results_604546.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_results_604546.json`
- Latest full-manifest results:
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_manifest_results_604559.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/full_manifest_results_604559.json`
- Latest CLIP-feature results:
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/clip_full_results_604995.csv`
  - `/home/nickyun/links/scratch/new_study/opensemcom/runs/clip_full_results_604995.json`

## Downloaded Models And Features

- CLIP model: `openai/clip-vit-base-patch32`
- HF model cache: `/home/nickyun/links/scratch/new_study/opensemcom/cache/hf`
- Extracted CLIP features: `/home/nickyun/links/scratch/new_study/opensemcom/data/features/openai__clip-vit-base-patch32`
- CLIP feature manifest: `/home/nickyun/links/scratch/new_study/opensemcom/manifests/opensemcom_real_clip.csv`
- Feature rows: 11,904
- Skipped feature rows: 0

## Remaining Limitation

The BDD100K Hugging Face mirror remains image-only and excluded. The official Berkeley BDD100K label and image archives are now downloaded under scratch, and a bounded real labeled subset is staged and included in the manifest.

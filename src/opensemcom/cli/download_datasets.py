"""Download accessible datasets into scratch for OpenSemCom."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


AG_NEWS = "SetFit/ag_news"
BDD100K = "dgural/bdd100k"
HF_API = "https://datasets-server.huggingface.co"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download accessible OpenSemCom datasets to scratch.")
    parser.add_argument("--data-root", default="/home/nickyun/links/scratch/new_study/opensemcom/data")
    parser.add_argument("--ag-news-per-split", type=int, default=512)
    parser.add_argument("--skip-ag-news", action="store_true")
    parser.add_argument("--skip-bdd-parquet", action="store_true")
    parser.add_argument("--bdd-images", type=int, default=256)
    parser.add_argument("--skip-bdd-images", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_ag_news:
        download_hf_parquet(AG_NEWS, data_root / "ag_news" / "parquet")
        materialize_ag_news_samples(data_root / "ag_news", args.ag_news_per_split)
    if not args.skip_bdd_parquet:
        download_hf_parquet(BDD100K, data_root / "bdd100k_hf" / "parquet")
    if not args.skip_bdd_images:
        download_bdd_images(data_root / "bdd100k_hf" / "images", args.bdd_images)


def download_hf_parquet(dataset: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _get_json(f"{HF_API}/parquet?dataset={urllib.parse.quote(dataset, safe='/')}")
    for item in payload.get("parquet_files", []):
        split = item["split"]
        path = output_dir / split / item["filename"]
        curl_download(item["url"], path)


def materialize_ag_news_samples(root: Path, per_split: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.jsonl"
    seen: set[tuple[str, int]] = set()
    existing: list[dict] = []
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                seen.add((record["split"], int(record["row_idx"])))
                existing.append(record)

    records = existing
    for split in ("train", "test"):
        offset = 0
        written = sum(1 for record in records if record["split"] == split)
        while written < per_split:
            length = min(100, per_split - written)
            url = (
                f"{HF_API}/rows?dataset={urllib.parse.quote(AG_NEWS, safe='/')}"
                f"&config=default&split={split}&offset={offset}&length={length}"
            )
            payload = _get_json(url)
            rows = payload.get("rows", [])
            if not rows:
                break
            for item in rows:
                row_idx = int(item["row_idx"])
                if (split, row_idx) in seen:
                    continue
                row = item["row"]
                label = int(row["label"])
                label_text = str(row.get("label_text") or label)
                rel_path = Path("samples") / split / f"{row_idx:06d}_{label}.txt"
                path = root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(row["text"]) + "\n", encoding="utf-8")
                record = {
                    "dataset": AG_NEWS,
                    "split": split,
                    "row_idx": row_idx,
                    "path": str(rel_path),
                    "label": label,
                    "label_text": label_text,
                }
                records.append(record)
                seen.add((split, row_idx))
                written += 1
                if written >= per_split:
                    break
            offset += len(rows)

    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def download_bdd_images(output_dir: Path, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    downloaded = len(list(output_dir.glob("*.jpg")))
    while downloaded < count:
        length = min(100, count - downloaded)
        url = (
            f"{HF_API}/rows?dataset={urllib.parse.quote(BDD100K, safe='/')}"
            f"&config=default&split=train&offset={offset}&length={length}"
        )
        payload = _get_json(url)
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            row_idx = int(item["row_idx"])
            path = output_dir / f"{row_idx:06d}.jpg"
            if not path.exists():
                image_url = item["row"]["image"]["src"]
                curl_download(image_url, path)
            downloaded = len(list(output_dir.glob("*.jpg")))
            if downloaded >= count:
                break
        offset += len(rows)


def curl_download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if path.exists() and path.stat().st_size > 0:
        return
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-C",
            "-",
            "-o",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(path)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


if __name__ == "__main__":
    main()

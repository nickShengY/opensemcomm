"""Aggregate OpenSemCom run metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate OpenSemCom metrics.json files.")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--pattern", default="*/metrics.json", help="Glob pattern relative to runs-dir.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    rows = [_flatten(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(runs_dir.glob(args.pattern))]
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {runs_dir}")
    fieldnames = sorted({key for row in rows for key in row})
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(output_csv), "json": str(output_json)}, indent=2))


def _flatten(payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "name": payload.get("name", ""),
        "regime": payload.get("regime", ""),
        "seed": payload.get("seed", ""),
        "samples": payload.get("samples", ""),
        "evaluated_samples": payload.get("evaluated_samples", ""),
        "calibration_samples": payload.get("calibration_samples", ""),
        "users": payload.get("users", ""),
        "dataset_manifest": payload.get("dataset_manifest", ""),
    }
    row.update(payload.get("metrics", {}))
    row.update({f"decision_{key}": value for key, value in payload.get("decisions", {}).items()})
    return row


if __name__ == "__main__":
    main()

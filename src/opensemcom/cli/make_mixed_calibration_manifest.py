"""Create a leakage-safe mixed-open calibration manifest from dataset rows."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from opensemcom.manifest import MANIFEST_COLUMNS, validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move open eval rows into an open-calibration split.")
    parser.add_argument("--input", required=True, help="Input dataset manifest.")
    parser.add_argument("--output", required=True, help="Output derived manifest under scratch.")
    parser.add_argument("--open-rows", type=int, default=768, help="Number of open rows to reserve.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regime", default="full-open")
    parser.add_argument("--open-split", default="open-calibration")
    parser.add_argument("--train-task", action="append", default=["classification"])
    parser.add_argument("--train-domain", action="append", default=["cifar10"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input manifest not found: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or MANIFEST_COLUMNS)
        rows = list(reader)
    required = set(MANIFEST_COLUMNS)
    for column in required:
        if column not in fieldnames:
            fieldnames.append(column)
    train_tasks = set(args.train_task)
    train_domains = set(args.train_domain)
    candidates = [
        idx for idx, row in enumerate(rows)
        if row.get("split", "eval") == "eval"
        and row.get("regime", "") == args.regime
        and _is_open_exposure(row, train_tasks, train_domains)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = set(candidates[: max(0, args.open_rows)])
    for idx in selected:
        rows[idx]["split"] = args.open_split
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = validate_manifest(output)
    report = {
        "input": str(source),
        "output": str(output),
        "open_split": args.open_split,
        "requested_open_rows": args.open_rows,
        "reserved_open_rows": len(selected),
        "manifest_summary": summary,
    }
    output.with_suffix(".mixed_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _is_open_exposure(row: dict[str, str], train_tasks: set[str], train_domains: set[str]) -> bool:
    return (
        row.get("is_unknown", "").strip().lower() in {"1", "true", "yes", "y"}
        or row.get("task", "") not in train_tasks
        or row.get("domain", "") not in train_domains
    )


if __name__ == "__main__":
    main()

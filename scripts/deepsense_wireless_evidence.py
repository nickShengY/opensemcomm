#!/usr/bin/env python3
"""Summarize DeepSense 6G Scenario1 wireless evidence.

This audit intentionally does not simulate channels. It verifies that rows link
camera, mmWave power, GPS, and beam-index artifacts, then writes compact
metadata summaries that can be cited separately from generated channel models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit DeepSense 6G wireless metadata.")
    parser.add_argument("--scenario-root", default="data/deepsense6g/Scenario1")
    parser.add_argument("--output-prefix", default="results/deepsense_scenario1_wireless")
    parser.add_argument("--max-detail-rows", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.scenario_root).expanduser().resolve()
    scenario_csv = root / "scenario1.csv"
    rows = read_rows(scenario_csv)
    details = []
    missing = {"camera": 0, "mmwave": 0, "unit1_gps": 0, "unit2_gps": 0}
    beam_counts: dict[str, int] = {}
    power_values = []
    vector_lengths = []
    for row in rows:
        camera = resolve(root, row.get("unit1_rgb", ""))
        mmwave = resolve(root, row.get("unit1_pwr_60ghz", ""))
        unit1_gps = resolve(root, row.get("unit1_loc", ""))
        unit2_gps = resolve(root, row.get("unit2_loc", ""))
        for key, path in (("camera", camera), ("mmwave", mmwave), ("unit1_gps", unit1_gps), ("unit2_gps", unit2_gps)):
            if not path.exists():
                missing[key] += 1
        beam = row.get("unit1_beam_index", "").strip()
        beam_counts[beam] = beam_counts.get(beam, 0) + 1
        mm_stats = read_mmwave_stats(mmwave) if mmwave.exists() else None
        if mm_stats:
            power_values.append(mm_stats["max_power"])
            vector_lengths.append(mm_stats["num_values"])
        if len(details) < args.max_detail_rows:
            details.append(
                {
                    "index": row.get("index", ""),
                    "beam_index": beam,
                    "camera_exists": camera.exists(),
                    "mmwave_exists": mmwave.exists(),
                    "unit1_gps_exists": unit1_gps.exists(),
                    "unit2_gps_exists": unit2_gps.exists(),
                    "mmwave_num_values": mm_stats["num_values"] if mm_stats else 0,
                    "mmwave_max_power": mm_stats["max_power"] if mm_stats else "",
                    "mmwave_mean_power": mm_stats["mean_power"] if mm_stats else "",
                    "unit2_pdop": row.get("unit2_PDOP", ""),
                    "unit2_hdop": row.get("unit2_HDOP", ""),
                    "unit2_num_sat": row.get("unit2_num_sat", ""),
                    "unit2_fix_type": row.get("unit2_fix_type", ""),
                }
            )
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenario_root": str(root),
        "scenario_csv": str(scenario_csv),
        "rows": len(rows),
        "missing": missing,
        "beam_counts": dict(sorted(beam_counts.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 9999)),
        "mmwave_rows": len(power_values),
        "mmwave_vector_length_min": int(min(vector_lengths)) if vector_lengths else 0,
        "mmwave_vector_length_max": int(max(vector_lengths)) if vector_lengths else 0,
        "mmwave_max_power_mean": float(np.mean(power_values)) if power_values else None,
        "mmwave_max_power_std": float(np.std(power_values)) if power_values else None,
        "field_collected_wireless_evidence": True,
        "generated_channel_samples": False,
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_prefix.with_suffix(".csv"), details)
    print(json.dumps(summary, indent=2, sort_keys=True))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(root: Path, value: str) -> Path:
    return (root / value.strip()).resolve()


def read_mmwave_stats(path: Path) -> dict[str, float | int] | None:
    values = []
    for token in path.read_text(encoding="utf-8", errors="replace").replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "num_values": int(arr.size),
        "max_power": float(np.max(arr)),
        "mean_power": float(np.mean(arr)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Analyze OpenSemCom traces for action rates and risk-goodput tradeoffs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze OpenSemCom trace files.")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--pattern", default="*/traces.json")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--curve-points", type=int, default=101)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    paths = sorted(runs_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No traces found under {runs_dir} with pattern {args.pattern}")
    summaries = []
    curves = []
    for path in paths:
        traces = json.loads(path.read_text(encoding="utf-8"))
        run = path.parent.name
        summaries.append(_summarize_run(run, traces))
        curves.extend(_risk_curve(run, traces, args.curve_points))
    prefix = Path(args.output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(prefix.with_name(prefix.name + "_action_rates.csv"), summaries)
    _write_csv(prefix.with_name(prefix.name + "_risk_curves.csv"), curves)
    print(json.dumps({"runs": len(summaries), "curve_rows": len(curves)}, indent=2))


def _summarize_run(run: str, traces: list[dict]) -> dict[str, object]:
    n = len(traces)
    decisions = Counter(t["decision"] for t in traces)
    correct = sum(t["y"] == t["y_hat"] for t in traces)
    accepted = [t for t in traces if t["decision"] == "accept"]
    accepted_correct = sum(t["y"] == t["y_hat"] for t in accepted)
    accepted_unsafe = sum((t["y"] != t["y_hat"]) or t.get("open_exposure", t.get("unknown", False)) for t in accepted)
    open_exposure = [t for t in traces if t.get("open_exposure", t.get("unknown", False))]
    known_id = [t for t in traces if not t.get("open_exposure", t.get("unknown", False))]
    row: dict[str, object] = {
        "run": run,
        "samples": n,
        "accuracy": correct / max(n, 1),
        "known_id_accuracy": sum(t["y"] == t["y_hat"] for t in known_id) / max(len(known_id), 1),
        "open_exposure_count": len(open_exposure),
        "known_id_count": len(known_id),
        "accepted": len(accepted),
        "accepted_correct": accepted_correct,
        "accepted_unsafe": accepted_unsafe,
        "accepted_precision": accepted_correct / max(len(accepted), 1),
        "accepted_open_outage": accepted_unsafe / max(len(accepted), 1),
    }
    for decision, count in sorted(decisions.items()):
        row[f"decision_{decision}"] = count
        row[f"rate_{decision}"] = count / max(n, 1)
    return row


def _risk_curve(run: str, traces: list[dict], points: int) -> list[dict[str, object]]:
    if not traces:
        return []
    risks = np.asarray([float(t["risk_score"]) for t in traces], dtype=np.float64)
    thresholds = np.quantile(risks, np.linspace(0.0, 1.0, points))
    rows = []
    n = len(traces)
    for threshold in thresholds:
        selected = [t for t in traces if float(t["risk_score"]) <= float(threshold)]
        if selected:
            correct = sum(t["y"] == t["y_hat"] for t in selected)
            unsafe = sum((t["y"] != t["y_hat"]) or t.get("open_exposure", t.get("unknown", False)) for t in selected)
            selected_open = sum(t.get("open_exposure", t.get("unknown", False)) for t in selected)
        else:
            correct = unsafe = selected_open = 0
        rows.append(
            {
                "run": run,
                "threshold": float(threshold),
                "coverage": len(selected) / max(n, 1),
                "selected": len(selected),
                "selected_open": selected_open,
                "goodput_proxy": correct / max(n, 1),
                "selected_precision": correct / max(len(selected), 1),
                "selected_open_outage": unsafe / max(len(selected), 1),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

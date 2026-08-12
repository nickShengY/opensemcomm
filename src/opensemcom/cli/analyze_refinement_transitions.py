"""Write stage-transition score diagnostics from progressive HARQ traces.

The analyser is intentionally post-hoc: it reads traces written after an
experiment has completed and never participates in model fitting, calibration,
or action selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RAW_FIELDS = (
    "run_identifier",
    "configuration_identifier",
    "seed",
    "evaluation_condition",
    "sample_key",
    "refinement_round",
    "current_stage_index",
    "next_stage_index",
    "current_stage",
    "next_stage",
    "transition_type",
    "current_risk_score",
    "next_risk_score",
    "current_q_accept",
    "next_q_accept",
    "current_q_refine",
    "next_q_refine",
    "raw_score_change",
    "current_acceptance_margin",
    "next_acceptance_margin",
    "margin_change",
    "current_action",
    "next_action",
    "final_terminal_action",
    "final_semantic_stage",
    "terminal_outcome",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize progressive-refinement score transitions.")
    parser.add_argument("--traces", nargs="+", required=True, help="One or more traces.json files.")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace_paths = [Path(value).expanduser().resolve() for value in args.traces]
    output_paths = write_refinement_diagnostics(trace_paths, Path(args.output_dir).expanduser().resolve())
    print(json.dumps({key: str(value) for key, value in output_paths.items()}, indent=2, sort_keys=True))


def write_refinement_diagnostics(
    trace_paths: Iterable[Path], output_dir: Path
) -> dict[str, Path]:
    """Flatten HARQ transition metadata and write manuscript-ready summaries."""
    rows: list[dict[str, Any]] = []
    stage_aware_values: set[bool] = set()
    for path in trace_paths:
        traces = json.loads(path.read_text(encoding="utf-8"))
        for trace in traces:
            if "stage_aware_thresholds" in trace:
                stage_aware_values.add(bool(trace["stage_aware_thresholds"]))
            for transition in trace.get("refinement_transitions", []):
                row = {
                    "run_identifier": trace.get("run_identifier", path.parent.name),
                    "configuration_identifier": trace.get("configuration_identifier", ""),
                    "seed": trace.get("seed", ""),
                    "evaluation_condition": trace.get("evaluation_condition", trace.get("regime", "")),
                    "sample_key": trace.get("sample_key", f"evaluation-index-{trace.get('index', '')}"),
                    **transition,
                }
                rows.append({field: row.get(field, "") for field in RAW_FIELDS})

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_transition_scores.csv"
    summary_path = output_dir / "transition_summary.csv"
    terminal_summary_path = output_dir / "transition_summary_by_terminal.csv"
    report_path = output_dir / "transition_report.md"
    _write_csv(raw_path, rows, RAW_FIELDS)
    _write_csv(summary_path, _summarize(rows, include_terminal=False), _summary_fields(False))
    _write_csv(terminal_summary_path, _summarize(rows, include_terminal=True), _summary_fields(True))
    report_path.write_text(_report(rows, stage_aware_values), encoding="utf-8")
    return {
        "raw_csv": raw_path,
        "summary_csv": summary_path,
        "summary_by_terminal_csv": terminal_summary_path,
        "report": report_path,
    }


def _summarize(rows: list[dict[str, Any]], include_terminal: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["evaluation_condition"]), str(row["transition_type"]))
        if include_terminal:
            key += (_terminal_group(str(row["terminal_outcome"])),)
        groups[key].append(row)

    output = []
    for key, values in sorted(groups.items()):
        row: dict[str, Any] = {
            "evaluation_condition": key[0],
            "transition_type": key[1],
            "count": len(values),
        }
        if include_terminal:
            row["terminal_group"] = key[2]
        row.update(_change_statistics(values, "raw_score_change", "raw"))
        row.update(_change_statistics(values, "margin_change", "margin"))
        output.append(row)
    return output


def _change_statistics(rows: list[dict[str, Any]], source: str, prefix: str) -> dict[str, float]:
    values = [float(row[source]) for row in rows if _finite(row.get(source))]
    if not values:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_positive_change_rate": float("nan"),
        }
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_positive_change_rate": sum(value > 0.0 for value in values) / len(values),
    }


def _summary_fields(include_terminal: bool) -> tuple[str, ...]:
    fields = ["evaluation_condition", "transition_type"]
    if include_terminal:
        fields.append("terminal_group")
    fields.extend(
        [
            "count",
            "raw_mean",
            "raw_median",
            "raw_std",
            "raw_positive_change_rate",
            "margin_mean",
            "margin_median",
            "margin_std",
            "margin_positive_change_rate",
        ]
    )
    return tuple(fields)


def _terminal_group(outcome: str) -> str:
    return "rejected/open" if outcome == "rejected/open" else "accepted"


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: list[dict[str, Any]], stage_aware_values: set[bool]) -> str:
    stage_aware = "yes" if stage_aware_values == {True} else "mixed/unknown"
    return "\n".join(
        [
            "# Progressive refinement score-change diagnostic",
            "",
            f"- Transition rows: {len(rows)}",
            f"- Stage-aware thresholds recorded in traces: {stage_aware}",
            "- This report is post-hoc: it does not fit or revise any threshold.",
            "- `raw_score_change = current_risk_score - next_risk_score`.",
            "- `margin_change = (current_risk_score - current_q_accept) - (next_risk_score - next_q_accept)`.",
            "- Positive values mean the next observation moved below, or closer to, its own acceptance threshold.",
            "",
        ]
    )


if __name__ == "__main__":
    main()

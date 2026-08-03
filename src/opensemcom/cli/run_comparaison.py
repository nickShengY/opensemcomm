"""Run one or more fair external-backbone comparisons for one benchmark regime."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from opensemcom.cli.run_config import _config_from_dict, _load_config
from opensemcom.comparaison import ComparisonConfig, ComparisonMethod, ComparisonOrchestrator
from opensemcom.comparaison.orchestrator import _read_manifest, _row_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fair OpenSemCom external-backbone comparisons.")
    parser.add_argument("--config", required=True, help="Existing OpenSemCom YAML/JSON config for this regime.")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--opensemcom-manifest", required=True)
    parser.add_argument("--dino-manifest")
    parser.add_argument("--siglip-manifest")
    parser.add_argument("--openclip-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="all", help="Comma-separated methods or 'all'.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--payload-blocks", type=int, default=1)
    parser.add_argument("--accept-quantile", type=float, default=0.95)
    parser.add_argument("--write-traces", action="store_true")
    parser.add_argument("--expected-calibration-known", type=int, help="Fail unless the common cohort has this many known calibration rows.")
    parser.add_argument("--expected-calibration-open", type=int, help="Fail unless the common cohort has this many labelled open calibration rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = _parse_methods(args.methods)
    base_config = replace(_config_from_dict(_load_config(args.config)), seed=args.seed)
    manifest_args = {
        ComparisonMethod.OPENSEMCOM: args.opensemcom_manifest,
        ComparisonMethod.DINO: args.dino_manifest,
        ComparisonMethod.SIGLIP: args.siglip_manifest,
        ComparisonMethod.OPENCLIP: args.openclip_manifest,
    }
    missing = [method.value for method in methods if not manifest_args[method]]
    if missing:
        raise ValueError(f"Selected methods are missing feature manifests: {', '.join(missing)}")
    manifests = {
        method.value: Path(manifest_args[method]).expanduser().resolve()
        for method in methods
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="opensemcom-comparaison-") as temp_dir:
        filtered_raw, filtered_manifests = _filter_to_regime(
            raw_manifest=Path(args.raw_manifest).expanduser().resolve(),
            manifests=manifests,
            regime=args.regime,
            temp_dir=Path(temp_dir),
        )
        summary_rows = []
        for method in methods:
            run = ComparisonOrchestrator(
                ComparisonConfig(
                    method=method,
                    raw_manifest=filtered_raw,
                    manifests=filtered_manifests,
                    channel=base_config.channel,
                    regime=args.regime,
                    cohort_methods=tuple(methods),
                    seed=args.seed,
                    payload_blocks=args.payload_blocks,
                    expected_calibration_known=args.expected_calibration_known,
                    expected_calibration_open=args.expected_calibration_open,
                    accept_quantile=args.accept_quantile,
                    opensemcom_config=base_config,
                )
            ).run()
            payload = {
                "method": run.method,
                "regime": args.regime,
                "seed": args.seed,
                "cohort_methods": list(run.cohort_methods),
                "cohort_rows": run.cohort_rows,
                "calibration_rows": run.calibration_rows,
                "calibration_known_rows": run.calibration_known_rows,
                "calibration_open_rows": run.calibration_open_rows,
                "evaluation_rows": run.evaluation_rows,
                "payload_values": run.payload_values,
                "metrics": run.result.metrics,
                "protocol": {
                    "task_domain_metadata_available": run.task_domain_metadata_available,
                    "evaluation_is_unknown_used_for_decision": False,
                },
                "decisions": run.result.decisions,
            }
            (output_dir / f"{run.method}_metrics.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if args.write_traces:
                (output_dir / f"{run.method}_traces.json").write_text(
                    json.dumps(run.result.traces, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            summary_rows.append({"method": run.method, "regime": args.regime, "seed": args.seed, **run.result.metrics})

    _write_summary(output_dir / "summary.csv", summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "methods": [method.value for method in methods]}, indent=2))


def _parse_methods(value: str) -> list[ComparisonMethod]:
    if value.strip().lower() == "all":
        return list(ComparisonMethod)
    requested = [part.strip() for part in value.split(",") if part.strip()]
    return [ComparisonMethod(part) for part in requested]


def _filter_to_regime(
    raw_manifest: Path,
    manifests: dict[str, Path],
    regime: str,
    temp_dir: Path,
) -> tuple[Path, dict[str, Path]]:
    raw_rows = _read_manifest(raw_manifest)
    selected_raw = [
        row
        for row in raw_rows
        if row.get("split") == "calibration" or (row.get("split") == "eval" and row.get("regime") == regime)
    ]
    if not selected_raw:
        raise ValueError(f"Raw manifest has no calibration/eval rows for regime '{regime}'.")
    selected_keys = {_row_key(row) for row in selected_raw}
    filtered_raw = temp_dir / "raw.csv"
    _write_manifest(filtered_raw, raw_rows, selected_raw)
    filtered_manifests = {}
    for method, manifest in manifests.items():
        rows = _read_manifest(manifest)
        selected = [row for row in rows if _row_key(row) in selected_keys]
        destination = temp_dir / f"{method}.csv"
        _write_manifest(destination, rows, selected)
        filtered_manifests[method] = destination
    return filtered_raw, filtered_manifests


def _write_manifest(path: Path, original_rows: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    if not original_rows:
        raise ValueError(f"Manifest is empty: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(original_rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

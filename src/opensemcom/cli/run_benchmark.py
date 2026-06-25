"""Run OpenSemCom-Bench from the command line."""

from __future__ import annotations

import argparse
import json

from opensemcom.benchmark import BenchmarkRegime
from opensemcom.simulation import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenSemCom-Bench.")
    parser.add_argument("--regime", choices=[r.value for r in BenchmarkRegime], default=BenchmarkRegime.FULL_OPEN.value)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--users", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-manifest", required=True, help="CSV manifest pointing to source artifacts.")
    parser.add_argument("--traces", action="store_true", help="Print per-sample traces in addition to aggregate metrics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(
        regime=BenchmarkRegime(args.regime),
        samples=args.samples,
        calibration_samples=args.calibration_samples,
        users=args.users,
        seed=args.seed,
        dataset_manifest=args.dataset_manifest,
    )
    payload = {"metrics": result.metrics, "decisions": result.decisions}
    if args.traces:
        payload["traces"] = result.traces
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

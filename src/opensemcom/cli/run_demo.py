"""Run a compact OpenSemCom demo on a real dataset manifest."""

from __future__ import annotations

import json
import argparse

from opensemcom.benchmark import BenchmarkRegime
from opensemcom.simulation import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact OpenSemCom demo.")
    parser.add_argument("--dataset-manifest", required=True, help="CSV manifest pointing to real data artifacts.")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(
        regime=BenchmarkRegime.FULL_OPEN,
        samples=128,
        calibration_samples=64,
        users=3,
        seed=args.seed,
        dataset_manifest=args.dataset_manifest,
    )
    print(json.dumps({"metrics": result.metrics, "decisions": result.decisions}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

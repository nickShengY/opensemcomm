"""Minimal full-open experiment example using a real dataset manifest."""

import argparse

from opensemcom.benchmark import BenchmarkRegime
from opensemcom.simulation import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    args = parser.parse_args()
    result = run_experiment(
        regime=BenchmarkRegime.FULL_OPEN,
        samples=256,
        users=4,
        seed=11,
        dataset_manifest=args.dataset_manifest,
    )
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4f}")

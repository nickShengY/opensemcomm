"""Run OpenSemCom from an experiment config file."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from opensemcom.benchmark import BenchmarkRegime
from opensemcom.config import (
    AblationConfig,
    CalibrationConfig,
    ChannelConfig,
    DetectorWeights,
    ModelConfig,
    OpenSemComConfig,
    ResourceWeights,
)
from opensemcom.manifest import validate_manifest
from opensemcom.simulation import run_experiment
from opensemcom.types import ChannelBackend, ChannelKind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one OpenSemCom experiment config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--regime", choices=[regime.value for regime in BenchmarkRegime])
    parser.add_argument("--users", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--traces", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw = _load_config(args.config)
    regime = args.regime or raw.get("regime", BenchmarkRegime.FULL_OPEN.value)
    manifest = args.manifest or raw["dataset_manifest"]
    samples = args.samples if args.samples is not None else int(raw.get("samples", 256))
    calibration_samples = (
        args.calibration_samples
        if args.calibration_samples is not None
        else int(raw.get("calibration_samples", 128))
    )
    seed = args.seed if args.seed is not None else int(raw.get("seed", 0))
    users = args.users if args.users is not None else int(raw.get("users", 1))
    output_root = Path(args.output_dir or raw.get("output_dir", "runs")).expanduser().resolve()
    run_name = args.run_name or raw.get("name") or f"{regime}_seed{seed}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_summary = validate_manifest(manifest)
    config = _config_from_dict(raw)
    result = run_experiment(
        config=config,
        regime=BenchmarkRegime(regime),
        samples=samples,
        calibration_samples=calibration_samples,
        users=users,
        seed=seed,
        dataset_manifest=manifest,
    )

    payload = {
        "name": run_name,
        "regime": regime,
        "seed": seed,
        "samples": samples,
        "evaluated_samples": len(result.traces),
        "calibration_samples": calibration_samples,
        "users": users,
        "dataset_manifest": str(Path(manifest).expanduser().resolve()),
        "manifest_summary": manifest_summary,
        "metrics": result.metrics,
        "decisions": result.decisions,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_metrics_csv(output_dir / "metrics.csv", payload)
    if args.traces or bool(raw.get("write_traces", False)):
        (output_dir / "traces.json").write_text(json.dumps(result.traces, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configs require PyYAML or use JSON instead.") from exc
    loaded = yaml.safe_load(text)
    return loaded or {}


def _config_from_dict(raw: dict[str, Any]) -> OpenSemComConfig:
    config = OpenSemComConfig(seed=int(raw.get("seed", 0)))
    if "model" in raw:
        config = replace(config, model=_dataclass_from_dict(ModelConfig, raw["model"]))
    if "channel" in raw:
        channel_raw = dict(raw["channel"])
        if "backend" in channel_raw:
            channel_raw["backend"] = ChannelBackend(channel_raw["backend"])
        if "kind" in channel_raw:
            channel_raw["kind"] = ChannelKind(channel_raw["kind"])
        config = replace(config, channel=_dataclass_from_dict(ChannelConfig, channel_raw))
    if "calibration" in raw:
        config = replace(config, calibration=_dataclass_from_dict(CalibrationConfig, raw["calibration"]))
    if "resource_weights" in raw:
        config = replace(config, resource_weights=_dataclass_from_dict(ResourceWeights, raw["resource_weights"]))
    if "detector_weights" in raw:
        config = replace(config, detector_weights=_dataclass_from_dict(DetectorWeights, raw["detector_weights"]))
    if "ablation" in raw:
        config = replace(config, ablation=_dataclass_from_dict(AblationConfig, raw["ablation"]))
    return config


def _dataclass_from_dict(cls, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _write_metrics_csv(path: Path, payload: dict[str, Any]) -> None:
    row = {
        "name": payload["name"],
        "regime": payload["regime"],
        "seed": payload["seed"],
        "samples": payload["samples"],
        "evaluated_samples": payload["evaluated_samples"],
        "calibration_samples": payload["calibration_samples"],
        "users": payload["users"],
        **payload["metrics"],
        **{f"decision_{key}": value for key, value in payload["decisions"].items()},
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()

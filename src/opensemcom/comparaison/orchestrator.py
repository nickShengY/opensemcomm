"""One-run-at-a-time fair comparison orchestrator.

Static baselines use one fixed Sionna transmission. Their sender reduces every
backbone feature to the same number of quantized values before the existing
PHY is called. OpenSemCom remains on its existing scheduler/HARQ path.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np

from opensemcom.benchmark import load_feature_vector
from opensemcom.channels import build_channel
from opensemcom.config import ChannelConfig, OpenSemComConfig
from opensemcom.metrics import MetricsAccumulator
from opensemcom.risk import OpenSemanticRisk, ResourceCostModel
from opensemcom.simulation import OpenSemComSystem
from opensemcom.types import ChannelBackend, Decision, ExperimentResult, SemanticSample

from opensemcom.comparaison.dino_receiver import DinoReceiver
from opensemcom.comparaison.dino_sender import DinoSender
from opensemcom.comparaison.openclip_receiver import OpenclipReceiver
from opensemcom.comparaison.openclip_sender import OpenclipSender
from opensemcom.comparaison.siglip_receiver import SiglipReceiver
from opensemcom.comparaison.siglip_sender import SiglipSender


class ComparisonMethod(str, Enum):
    OPENSEMCOM = "opensemcom"
    DINO = "dino"
    SIGLIP = "siglip"
    OPENCLIP = "openclip"


@dataclass(frozen=True)
class ComparisonConfig:
    """Inputs for one method run over the cohort common to configured methods.

    ``manifests`` must provide every entry named by ``cohort_methods``. When
    omitted, the cohort remains the full four-method comparison. The raw manifest controls row identity and split;
    feature-manifest split labels are intentionally ignored after validation.
    ``payload_blocks`` is the fixed one-shot baseline allocation. With the
    default Sionna configuration, one block is 256 information bits, or 32
    eight-bit values, followed by 512 coded bits and QPSK mapping.
    """

    method: ComparisonMethod
    raw_manifest: str | Path
    manifests: dict[str, str | Path]
    channel: ChannelConfig
    cohort_methods: tuple[ComparisonMethod, ...] | None = None
    seed: int = 0
    payload_blocks: int = 1
    accept_quantile: float = 0.95
    opensemcom_config: OpenSemComConfig | None = None


@dataclass(frozen=True)
class ComparisonRun:
    method: str
    cohort_methods: tuple[str, ...]
    cohort_rows: int
    calibration_rows: int
    evaluation_rows: int
    payload_values: int
    result: ExperimentResult


@dataclass(frozen=True)
class _Row:
    key: tuple[str, ...]
    feature_paths: dict[str, Path]
    label: int
    task: str
    domain: str
    is_unknown: bool
    split: str
    regime: str

    def sample(self, feature_path: Path, input_dim: int) -> SemanticSample:
        return SemanticSample(
            x=load_feature_vector(feature_path, input_dim),
            y=self.label,
            task=self.task,
            domain=self.domain,
            is_unknown=self.is_unknown,
            context={"comparison_key": self.key, "split": self.split, "regime": self.regime},
        )


class ComparisonOrchestrator:
    """Runs one configured method; it never chooses a backbone per sample."""

    _REQUIRED_MANIFESTS = tuple(method.value for method in ComparisonMethod)

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self._validate_cohort_methods()
        self._validate_payload_budget()

    def _cohort_methods(self) -> tuple[ComparisonMethod, ...]:
        return self.config.cohort_methods or tuple(ComparisonMethod)

    def _validate_cohort_methods(self) -> None:
        cohort_methods = self._cohort_methods()
        if not cohort_methods:
            raise ValueError("cohort_methods must not be empty.")
        if self.config.method not in cohort_methods:
            raise ValueError("The selected method must be part of cohort_methods.")
        if len(set(cohort_methods)) != len(cohort_methods):
            raise ValueError("cohort_methods must not contain duplicates.")

    @property
    def payload_values(self) -> int:
        return self.config.payload_blocks * self.config.channel.sionna_ldpc_info_bits // self.config.channel.sionna_quantization_bits

    def run(self) -> ComparisonRun:
        rows = self._load_common_rows()
        calibration = [row for row in rows if row.split == "calibration"]
        evaluation = [row for row in rows if row.split == "eval"]
        if not calibration or not evaluation:
            raise ValueError("The shared raw-manifest cohort requires both calibration and eval rows.")
        if self.config.method == ComparisonMethod.OPENSEMCOM:
            result = self._run_opensemcom(calibration, evaluation)
        else:
            result = self._run_static(calibration, evaluation)
        return ComparisonRun(
            method=self.config.method.value,
            cohort_methods=tuple(method.value for method in self._cohort_methods()),
            cohort_rows=len(rows),
            calibration_rows=len(calibration),
            evaluation_rows=len(evaluation),
            payload_values=self.payload_values,
            result=result,
        )

    def _run_static(self, calibration: list[_Row], evaluation: list[_Row]) -> ExperimentResult:
        method = self.config.method.value
        sender, receiver = self._static_adapters()
        config = self.config.opensemcom_config or OpenSemComConfig(seed=self.config.seed, channel=self.config.channel)
        calibration_features = [self._load_feature(row.feature_paths[method]) for row in calibration]
        calibration_open = [self._declared_open_row(row, config) for row in calibration]
        known_calibration_features = [feature for feature, is_open in zip(calibration_features, calibration_open) if not is_open]
        sender.fit(known_calibration_features)
        calibration_payloads = [sender.encode(feature) for feature in calibration_features]
        labels = [row.label for row in calibration]
        receiver.fit(calibration_payloads, labels, calibration_open)

        channel = self._build_channel()
        received_calibration = [channel.transmit(payload).received for payload in calibration_payloads]
        receiver.calibrate(received_calibration, labels, calibration_open)

        risk = OpenSemanticRisk(config.risk_weights, ResourceCostModel(config.resource_weights))
        metrics = MetricsAccumulator()
        traces = []
        for index, row in enumerate(evaluation):
            payload = sender.encode(self._load_feature(row.feature_paths[method]))
            observation = channel.transmit(payload)
            output = receiver.receive(observation.received, observation.state)
            sample = row.sample(row.feature_paths[method], input_dim=payload.size)
            declared_open = self._declared_open_row(row, config)
            if declared_open:
                output = replace(
                    output,
                    decision=Decision.REJECT_OPEN,
                    features={**output.features, "task_open_gate": 1.0},
                )
            breakdown = risk.breakdown(
                sample=sample,
                y_hat=output.y_hat,
                decision=output.decision,
                action=output.action,
                known_tasks=config.model.train_tasks,
                adaptation_harm=0.0,
                calibration_error=0.0,
            )
            metrics.add(
                sample,
                output,
                breakdown,
                risk.total(breakdown),
                ood_label=self._is_open_exposure(sample, config),
            )
            traces.append(
                {
                    "index": index,
                    "comparison_key": row.key,
                    "method": method,
                    "y": sample.y,
                    "y_hat": output.y_hat,
                    "decision": output.decision.value,
                    "risk_score": output.risk_score,
                    "payload_values": int(payload.size),
                    "payload_information_bits": int(payload.size * self.config.channel.sionna_quantization_bits),
                    "payload_ldpc_blocks": int(self.config.payload_blocks),
                    "features": output.features,
                    "task": sample.task,
                    "domain": sample.domain,
                    "declared_open": declared_open,
                }
            )
        return ExperimentResult(metrics=metrics.summarize(), decisions=dict(metrics.decision_counts), traces=traces)

    def _run_opensemcom(self, calibration: list[_Row], evaluation: list[_Row]) -> ExperimentResult:
        base = self.config.opensemcom_config or OpenSemComConfig(seed=self.config.seed, channel=self.config.channel)
        channel_config = self._channel_config_with_seed()
        system_config = replace(base, seed=self.config.seed, channel=channel_config)
        channel = build_channel(channel_config, np.random.default_rng(self.config.seed + 100))
        system = OpenSemComSystem(system_config)
        manifest_name = ComparisonMethod.OPENSEMCOM.value
        calibration_samples = [row.sample(row.feature_paths[manifest_name], system_config.model.input_dim) for row in calibration]
        evaluation_samples = [row.sample(row.feature_paths[manifest_name], system_config.model.input_dim) for row in evaluation]
        system.calibrate(calibration_samples, channel)
        return system.run(evaluation_samples, channel)

    def _static_adapters(self):
        if self.config.method == ComparisonMethod.DINO:
            return DinoSender(self.payload_values, self.config.seed), DinoReceiver(self.config.accept_quantile)
        if self.config.method == ComparisonMethod.SIGLIP:
            return SiglipSender(self.payload_values, self.config.seed), SiglipReceiver(self.config.accept_quantile)
        if self.config.method == ComparisonMethod.OPENCLIP:
            return OpenclipSender(self.payload_values, self.config.seed), OpenclipReceiver(self.config.accept_quantile)
        raise ValueError(f"{self.config.method.value} is not a static baseline.")

    def _load_common_rows(self) -> list[_Row]:
        cohort_methods = tuple(method.value for method in self._cohort_methods())
        missing = set(cohort_methods) - set(self.config.manifests)
        if missing:
            raise ValueError(f"Comparison manifests are missing: {', '.join(sorted(missing))}")
        raw = _read_manifest(self.config.raw_manifest)
        raw_by_key = _index_rows(raw, str(self.config.raw_manifest))
        features_by_method = {
            method: _index_rows(_read_manifest(self.config.manifests[method]), str(self.config.manifests[method]))
            for method in cohort_methods
        }
        common = set(raw_by_key)
        for rows in features_by_method.values():
            common &= set(rows)
        if not common:
            raise ValueError("No raw rows are shared by every comparison manifest.")
        output = []
        for key in sorted(common):
            source = raw_by_key[key]
            _validate_metadata(source, features_by_method, key)
            output.append(
                _Row(
                    key=key,
                    feature_paths={
                        method: _resolve_source(features_by_method[method][key], Path(self.config.manifests[method]))
                        for method in cohort_methods
                    },
                    label=int(source["label"]),
                    task=source["task"],
                    domain=source["domain"],
                    is_unknown=_parse_bool(source["is_unknown"]),
                    split=source.get("split") or "eval",
                    regime=source.get("regime") or "",
                )
            )
        return output

    def _build_channel(self):
        return build_channel(self._channel_config_with_seed(), np.random.default_rng(self.config.seed + 100))

    def _channel_config_with_seed(self) -> ChannelConfig:
        if self.config.channel.backend == ChannelBackend.SIONNA and self.config.channel.sionna_seed is None:
            return replace(self.config.channel, sionna_seed=self.config.seed + 100)
        return self.config.channel

    @staticmethod
    def _declared_open_row(row: _Row, config: OpenSemComConfig) -> bool:
        """Respect the benchmark's explicit mixed-open task/domain policy."""
        return row.is_unknown or (
            config.calibration.mixed_open
            and (
                row.task not in config.model.train_tasks
                or row.domain not in config.model.train_domains
            )
        )

    @staticmethod
    def _is_open_exposure(sample: SemanticSample, config: OpenSemComConfig) -> bool:
        """Use the same task/domain/unknown OOD target as the OpenSemCom path."""
        return (
            sample.is_unknown
            or sample.task not in config.model.train_tasks
            or sample.domain not in config.model.train_domains
        )

    def _validate_payload_budget(self) -> None:
        if self.config.payload_blocks <= 0:
            raise ValueError("payload_blocks must be positive.")
        info_bits = self.config.payload_blocks * self.config.channel.sionna_ldpc_info_bits
        if info_bits % self.config.channel.sionna_quantization_bits:
            raise ValueError("The information-bit budget must be divisible by sionna_quantization_bits.")

    @staticmethod
    def _load_feature(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Comparison feature artifact not found: {path}")
        return np.load(path).reshape(-1).astype(np.float64)


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest = Path(path).expanduser().resolve()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _index_rows(rows: list[dict[str, str]], manifest: str) -> dict[tuple[str, ...], dict[str, str]]:
    indexed = {}
    for row in rows:
        required = {"source_path", "label", "task", "domain", "is_unknown"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"Manifest {manifest} is missing columns: {', '.join(sorted(missing))}")
        key = _row_key(row)
        if key in indexed:
            raise ValueError(f"Manifest {manifest} contains duplicate comparison row key {key}.")
        indexed[key] = row
    return indexed


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("raw_source_path") or row["source_path"],
        row.get("raw_artifact_index") or row.get("artifact_index") or "",
        row.get("regime") or "",
        row.get("task") or "",
        row.get("domain") or "",
        row.get("label") or "",
        str(_parse_bool(row.get("is_unknown") or "")),
    )


def _validate_metadata(source: dict[str, str], features: dict[str, dict[tuple[str, ...], dict[str, str]]], key: tuple[str, ...]) -> None:
    for method, rows in features.items():
        feature = rows[key]
        for field in ("label", "task", "domain", "is_unknown", "regime"):
            if str(feature.get(field, "")) != str(source.get(field, "")):
                raise ValueError(f"{method} manifest metadata does not match raw manifest for key {key}: {field}")


def _resolve_source(row: dict[str, str], manifest: Path) -> Path:
    source = Path(row["source_path"]).expanduser()
    if not source.is_absolute():
        source = manifest.expanduser().resolve().parent / source
    return source.resolve()


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

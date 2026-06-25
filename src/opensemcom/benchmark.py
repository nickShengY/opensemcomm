"""OpenSemCom benchmark manifests."""

from __future__ import annotations

import csv
import pickle
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

import numpy as np

from opensemcom.channels import shifted_channel
from opensemcom.config import ChannelConfig, OpenSemComConfig
from opensemcom.types import ChannelKind, SemanticSample


class BenchmarkRegime(str, Enum):
    CLOSED_ID = "closed-id"
    CHANNEL_OPEN = "channel-open"
    SOURCE_OPEN = "source-open"
    CLASS_OPEN = "class-open"
    TASK_OPEN = "task-open"
    SUPERVISION_LIMITED = "supervision-limited"
    RESOURCE_OPEN = "resource-open"
    FULL_OPEN = "full-open"


@dataclass(frozen=True)
class ManifestRow:
    source_path: Path
    label: int
    task: str
    domain: str
    is_unknown: bool
    split: str = "eval"
    regime: str = ""
    artifact_index: int | None = None
    artifact_key: str = ""
    dataset: str = ""
    label_name: str = ""


class OpenSemComBench:
    """Loads OpenSemCom samples from a dataset manifest."""

    def __init__(
        self,
        config: OpenSemComConfig,
        regime: BenchmarkRegime,
        manifest_path: str | Path,
    ):
        self.config = config
        self.regime = regime
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {self.manifest_path}")
        self.rows = self._read_manifest()

    def samples(self, n: int | None = None, users: int = 1, split: str = "eval") -> list[SemanticSample]:
        rows = [row for row in self.rows if row.split == split or split == "all"]
        if self.regime == BenchmarkRegime.CLOSED_ID:
            rows = [
                row for row in rows
                if not row.is_unknown and row.task in self.config.model.train_tasks
            ]
            regime_rows = [row for row in rows if row.regime in {"", self.regime.value}]
            rows = regime_rows or rows
        else:
            regime_rows = [row for row in rows if row.regime in {"", self.regime.value}]
            rows = regime_rows or rows
            if self.regime in {
                BenchmarkRegime.SOURCE_OPEN,
                BenchmarkRegime.CLASS_OPEN,
                BenchmarkRegime.TASK_OPEN,
            }:
                closed_rows = [
                    row for row in self.rows
                    if row.split == split and row.regime == BenchmarkRegime.CLOSED_ID.value
                ]
                rows = closed_rows + [row for row in rows if row.regime != BenchmarkRegime.CLOSED_ID.value]
        if n is not None:
            rows = rows[:n]
        samples = [self._row_to_sample(row, i, users) for i, row in enumerate(rows)]
        if not samples:
            raise ValueError(f"No samples found for split '{split}' in {self.manifest_path}")
        return samples

    def calibration_samples(self, n: int | None = None) -> list[SemanticSample]:
        rows = [row for row in self.rows if row.split == "calibration"]
        if not rows:
            rows = [row for row in self.rows if row.split in {"train", "eval"} and not row.is_unknown]
        if self.config.calibration.mixed_open:
            rows = self._mixed_calibration_rows(rows, n)
            n = None
        if n is not None:
            rows = rows[:n]
        samples = [self._row_to_sample(row, i, users=1) for i, row in enumerate(rows)]
        if not samples:
            raise ValueError(f"No calibration samples found in {self.manifest_path}")
        return samples

    def _mixed_calibration_rows(self, known_rows: list[ManifestRow], n: int | None) -> list[ManifestRow]:
        open_rows = [
            row for row in self.rows
            if row.split == self.config.calibration.open_split
            and (
                row.is_unknown
                or row.task not in self.config.model.train_tasks
                or row.domain not in self.config.model.train_domains
            )
        ]
        if not open_rows:
            return known_rows[:n] if n is not None else known_rows
        if n is None:
            return known_rows + open_rows
        open_fraction = min(max(float(self.config.calibration.open_fraction), 0.0), 0.95)
        open_n = min(len(open_rows), int(round(n * open_fraction)))
        known_n = max(1, n - open_n)
        selected_known = known_rows[:known_n]
        selected_open = open_rows[:open_n]
        return selected_known + selected_open

    def channel_config(self) -> ChannelConfig:
        base = self.config.channel
        if self.regime == BenchmarkRegime.CHANNEL_OPEN:
            return shifted_channel(base, ChannelKind.RAYLEIGH, snr_delta=-6.0)
        if self.regime == BenchmarkRegime.FULL_OPEN:
            return shifted_channel(base, ChannelKind.INTERFERENCE, snr_delta=-8.0)
        return base

    def _read_manifest(self) -> list[ManifestRow]:
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source_path", "label", "task", "domain", "is_unknown"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                missing_cols = ", ".join(sorted(missing))
                raise ValueError(f"Manifest missing required columns: {missing_cols}")
            rows = []
            for raw in reader:
                source = Path(raw["source_path"])
                if not source.is_absolute():
                    source = self.manifest_path.parent / source
                rows.append(
                    ManifestRow(
                        source_path=source.resolve(),
                        label=int(raw["label"]),
                        task=raw["task"],
                        domain=raw["domain"],
                        is_unknown=_parse_bool(raw["is_unknown"]),
                        split=raw.get("split") or "eval",
                        regime=raw.get("regime") or "",
                        artifact_index=_parse_optional_int(raw.get("artifact_index", "")),
                        artifact_key=raw.get("artifact_key") or "",
                        dataset=raw.get("dataset") or "",
                        label_name=raw.get("label_name") or "",
                    )
                )
        return rows

    def _row_to_sample(self, row: ManifestRow, i: int, users: int) -> SemanticSample:
        x = load_feature_vector(
            row.source_path,
            self.config.model.input_dim,
            artifact_index=row.artifact_index,
            artifact_key=row.artifact_key,
        )
        return SemanticSample(
            x=x,
            y=row.label,
            task=row.task,
            domain=row.domain,
            is_unknown=row.is_unknown,
            channel_state={
                "user_id": float(i % max(users, 1)),
            },
            context={
                "regime": self.regime.value,
                "source_path": str(row.source_path),
                "split": row.split,
                "dataset": row.dataset,
                "label_name": row.label_name,
                "artifact_index": row.artifact_index,
            },
        )


def load_feature_vector(
    path: Path,
    dim: int,
    artifact_index: int | None = None,
    artifact_key: str = "",
) -> np.ndarray:
    """Load a fixed-size feature vector from an existing manifest artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Manifest artifact not found: {path}")
    suffix = path.suffix.lower()
    if artifact_index is not None:
        values = _load_indexed_artifact(path, artifact_index, artifact_key)
    elif suffix == ".npy":
        values = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        key = "features" if "features" in archive else "x"
        if key not in archive:
            raise ValueError(f"NPZ manifest artifact must contain 'features' or 'x': {path}")
        values = archive[key]
    else:
        data = path.read_bytes()
        if not data:
            raise ValueError(f"Manifest artifact is empty: {path}")
        values = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < dim:
        vector = np.pad(vector, (0, dim - vector.size))
    vector = vector[:dim]
    norm = np.linalg.norm(vector)
    return (vector / max(float(norm), 1e-9)).astype(np.float64)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _load_indexed_artifact(path: Path, index: int, artifact_key: str = "") -> np.ndarray:
    data = _load_pickle_artifact(str(path))
    key = artifact_key or "data"
    if key not in data:
        raise ValueError(f"Indexed artifact key '{key}' not found in manifest artifact: {path}")
    values = data[key]
    if index < 0 or index >= len(values):
        raise IndexError(f"Artifact index {index} out of range for {path}")
    return np.asarray(values[index])


@lru_cache(maxsize=16)
def _load_pickle_artifact(path: str) -> dict:
    with Path(path).open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="dtype\\(\\): align should be passed")
            loaded = pickle.load(handle, encoding="latin1")
    if not isinstance(loaded, dict):
        raise ValueError(f"Indexed manifest artifact must be a pickle dictionary: {path}")
    normalized = {}
    for key, value in loaded.items():
        normalized[str(key)] = value
    return normalized

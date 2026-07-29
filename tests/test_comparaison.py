import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from opensemcom.comparaison import ComparisonConfig, ComparisonMethod, ComparisonOrchestrator
from opensemcom.config import CalibrationConfig, ChannelConfig, ModelConfig, OpenSemComConfig
from opensemcom.types import ChannelBackend, ChannelKind, SemanticSample


def _write_manifests(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    raw_path = tmp_path / "raw.csv"
    feature_paths = {name: tmp_path / f"{name}.csv" for name in ("opensemcom", "dino", "siglip", "openclip")}
    columns = ["source_path", "label", "task", "domain", "is_unknown", "split", "regime", "artifact_index", "raw_source_path", "raw_artifact_index"]
    raw_rows = []
    feature_rows = {name: [] for name in feature_paths}
    for index in range(12):
        label = index % 2
        split = "calibration" if index < 6 else "eval"
        raw_artifact = tmp_path / f"raw-{index}.bin"
        raw_artifact.write_bytes(bytes([index]))
        raw = {
            "source_path": str(raw_artifact),
            "label": str(label),
            "task": "classification",
            "domain": "unit",
            "is_unknown": "false",
            "split": split,
            "regime": "closed-id",
            "artifact_index": str(index),
            "raw_source_path": "",
            "raw_artifact_index": "",
        }
        raw_rows.append(raw)
        for offset, name in enumerate(feature_paths):
            feature = np.full(8, -0.8 if label == 0 else 0.8, dtype=np.float64)
            feature[index % feature.size] += offset * 0.01
            feature_path = tmp_path / f"{name}-{index}.npy"
            np.save(feature_path, feature)
            feature_rows[name].append(
                {
                    **raw,
                    "source_path": str(feature_path),
                    "artifact_index": "",
                    "raw_source_path": str(raw_artifact),
                    "raw_artifact_index": str(index),
                }
            )
    _write_csv(raw_path, columns, raw_rows)
    for name, path in feature_paths.items():
        _write_csv(path, columns, feature_rows[name])
    return raw_path, feature_paths


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _config(tmp_path: Path, method: ComparisonMethod, backend: ChannelBackend = ChannelBackend.NUMPY) -> ComparisonConfig:
    raw, manifests = _write_manifests(tmp_path)
    return ComparisonConfig(
        method=method,
        raw_manifest=raw,
        manifests=manifests,
        channel=ChannelConfig(backend=backend, kind=ChannelKind.AWGN, snr_db=30.0, sionna_seed=17),
        seed=7,
    )


def test_static_baseline_enforces_one_ldpc_block_payload(tmp_path):
    run = ComparisonOrchestrator(_config(tmp_path, ComparisonMethod.DINO)).run()

    assert run.method == "dino"
    assert run.payload_values == 32
    assert run.cohort_rows == 12
    assert {trace["payload_values"] for trace in run.result.traces} == {32}
    assert {trace["payload_information_bits"] for trace in run.result.traces} == {256}
    assert {trace["payload_ldpc_blocks"] for trace in run.result.traces} == {1}


@pytest.mark.parametrize(
    "method",
    [ComparisonMethod.DINO, ComparisonMethod.SIGLIP, ComparisonMethod.OPENCLIP],
)
def test_orchestrator_selects_one_static_adapter(tmp_path, method):
    run = ComparisonOrchestrator(_config(tmp_path, method)).run()

    assert run.method == method.value
    assert all(trace["method"] == method.value for trace in run.result.traces)
    assert set(run.result.decisions) <= {"accept", "reject/open"}


SIONNA_AVAILABLE = importlib.util.find_spec("sionna") is not None


@pytest.mark.skipif(not SIONNA_AVAILABLE, reason="Sionna is not installed.")
def test_dino_static_baseline_runs_through_local_sionna(tmp_path):
    run = ComparisonOrchestrator(_config(tmp_path, ComparisonMethod.DINO, ChannelBackend.SIONNA)).run()

    assert len(run.result.traces) == 6
    assert "semantic_goodput" in run.result.metrics
    assert all("phy_payload_bit_error_rate" in trace["features"] for trace in run.result.traces)

def test_orchestrator_supports_a_selected_three_method_cohort(tmp_path):
    raw, manifests = _write_manifests(tmp_path)
    cohort_methods = (
        ComparisonMethod.OPENSEMCOM,
        ComparisonMethod.SIGLIP,
        ComparisonMethod.OPENCLIP,
    )
    run = ComparisonOrchestrator(
        ComparisonConfig(
            method=ComparisonMethod.OPENCLIP,
            raw_manifest=raw,
            manifests={method.value: manifests[method.value] for method in cohort_methods},
            channel=ChannelConfig(backend=ChannelBackend.NUMPY, kind=ChannelKind.AWGN, snr_db=30.0),
            cohort_methods=cohort_methods,
            seed=7,
        )
    ).run()

    assert run.cohort_methods == ("opensemcom", "siglip", "openclip")
    assert run.cohort_rows == 12
    assert run.method == "openclip"


def test_static_ood_target_includes_unseen_task_and_domain(tmp_path):
    orchestrator = ComparisonOrchestrator(_config(tmp_path, ComparisonMethod.DINO))
    config = OpenSemComConfig(
        model=ModelConfig(train_tasks=("classification",), train_domains=("known-domain",)),
    )

    assert orchestrator._is_open_exposure(
        SemanticSample(np.zeros(2), 0, "classification", "other-domain", False), config
    )
    assert orchestrator._is_open_exposure(
        SemanticSample(np.zeros(2), 0, "other-task", "known-domain", False), config
    )
    assert not orchestrator._is_open_exposure(
        SemanticSample(np.zeros(2), 0, "classification", "known-domain", False), config
    )

def test_static_baseline_rejects_declared_open_text_rows(tmp_path):
    raw_path, manifests = _write_manifests(tmp_path)
    paths = [raw_path, *manifests.values()]
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for index in (0, 6):
            rows[index]["task"] = "text-classification"
            rows[index]["domain"] = "ag-news"
        _write_csv(path, fields, rows)

    config = OpenSemComConfig(
        model=ModelConfig(train_tasks=("classification",), train_domains=("unit",)),
        calibration=CalibrationConfig(mixed_open=True),
    )
    run = ComparisonOrchestrator(
        ComparisonConfig(
            method=ComparisonMethod.SIGLIP,
            raw_manifest=raw_path,
            manifests=manifests,
            channel=ChannelConfig(backend=ChannelBackend.NUMPY, kind=ChannelKind.AWGN, snr_db=30.0),
            opensemcom_config=config,
            seed=7,
        )
    ).run()

    text_traces = [trace for trace in run.result.traces if trace["task"] == "text-classification"]
    assert len(text_traces) == 1
    assert text_traces[0]["declared_open"] is True
    assert text_traces[0]["decision"] == "reject/open"
    assert text_traces[0]["features"]["task_open_gate"] == 1.0

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from opensemcom.comparaison import ComparisonConfig, ComparisonMethod, ComparisonOrchestrator
from opensemcom.cli.run_comparaison import _filter_to_regime
from opensemcom.config import CalibrationConfig, ChannelConfig, ModelConfig, OpenSemComConfig
from opensemcom.types import ChannelBackend, ChannelKind, SemanticSample


def _write_manifests(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    raw_path = tmp_path / "raw.csv"
    feature_paths = {name: tmp_path / f"{name}.csv" for name in ("opensemcom", "dino", "siglip", "openclip", "imagebind")}
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
    assert set(run.timing_seconds) == {"cohort_load", "calibration", "evaluation", "total"}
    assert all(value >= 0.0 for value in run.timing_seconds.values())


@pytest.mark.parametrize(
    "method",
    [ComparisonMethod.DINO, ComparisonMethod.SIGLIP, ComparisonMethod.OPENCLIP, ComparisonMethod.IMAGEBIND],
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



@pytest.mark.parametrize(
    ("regime", "expected_kind", "expected_snr"),
    [
        ("closed-id", ChannelKind.AWGN, 24.0),
        ("channel-open", ChannelKind.RAYLEIGH, 18.0),
        ("full-open", ChannelKind.INTERFERENCE, 16.0),
    ],
)
def test_comparison_uses_the_native_regime_channel_definition(tmp_path, regime, expected_kind, expected_snr):
    config = _config(tmp_path, ComparisonMethod.DINO)
    config = ComparisonConfig(
        method=config.method,
        raw_manifest=config.raw_manifest,
        manifests=config.manifests,
        channel=ChannelConfig(backend=ChannelBackend.NUMPY, kind=ChannelKind.AWGN, snr_db=24.0),
        regime=regime,
        seed=config.seed,
    )

    channel = ComparisonOrchestrator(config)._channel_config_with_seed()

    assert channel.kind == expected_kind
    assert channel.snr_db == expected_snr
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

def test_static_baseline_rejects_task_metadata_open_rows(tmp_path):
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
            expected_calibration_known=5,
            expected_calibration_open=1,
        )
    ).run()

    text_traces = [trace for trace in run.result.traces if trace["task"] == "text-classification"]
    assert len(text_traces) == 1
    assert run.calibration_known_rows == 5
    assert run.calibration_open_rows == 1
    assert text_traces[0]["task_domain_open"] is True
    assert text_traces[0]["decision"] == "reject/open"
    assert text_traces[0]["features"]["task_domain_gate"] == 1.0


def test_static_baseline_never_gates_on_evaluation_unknown_label(tmp_path):
    raw_path, manifests = _write_manifests(tmp_path)
    paths = [raw_path, *manifests.values()]
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[6]["is_unknown"] = "true"
        _write_csv(path, fields, rows)

    run = ComparisonOrchestrator(
        ComparisonConfig(
            method=ComparisonMethod.SIGLIP,
            raw_manifest=raw_path,
            manifests=manifests,
            channel=ChannelConfig(backend=ChannelBackend.NUMPY, kind=ChannelKind.AWGN, snr_db=30.0),
            seed=7,
        )
    ).run()

    trace = next(trace for trace in run.result.traces if trace["is_unknown"])
    assert trace["task_domain_open"] is False
    assert "task_domain_gate" not in trace["features"]


def test_mixed_calibration_requires_open_rows_after_manifest_intersection(tmp_path):
    raw_path, manifests = _write_manifests(tmp_path)
    config = OpenSemComConfig(
        model=ModelConfig(train_tasks=("classification",), train_domains=("unit",)),
        calibration=CalibrationConfig(mixed_open=True),
    )
    with pytest.raises(ValueError, match="Mixed calibration requires both known and open"):
        ComparisonOrchestrator(
            ComparisonConfig(
                method=ComparisonMethod.SIGLIP,
                raw_manifest=raw_path,
                manifests=manifests,
                channel=ChannelConfig(backend=ChannelBackend.NUMPY, kind=ChannelKind.AWGN, snr_db=30.0),
                opensemcom_config=config,
                seed=7,
            )
        ).run()

def test_expected_mixed_calibration_counts_are_enforced(tmp_path):
    raw_path, manifests = _write_manifests(tmp_path)
    paths = [raw_path, *manifests.values()]
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[0]["task"] = "text-classification"
        rows[0]["domain"] = "ag-news"
        _write_csv(path, fields, rows)

    config = OpenSemComConfig(
        model=ModelConfig(train_tasks=("classification",), train_domains=("unit",)),
        calibration=CalibrationConfig(mixed_open=True),
    )
    with pytest.raises(ValueError, match="Unexpected open calibration cohort size"):
        ComparisonOrchestrator(
            ComparisonConfig(
                method=ComparisonMethod.SIGLIP,
                raw_manifest=raw_path,
                manifests=manifests,
                channel=ChannelConfig(backend=ChannelBackend.NUMPY, kind=ChannelKind.AWGN, snr_db=30.0),
                expected_calibration_known=5,
                expected_calibration_open=2,
                opensemcom_config=config,
                seed=7,
            )
        ).run()


def test_mixed_deployment_cohorts_add_closed_controls_without_changing_pure_open(tmp_path):
    raw_path, manifests = _write_manifests(tmp_path)
    paths = [raw_path, *manifests.values()]
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for index, regime in {
            6: "closed-id",
            7: "class-open",
            8: "source-open",
            9: "task-open",
            10: "full-open",
            11: "resource-open",
        }.items():
            rows[index]["regime"] = regime
        _write_csv(path, fields, rows)

    pure_dir = tmp_path / "pure"
    pure_dir.mkdir()
    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    full_dir = tmp_path / "full"
    full_dir.mkdir()

    pure_raw, _ = _filter_to_regime(raw_path, manifests, "class-open", pure_dir, "pure-open")
    mixed_raw, _ = _filter_to_regime(raw_path, manifests, "class-open", mixed_dir, "mixed-deployment")
    full_raw, _ = _filter_to_regime(raw_path, manifests, "full-open", full_dir, "mixed-deployment")

    def eval_regimes(path: Path) -> list[str]:
        return [row["regime"] for row in csv.DictReader(path.open(encoding="utf-8")) if row["split"] == "eval"]

    assert eval_regimes(pure_raw) == ["class-open"]
    assert eval_regimes(mixed_raw) == ["closed-id", "class-open"]
    assert eval_regimes(full_raw) == ["closed-id", "class-open", "source-open", "task-open"]

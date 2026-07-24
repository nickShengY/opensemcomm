import ast
import importlib.util

import numpy as np

import pytest

from opensemcom.benchmark import BenchmarkRegime, OpenSemComBench
from opensemcom.config import ChannelConfig, OpenSemComConfig
from opensemcom.channels import ChannelObservation, WirelessChannel
from opensemcom.simulation import OpenSemComSystem, run_experiment
from opensemcom.types import ChannelBackend


def write_manifest(tmp_path):
    plan = (tmp_path.cwd() / "OpenSemCom_Research_Plan.md").resolve()
    readme = (tmp_path.cwd() / "README.md").resolve()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "source_path,label,task,domain,is_unknown,split,regime",
                f"{plan},0,classification,paper,false,calibration,closed-id",
                f"{readme},1,classification,docs,false,calibration,closed-id",
                f"{plan},0,classification,paper,false,eval,closed-id",
                f"{readme},1,classification,docs,false,eval,closed-id",
                f"{plan},6,hazard,paper,true,eval,full-open",
                f"{readme},1,retrieval,docs,false,eval,full-open",
                f"{plan},0,classification,paper,false,eval,full-open",
                f"{readme},6,hazard,docs,true,eval,full-open",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def test_full_open_experiment_runs_and_reports_main_metrics(tmp_path):
    manifest = write_manifest(tmp_path)
    result = run_experiment(
        regime=BenchmarkRegime.FULL_OPEN,
        samples=4,
        calibration_samples=2,
        users=2,
        seed=3,
        dataset_manifest=str(manifest),
    )
    assert "open_semantic_risk" in result.metrics
    assert "open_semantic_outage" in result.metrics
    assert "semantic_goodput" in result.metrics
    assert sum(result.decisions.values()) == 4


def test_closed_id_experiment_has_traces(tmp_path):
    manifest = write_manifest(tmp_path)
    result = run_experiment(
        regime=BenchmarkRegime.CLOSED_ID,
        samples=2,
        calibration_samples=2,
        seed=4,
        dataset_manifest=str(manifest),
    )
    assert len(result.traces) == 2
    assert all("decision" in trace for trace in result.traces)
    assert all("harq_refinement_rounds" in trace for trace in result.traces)
    assert all("harq_transmissions" in trace for trace in result.traces)
    assert all("harq_hit_max_refinements" in trace for trace in result.traces)


def test_manifest_with_utf8_bom_runs_from_windows_tools(tmp_path):
    manifest = write_manifest(tmp_path)
    manifest.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8-sig")
    result = run_experiment(
        regime=BenchmarkRegime.CLOSED_ID,
        samples=2,
        calibration_samples=2,
        seed=4,
        dataset_manifest=str(manifest),
    )
    assert len(result.traces) == 2


def test_calibration_uses_core_detector_fit_and_full_policy_thresholds(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path)
    config = OpenSemComConfig()
    system = OpenSemComSystem(config)
    bench = OpenSemComBench(config, BenchmarkRegime.CLOSED_ID, manifest)
    encoded_layers = []
    original_encode = system.encoder.encode

    def capture_encode(layers, layer_names):
        encoded_layers.append(tuple(layer_names))
        return original_encode(layers, layer_names)

    monkeypatch.setattr(system.encoder, "encode", capture_encode)
    system.calibrate(
        bench.calibration_samples(2),
        WirelessChannel(config.channel, np.random.default_rng(7)),
    )

    assert encoded_layers
    assert ("core",) in encoded_layers
    assert ("core", "refinement", "evidence") in encoded_layers


def test_experiment_reports_resource_usage_metrics(tmp_path):
    manifest = write_manifest(tmp_path)
    result = run_experiment(
        regime=BenchmarkRegime.FULL_OPEN,
        samples=4,
        calibration_samples=2,
        users=2,
        seed=3,
        dataset_manifest=str(manifest),
    )
    for key in (
        "total_bandwidth",
        "avg_bandwidth",
        "bandwidth_per_accepted",
        "bandwidth_per_correct_accepted",
        "goodput_per_bandwidth",
        "total_resource_cost",
        "avg_resource_cost",
        "total_latency",
        "avg_latency",
        "total_repetitions",
        "avg_repetitions",
        "total_harq_refinement_rounds",
        "avg_harq_refinement_rounds",
        "harq_refined_sample_rate",
        "total_harq_transmissions",
        "avg_harq_transmissions",
        "total_harq_full_payload_rounds",
        "avg_harq_full_payload_rounds",
        "harq_full_payload_sample_rate",
        "harq_hit_max_refinements_rate",
    ):
        assert key in result.metrics
    assert result.metrics["total_bandwidth"] >= 0.0
    assert result.metrics["total_resource_cost"] >= 0.0
    assert result.metrics["total_repetitions"] >= 4.0
    assert result.metrics["total_harq_transmissions"] >= 4.0
    assert 0.0 <= result.metrics["harq_refined_sample_rate"] <= 1.0
    assert 0.0 <= result.metrics["harq_hit_max_refinements_rate"] <= 1.0
SIONNA_AVAILABLE = importlib.util.find_spec("sionna") is not None


@pytest.mark.skipif(not SIONNA_AVAILABLE, reason="Sionna is not installed.")
def test_full_open_sionna_experiment_runs(tmp_path):
    manifest = write_manifest(tmp_path)
    config = OpenSemComConfig(
        channel=ChannelConfig(
            backend=ChannelBackend.SIONNA,
            snr_db=16.0,
            sionna_seed=9,
        )
    )
    result = run_experiment(
        config=config,
        regime=BenchmarkRegime.FULL_OPEN,
        samples=4,
        calibration_samples=2,
        users=2,
        seed=3,
        dataset_manifest=str(manifest),
    )
    assert "open_semantic_risk" in result.metrics
    assert "total_harq_transmissions" in result.metrics
    assert len(result.traces) == 4
@pytest.mark.skipif(not SIONNA_AVAILABLE, reason="Sionna is not installed.")
def test_channel_open_sionna_experiment_runs(tmp_path):
    manifest = write_manifest(tmp_path)
    config = OpenSemComConfig(
        channel=ChannelConfig(
            backend=ChannelBackend.SIONNA,
            snr_db=16.0,
            sionna_seed=9,
        )
    )
    result = run_experiment(
        config=config,
        regime=BenchmarkRegime.CHANNEL_OPEN,
        samples=2,
        calibration_samples=2,
        users=2,
        seed=3,
        dataset_manifest=str(manifest),
    )
    assert "open_semantic_risk" in result.metrics
    assert len(result.traces) == 2


def test_calibration_debug_reports_phy_quantiles(tmp_path, monkeypatch, capsys):
    manifest = write_manifest(tmp_path)
    config = OpenSemComConfig()
    system = OpenSemComSystem(config)
    bench = OpenSemComBench(config, BenchmarkRegime.CLOSED_ID, manifest)
    channel = WirelessChannel(config.channel, np.random.default_rng(7))

    def transmit(symbols):
        return ChannelObservation(
            received=np.asarray(symbols, dtype=np.float64),
            state={
                "phy_payload_bit_error_rate": 0.125,
                "phy_ldpc_block_error_rate": 0.25,
                "phy_payload_mse": 0.5,
                "phy_quantization_mse": 0.0625,
            },
        )

    monkeypatch.setattr(channel, "transmit", transmit)
    monkeypatch.setenv("OPENSEMCOM_CALIBRATION_DEBUG", "1")
    system.calibrate(bench.calibration_samples(2), channel)

    output = capsys.readouterr().out
    debug = ast.literal_eval(output.removeprefix("CALIB_DEBUG "))
    assert debug["phy_q"]["phy_payload_bit_error_rate"] == [0.125] * 8
    assert debug["phy_q"]["phy_ldpc_block_error_rate"] == [0.25] * 8
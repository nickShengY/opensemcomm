import ast
import importlib.util

import numpy as np

import pytest

from opensemcom.benchmark import BenchmarkRegime, OpenSemComBench
from opensemcom.config import CalibrationConfig, ChannelConfig, ModelConfig, OpenSemComConfig, ResourceWeights
from opensemcom.channels import ChannelObservation, WirelessChannel
from opensemcom.semantic import PrototypeSemanticDecoder
from opensemcom.simulation import OpenSemComSystem, run_experiment
from opensemcom.types import ChannelBackend, ChannelKind, ResourceAction, SemanticSample


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

def test_stage_aware_calibration_registers_a_policy_for_every_payload_stage(tmp_path):
    manifest = write_manifest(tmp_path)
    config = OpenSemComConfig(calibration=CalibrationConfig(stage_aware=True))
    system = OpenSemComSystem(config)
    bench = OpenSemComBench(config, BenchmarkRegime.CLOSED_ID, manifest)

    system.calibrate(
        bench.calibration_samples(2),
        WirelessChannel(config.channel, np.random.default_rng(7)),
    )

    stages = (
        ("core",),
        ("core", "refinement"),
        ("core", "refinement", "evidence"),
    )
    policies = [system.receiver._policy_for_action(ResourceAction(layers=stage)) for stage in stages]
    assert all(policy.calibrator.fitted for policy in policies)
    assert len({id(policy.calibrator) for policy in policies}) == len(stages)

def test_logistic_core_policy_uses_held_out_calibration_and_can_refine(monkeypatch):
    """Avoid in-sample logistic confidence emptying core conformal sets."""
    rng = np.random.default_rng(9)
    input_dim = 64
    per_class = 16

    def make_sample(label: int) -> SemanticSample:
        features = rng.normal(0.0, 0.6, input_dim)
        width = input_dim // 12
        features[label * width : (label + 1) * width] += 1.2
        return SemanticSample(features, label, "classification", "cifar10", False)

    calibration = [make_sample(label) for label in range(6) for _ in range(per_class)]
    evaluation = [make_sample(label) for label in range(6) for _ in range(4)]
    config = OpenSemComConfig(
        seed=7,
        model=ModelConfig(
            input_dim=input_dim,
            latent_dim=input_dim,
            projection="identity",
            classifier="logistic",
            num_known_classes=6,
            train_tasks=("classification",),
            train_domains=("cifar10",),
        ),
        channel=ChannelConfig(snr_db=100.0),
        calibration=CalibrationConfig(stage_aware=True),
        resource_weights=ResourceWeights(scheduler_resource_penalty=0.60),
    )
    system = OpenSemComSystem(config)
    channel = WirelessChannel(config.channel, np.random.default_rng(11))

    captured = {}
    original_conformal_fit = system.calibrator.fit
    original_selective_fit = system.detector.fit_selective

    def capture_conformal_fit(probabilities, labels):
        captured["conformal_count"] = len(probabilities)
        captured["conformal_labels"] = list(labels)
        return original_conformal_fit(probabilities, labels)

    def capture_selective_fit(items):
        captured["selective_count"] = len(items)
        return original_selective_fit(items)

    monkeypatch.setattr(system.calibrator, "fit", capture_conformal_fit)
    monkeypatch.setattr(system.detector, "fit_selective", capture_selective_fit)
    system.calibrate(calibration, channel)
    result = system.run(evaluation, channel)

    holdout = system._threshold_calibration_indices(calibration)
    assert len(holdout) == 24
    assert all(sum(calibration[index].y == label for index in holdout) == 4 for label in range(6))
    assert captured["conformal_count"] == len(holdout)
    assert all(captured["conformal_labels"].count(label) == 4 for label in range(6))
    assert captured["selective_count"] == len(calibration) - len(holdout)
    assert result.metrics["harq_refined_sample_rate"] > 0.0

def test_channel_open_uses_rayleigh_at_six_db_below_its_base_channel(tmp_path):
    manifest = write_manifest(tmp_path)
    config = OpenSemComConfig(channel=ChannelConfig(snr_db=24.0))
    bench = OpenSemComBench(config, BenchmarkRegime.CHANNEL_OPEN, manifest)

    channel = bench.channel_config()

    assert channel.kind == ChannelKind.RAYLEIGH
    assert channel.snr_db == 18.0
def test_mixed_open_calibration_keeps_text_out_of_known_decoder_fit(monkeypatch):

    rng = np.random.default_rng(31)
    input_dim = 48

    def known_sample(label: int) -> SemanticSample:
        features = rng.normal(0.0, 0.4, input_dim)
        features[label * 4 : (label + 1) * 4] += 1.0
        return SemanticSample(features, label, "classification", "cifar10", False)

    known = [known_sample(label) for label in range(6) for _ in range(4)]
    text_open = [
        SemanticSample(rng.normal(0.0, 0.4, input_dim), index % 4, "text-classification", "ag-news", False)
        for index in range(8)
    ]
    calibration = known + text_open
    config = OpenSemComConfig(
        seed=5,
        model=ModelConfig(
            input_dim=input_dim,
            latent_dim=input_dim,
            projection="identity",
            classifier="logistic",
            num_known_classes=6,
            train_tasks=("classification",),
            train_domains=("cifar10",),
        ),
        channel=ChannelConfig(snr_db=100.0),
        calibration=CalibrationConfig(mixed_open=True),
    )
    system = OpenSemComSystem(config)
    channel = WirelessChannel(config.channel, np.random.default_rng(37))
    captured = {}
    original_fit_prototypes = system.decoder.fit_prototypes
    original_fit_calibration = system.detector.fit_calibration

    def capture_prototypes(latents):
        captured["prototype_latents"] = list(latents)
        return original_fit_prototypes(latents)

    def capture_detector(latents):
        captured["detector_latents"] = list(latents)
        return original_fit_calibration(latents)

    monkeypatch.setattr(system.decoder, "fit_prototypes", capture_prototypes)
    monkeypatch.setattr(system.detector, "fit_calibration", capture_detector)
    system.calibrate(calibration, channel)

    holdout = system._threshold_calibration_indices(calibration)
    assert len(holdout) == 8
    assert sum(index < len(known) for index in holdout) == 6
    assert sum(index >= len(known) for index in holdout) == 2
    assert len(captured["prototype_latents"]) == 18
    assert all(not is_open for _, _, is_open in captured["prototype_latents"])
    assert len(captured["detector_latents"]) == 24
    assert sum(is_open for _, _, is_open in captured["detector_latents"]) == 6


def test_stage_specific_decoder_uses_the_matching_payload_head():
    config = ModelConfig(
        input_dim=4,
        latent_dim=4,
        projection="identity",
        num_known_classes=2,
        classifier="prototype",
        stage_specific_heads=True,
    )
    decoder = PrototypeSemanticDecoder(config, np.random.default_rng(13))
    core = [
        (np.asarray([1.0, 0.0, 0.0, 0.0]), 0),
        (np.asarray([0.0, 1.0, 0.0, 0.0]), 1),
    ]
    full = [
        (np.asarray([0.0, 0.0, 1.0, 0.0]), 0),
        (np.asarray([0.0, 0.0, 0.0, 1.0]), 1),
    ]
    decoder.fit_stage_heads(
        {
            ("core",): core,
            ("core", "refinement", "evidence"): full,
        }
    )

    full_layers = ("core", "refinement", "evidence")
    assert set(decoder.stage_heads) == {("core",), full_layers}
    assert decoder.prototype_book_for(("core",)) is not decoder.prototype_book_for(full_layers)
    assert decoder.prototype_distance(full[0][0], full_layers) == pytest.approx(0.0)
    assert decoder.prototype_distance(full[0][0], ("core",)) > 0.0


def test_stage_specific_calibration_fits_all_progressive_payload_heads(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path)
    config = OpenSemComConfig(
        model=ModelConfig(
            input_dim=32,
            latent_dim=32,
            projection="identity",
            num_known_classes=2,
            classifier="logistic",
            stage_specific_heads=True,
        ),
        calibration=CalibrationConfig(stage_aware=True),
        channel=ChannelConfig(snr_db=100.0),
    )
    system = OpenSemComSystem(config)
    bench = OpenSemComBench(config, BenchmarkRegime.CLOSED_ID, manifest)
    captured = {}
    original_fit_calibration = system.detector.fit_calibration

    def capture_detector_latents(latents):
        captured["detector_latents"] = list(latents)
        return original_fit_calibration(latents)

    monkeypatch.setattr(system.detector, "fit_calibration", capture_detector_latents)
    system.calibrate(bench.calibration_samples(2), WirelessChannel(config.channel, np.random.default_rng(17)))

    assert len(captured["detector_latents"]) == 2
    assert set(system.decoder.stage_heads) == {
        ("core",),
        ("core", "refinement"),
        ("core", "refinement", "evidence"),
    }
    assert all(head[1] is not None for head in system.decoder.stage_heads.values())

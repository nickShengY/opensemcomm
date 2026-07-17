"""End-to-end OpenSemCom system loop."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import os
from typing import Iterable

import numpy as np

from opensemcom.adaptation import SafeAdapter
from opensemcom.benchmark import BenchmarkRegime, OpenSemComBench
from opensemcom.calibration import ConformalCalibrator
from opensemcom.channels import WirelessChannel
from opensemcom.codec import CodecLibrary
from opensemcom.config import OpenSemComConfig
from opensemcom.harq import SemanticHARQ
from opensemcom.metrics import MetricsAccumulator
from opensemcom.receiver import SelectiveSemanticReceiver
from opensemcom.risk import OpenRiskDetector, OpenSemanticRisk, ResourceCostModel
from opensemcom.scheduler import RiskAwareScheduler
from opensemcom.semantic import LayeredSemanticEncoder, PrototypeSemanticDecoder, WorldAwareSemanticParser
from opensemcom.types import Decision, ExperimentResult, SemanticSample, ResourceAction
from opensemcom.types import ChannelKind


class OpenSemComSystem:
    """Coordinates parser, channel, receiver, detector, adapter, HARQ, and scheduler."""

    def __init__(self, config: OpenSemComConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.parser = WorldAwareSemanticParser(config.model, self.rng)
        self.encoder = LayeredSemanticEncoder(self.rng)
        self.decoder = PrototypeSemanticDecoder(config.model, self.rng)
        self.calibrator = ConformalCalibrator(delta=config.calibration.delta)
        self.detector = OpenRiskDetector(config.detector_weights, config.model.train_tasks, config.model.train_domains)
        self.receiver = SelectiveSemanticReceiver(
            self.decoder,
            self.detector,
            self.calibrator,
            config.calibration,
            use_detector=config.ablation.use_detector,
            use_conformal=config.ablation.use_conformal,
        )
        self.adapter = SafeAdapter(self.decoder, config.adaptation)
        self.scheduler = RiskAwareScheduler(config.resource_budget, config.resource_weights)
        self.codec_library = CodecLibrary()
        self.resource_cost = ResourceCostModel(config.resource_weights)
        self.open_risk = OpenSemanticRisk(config.risk_weights, self.resource_cost)
        self.adaptation_buffer: deque[tuple[np.ndarray, int]] = deque(maxlen=128)

    def calibrate(self, samples: Iterable[SemanticSample], channel: WirelessChannel) -> None:
        samples = list(samples)
        latents = []
        threshold_indices = self._threshold_calibration_indices(samples)
        augmentations = max(1, int(self.config.model.channel_augmentations))
        for idx, sample in enumerate(samples):
            if idx in threshold_indices:
                continue
            layers = self.parser.parse(sample)
            open_exposure = self.config.calibration.mixed_open and self._is_open_exposure(sample)
            for layer_names in self._calibration_layer_sets():
                symbols = self.encoder.encode(layers, layer_names)
                for _ in range(augmentations):
                    observation = self._calibration_transmit(channel, symbols)
                    _, probs, latent = self.decoder.decode(observation.received)
                    latents.append((latent, sample.y, open_exposure))
        self.decoder.fit_prototypes(latents)
        self.detector.fit_calibration(latents)
        selective_items = []
        selective_repeats = max(1, min(4, augmentations))
        for sample in samples:
            layers = self.parser.parse(sample)
            open_exposure = self._is_open_exposure(sample)
            for layer_names in self._calibration_layer_sets():
                symbols = self.encoder.encode(layers, layer_names)
                for _ in range(selective_repeats):
                    observation = self._calibration_transmit(channel, symbols)
                    y_hat, probs, latent = self.decoder.decode(observation.received)
                    _, prototype_distance = self.decoder.prototype_book.nearest(latent)
                    _, features = self.detector.score(
                        probabilities=probs,
                        latent=latent,
                        prototype_distance=prototype_distance,
                        reconstruction_error=0.0,
                        channel_state=observation.state,
                        task=sample.task,
                        domain=sample.domain,
                    )
                    unsafe = open_exposure or y_hat != sample.y
                    selective_items.append((probs, features, unsafe))
        self.detector.fit_selective(selective_items)
        probabilities = []
        labels = []
        known_probabilities = []
        known_labels = []
        risk_scores = []
        correct_risk_scores = []
        open_risk_scores = []
        use_threshold_subset = bool(threshold_indices)
        for idx, sample in enumerate(samples):
            layers = self.parser.parse(sample)
            symbols = self.encoder.encode(layers, ("core", "refinement", "evidence"))
            observation = self._calibration_transmit(channel, symbols)
            y_hat, probs, latent = self.decoder.decode(observation.received)
            _, prototype_distance = self.decoder.prototype_book.nearest(latent)
            risk_score, _ = self.detector.score(
                probabilities=probs,
                latent=latent,
                prototype_distance=prototype_distance,
                reconstruction_error=0.0,
                channel_state=observation.state,
                task=sample.task,
                domain=sample.domain,
            )
            probabilities.append(probs)
            labels.append(sample.y)
            risk_scores.append(risk_score)
            open_exposure = self._is_open_exposure(sample)
            if open_exposure:
                open_risk_scores.append(risk_score)
            else:
                known_probabilities.append(probs)
                known_labels.append(sample.y)
            threshold_sample = not use_threshold_subset or idx in threshold_indices
            if threshold_sample and not open_exposure and y_hat == sample.y:
                correct_risk_scores.append(risk_score)
        self.calibrator.fit(known_probabilities or probabilities, known_labels or labels)
        accept_quantile = float(np.clip(self.config.calibration.accept_quantile, 0.0, 1.0))
        refine_quantile = float(np.clip(self.config.calibration.refine_quantile, 0.0, 1.0))
        if correct_risk_scores:
            self.receiver.q_accept = float(np.quantile(correct_risk_scores, accept_quantile))
        elif risk_scores:
            self.receiver.q_accept = float(np.quantile(risk_scores, accept_quantile))
        else:
            self.receiver.q_accept = self.config.calibration.accept_quantile
        if self.config.calibration.mixed_open and open_risk_scores:
            open_accept_cap = float(np.quantile(open_risk_scores, self.config.calibration.target_open_outage))
            self.receiver.q_accept = min(self.receiver.q_accept, open_accept_cap)
        if risk_scores:
            self.receiver.q_refine = float(np.quantile(risk_scores, refine_quantile))
        else:
            self.receiver.q_refine = self.config.calibration.refine_quantile
        if self.config.calibration.mixed_open and open_risk_scores:
            self.receiver.q_refine = min(self.receiver.q_refine, float(np.quantile(open_risk_scores, 0.50)))
        self.receiver.q_refine = max(self.receiver.q_accept + 0.05, self.receiver.q_refine)
        if os.environ.get("OPENSEMCOM_CALIBRATION_DEBUG") == "1":
            quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
            print(
                "CALIB_DEBUG",
                {
                    "q_accept": float(self.receiver.q_accept),
                    "q_refine": float(self.receiver.q_refine),
                    "risk_q": np.quantile(risk_scores, quantiles).tolist() if risk_scores else [],
                    "correct_risk_q": np.quantile(correct_risk_scores, quantiles).tolist() if correct_risk_scores else [],
                    "open_risk_q": np.quantile(open_risk_scores, quantiles).tolist() if open_risk_scores else [],
                },
                flush=True,
            )

    def run(self, samples: list[SemanticSample], channel: WirelessChannel) -> ExperimentResult:
        metrics = MetricsAccumulator()
        harq = SemanticHARQ(self.encoder, channel, self.receiver, max_refinements=self.config.calibration.max_refinements)
        traces = []
        covered_so_far: list[bool] = []
        adaptation_harm = 0.0

        for idx, sample in enumerate(samples):
            layers = self.parser.parse(sample)
            codec = self.codec_library.route(sample.domain, sample.task, self.config.channel.kind.value)
            if self.config.ablation.use_scheduler:
                action = self.scheduler.schedule(base_risk=0.5 + codec.risk_bias, codec_ids=(codec.codec_id,))
            else:
                action = ResourceAction(codec_id="fixed", layers=("core",), repetitions=1)
            if self.config.ablation.use_harq:
                output = harq.run(layers, action, sample.task, sample.domain)
            else:
                observation = self._transmit_repeated(
                    channel,
                    self.encoder.encode(layers, action.layers),
                    action.repetitions,
                    action.power,
                )
                output = self.receiver.receive(observation.received, action, observation.state, sample.task, sample.domain)

            self.adaptation_buffer.append((self.encoder.encode(layers, ("core", "refinement", "evidence")), sample.y))
            if self.config.ablation.use_adaptation and idx > 0 and idx % self.config.adaptation.min_buffer == 0:
                adaptation = self.adapter.propose_and_gate(list(self.adaptation_buffer))
                adaptation_harm = adaptation.harm

            covered_so_far.append(sample.y in output.prediction_set)
            calibration_error = self.calibrator.calibration_error(covered_so_far)
            breakdown = self.open_risk.breakdown(
                sample=sample,
                y_hat=output.y_hat,
                decision=output.decision,
                action=output.action,
                known_tasks=self.config.model.train_tasks,
                adaptation_harm=adaptation_harm,
                calibration_error=calibration_error,
            )
            total_risk = self.open_risk.total(breakdown)
            open_exposure = self._is_open_exposure(sample)
            metrics.add(sample, output, breakdown, total_risk, ood_label=open_exposure)
            traces.append(
                {
                    "index": idx,
                    "y": sample.y,
                    "y_hat": output.y_hat,
                    "task": sample.task,
                    "domain": sample.domain,
                    "unknown": sample.is_unknown,
                    "open_exposure": open_exposure,
                    "decision": output.decision.value,
                    "risk_score": output.risk_score,
                    "open_risk": total_risk,
                    "layers": output.action.layers,
                    "codec_id": output.action.codec_id,
                    "repetitions": output.action.repetitions,
                    "confidence": float(np.max(output.probabilities)) if output.probabilities.size else 0.0,
                    "features": output.features,
                    "harq_refinement_rounds": int(output.features.get("harq_refinement_rounds", 0.0)),
                    "harq_transmissions": int(output.features.get("harq_transmissions", 1.0)),
                    "harq_hit_max_refinements": bool(int(output.features.get("harq_hit_max_refinements", 0.0))),
                }
            )

        summary = metrics.summarize(
            adaptation_harm_rate=self.adapter.adaptation_harm_rate,
            certified_accept_rate=self.adapter.certified_accept_rate,
        )
        return ExperimentResult(metrics=summary, decisions=dict(metrics.decision_counts), traces=traces)

    def _calibration_transmit(self, channel: WirelessChannel, symbols: np.ndarray):
        repetitions = 3 if channel.config.kind == ChannelKind.INTERFERENCE else 1
        power = 1.5 if channel.config.kind == ChannelKind.INTERFERENCE else 1.0
        return self._transmit_repeated(channel, symbols, repetitions, power)

    def _transmit_repeated(self, channel: WirelessChannel, symbols: np.ndarray, repetitions: int, power: float = 1.0):
        repetitions = max(1, int(repetitions))
        amplitude = float(np.sqrt(max(power, 1e-9)))
        transmitted = symbols * amplitude
        first = channel.transmit(transmitted)
        first = type(first)(received=first.received / amplitude, state={**first.state, "tx_power": float(power)})
        if repetitions == 1:
            return first
        received = [first.received]
        states = [first.state]
        for _ in range(repetitions - 1):
            obs = channel.transmit(transmitted)
            received.append(obs.received / amplitude)
            states.append({**obs.state, "tx_power": float(power)})
        state = {}
        for key in states[0]:
            values = [s.get(key) for s in states if isinstance(s.get(key), (int, float))]
            state[key] = float(sum(values) / len(values)) if values else states[0][key]
        state["repetitions"] = float(repetitions)
        return type(first)(received=sum(received) / repetitions, state=state)

    def _is_open_exposure(self, sample: SemanticSample) -> bool:
        return (
            sample.is_unknown
            or sample.task not in self.config.model.train_tasks
            or sample.domain not in self.config.model.train_domains
        )

    def _threshold_calibration_indices(self, samples: list[SemanticSample]) -> set[int]:
        if self.config.model.classifier != "torch_mlp":
            return set()
        known_indices = [
            idx for idx, sample in enumerate(samples)
            if not self._is_open_exposure(sample) and 0 <= sample.y < self.config.model.num_known_classes
        ]
        if len(known_indices) < self.config.model.num_known_classes * 4:
            return set()
        grouped: dict[int, list[int]] = {}
        for idx in known_indices:
            grouped.setdefault(samples[idx].y, []).append(idx)
        threshold_indices: set[int] = set()
        for values in grouped.values():
            holdout = max(1, len(values) // 4)
            threshold_indices.update(values[-holdout:])
        return threshold_indices

    def _calibration_layer_sets(self) -> tuple[tuple[str, ...], ...]:
        return (("core", "refinement", "evidence"),)


def run_experiment(
    config: OpenSemComConfig | None = None,
    regime: BenchmarkRegime = BenchmarkRegime.FULL_OPEN,
    samples: int = 256,
    calibration_samples: int = 128,
    users: int = 1,
    seed: int | None = None,
    dataset_manifest: str | None = None,
) -> ExperimentResult:
    if dataset_manifest is None:
        raise ValueError("OpenSemCom requires a dataset manifest; no no-data fallback is available.")
    config = config or OpenSemComConfig()
    if seed is not None:
        config = replace(config, seed=seed)
    bench = OpenSemComBench(config, regime, manifest_path=dataset_manifest)
    channel_config = bench.channel_config()
    channel = WirelessChannel(channel_config, np.random.default_rng(config.seed + 100))
    system = OpenSemComSystem(replace(config, channel=channel_config))
    calibration_stream = bench.calibration_samples(calibration_samples)
    system.calibrate(calibration_stream, channel)
    return system.run(bench.samples(samples, users=users), channel)


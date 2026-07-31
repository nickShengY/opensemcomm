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
from opensemcom.certification import (
    ChannelSupportProfile,
    SelectiveCalibrationItem,
    certify_fixed_policy,
    minimum_zero_error_accepts,
    select_accept_threshold,
)
from opensemcom.channels import WirelessChannel, build_channel
from opensemcom.codec import CodecLibrary
from opensemcom.config import OpenSemComConfig
from opensemcom.harq import SemanticHARQ
from opensemcom.metrics import MetricsAccumulator
from opensemcom.receiver import SelectiveSemanticReceiver
from opensemcom.risk import OpenRiskDetector, OpenSemanticRisk, ResourceCostModel
from opensemcom.scheduler import RiskAwareScheduler
from opensemcom.semantic import LayeredSemanticEncoder, PrototypeSemanticDecoder, WorldAwareSemanticParser
from opensemcom.types import Decision, ExperimentResult, ReliabilityCertificate, SemanticSample, ResourceAction
from opensemcom.types import ChannelKind


_PHY_DIAGNOSTIC_KEYS = (
    "phy_payload_bit_error_rate",
    "phy_ldpc_block_error_rate",
    "phy_payload_mse",
    "phy_quantization_mse",
)


class OpenSemComSystem:
    """Coordinates parser, channel, receiver, detector, adapter, HARQ, and scheduler."""

    def __init__(
        self,
        config: OpenSemComConfig,
        deployment_regime: BenchmarkRegime | None = None,
    ):
        self.config = config
        self.deployment_regime = deployment_regime
        self.rng = np.random.default_rng(config.seed)
        self.parser = WorldAwareSemanticParser(config.model, self.rng)
        self.encoder = LayeredSemanticEncoder(self.rng)
        self.decoder = PrototypeSemanticDecoder(config.model, self.rng)
        self.calibrator = ConformalCalibrator(delta=config.calibration.delta)
        self.detector = OpenRiskDetector(config.detector_weights, config.model.train_tasks, config.model.train_domains)
        self.channel_support = ChannelSupportProfile(
            lower_quantile=config.calibration.channel_support_lower_quantile,
            upper_quantile=config.calibration.channel_support_upper_quantile,
            relative_margin=config.calibration.channel_support_relative_margin,
            absolute_margin=config.calibration.channel_support_absolute_margin,
        )
        self.receiver = SelectiveSemanticReceiver(
            self.decoder,
            self.detector,
            self.calibrator,
            config.calibration,
            channel_support=self.channel_support,
            use_detector=config.ablation.use_detector,
            use_conformal=config.ablation.use_conformal,
        )
        self.adapter = SafeAdapter(self.decoder, config.adaptation)
        self.scheduler = RiskAwareScheduler(config.resource_budget, config.resource_weights)
        self.codec_library = CodecLibrary()
        self.resource_cost = ResourceCostModel(config.resource_weights)
        self.open_risk = OpenSemanticRisk(config.risk_weights, self.resource_cost)
        self.adaptation_proposal_buffer: deque[tuple[np.ndarray, int]] = deque(maxlen=128)
        self.adaptation_validation_buffer: deque[tuple[np.ndarray, int]] = deque(maxlen=128)

    def calibrate(self, samples: Iterable[SemanticSample], channel: WirelessChannel) -> None:
        samples = list(samples)
        model_samples, conformal_samples, threshold_samples, certificate_samples = (
            self._partition_calibration_samples(samples)
        )
        latents = []
        support_states = []
        augmentations = max(1, int(self.config.model.channel_augmentations))
        for sample in model_samples:
            layers = self.parser.parse(sample)
            open_exposure = self.config.calibration.mixed_open and self._is_open_exposure(sample)
            for layer_names in self._calibration_layer_sets():
                symbols = self.encoder.encode(layers, layer_names)
                for _ in range(augmentations):
                    observation = self._calibration_transmit(channel, symbols)
                    support_states.append(observation.state)
                    _, probs, latent = self.decoder.decode(observation.received)
                    latents.append((latent, sample.y, open_exposure))
        self.decoder.fit_prototypes(latents)
        self.detector.fit_calibration(latents)
        self.channel_support.fit(support_states)
        selective_items = []
        selective_repeats = max(1, min(4, augmentations))
        for sample in model_samples:
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
        self.receiver.clear_stage_policies()
        self.receiver.enforce_certificates = False
        policy_layers = self._policy_calibration_layer_sets()
        policy_debug: dict[str, dict[str, object]] = {}
        primary_layers = policy_layers[0]
        for layer_names in policy_layers:
            calibrator = self.calibrator if layer_names == primary_layers else ConformalCalibrator(
                delta=self.config.calibration.delta,
            )
            q_accept, q_refine, debug = self._fit_policy_stage(
                channel,
                layer_names,
                calibrator,
                conformal_samples,
                threshold_samples,
                certificate_samples,
            )
            self.receiver.set_stage_policy(
                layer_names,
                calibrator,
                q_accept,
                q_refine,
                certificate=None,
            )
            policy_debug["+".join(layer_names)] = debug
            if layer_names == primary_layers:
                self.receiver.q_accept = q_accept
                self.receiver.q_refine = q_refine

        global_certificate = self._certify_composed_policy(
            certificate_samples,
            channel,
        )
        for layer_names in policy_layers:
            policy = self.receiver._policy_for_action(ResourceAction(layers=layer_names))
            q_accept = policy.q_accept
            if (
                self.config.calibration.certification_enabled
                and (global_certificate is None or not global_certificate.valid)
            ):
                q_accept = float("-inf")
            self.receiver.set_stage_policy(
                layer_names,
                policy.calibrator,
                q_accept,
                policy.q_refine,
                certificate=global_certificate,
            )
            policy_debug["+".join(layer_names)]["q_accept"] = (
                q_accept if np.isfinite(q_accept) else None
            )
        primary_policy = self.receiver._policy_for_action(
            ResourceAction(layers=primary_layers)
        )
        self.receiver.q_accept = primary_policy.q_accept
        self.receiver.q_refine = primary_policy.q_refine
        self.receiver.enforce_certificates = self.config.calibration.certification_enabled

        if os.environ.get("OPENSEMCOM_CALIBRATION_DEBUG") == "1":
            primary_debug = policy_debug["+".join(primary_layers)]
            print(
                "CALIB_DEBUG",
                {
                    "initial_layers": primary_layers,
                    "composed_certificate": self._certificate_debug(global_certificate),
                    **primary_debug,
                    "stage_policies": policy_debug,
                },
                flush=True,
            )

    def _fit_policy_stage(
        self,
        channel: WirelessChannel,
        layer_names: tuple[str, ...],
        calibrator: ConformalCalibrator,
        conformal_samples: list[SemanticSample],
        threshold_samples: list[SemanticSample],
        certificate_samples: list[SemanticSample],
    ) -> tuple[float, float, dict[str, object]]:
        conformal_records = self._stage_records(conformal_samples, channel, layer_names)
        known_records = [record for record in conformal_records if not record["open_exposure"]]
        fit_records = known_records or conformal_records
        calibrator.fit(
            [record["probabilities"] for record in fit_records],
            [int(record["label"]) for record in fit_records],
        )

        threshold_records = self._stage_records(threshold_samples, channel, layer_names)
        threshold_items = self._selective_items(threshold_records, calibrator)
        risk_scores = [float(record["risk_score"]) for record in threshold_records]
        correct_risk_scores = [
            float(record["risk_score"])
            for record in threshold_records
            if not record["unsafe"]
        ]
        open_risk_scores = [
            float(record["risk_score"])
            for record in threshold_records
            if record["open_exposure"]
        ]
        minimum_accepts = self._minimum_certified_accepts()
        selected_threshold = select_accept_threshold(
            threshold_items,
            target_outage=self.config.calibration.target_open_outage,
            minimum_accepts=minimum_accepts,
            safety_factor=self.config.calibration.threshold_safety_factor,
        )
        if self.config.calibration.certification_enabled:
            q_accept = (
                float(selected_threshold)
                if selected_threshold is not None
                else float("-inf")
            )
        else:
            accept_quantile = float(np.clip(self.config.calibration.accept_quantile, 0.0, 1.0))
            q_accept = (
                float(np.quantile(correct_risk_scores, accept_quantile))
                if correct_risk_scores
                else float("-inf")
            )
        refine_quantile = float(np.clip(self.config.calibration.refine_quantile, 0.0, 1.0))
        if risk_scores:
            q_refine = float(np.quantile(risk_scores, refine_quantile))
        else:
            q_refine = self.config.calibration.refine_quantile
        if self.config.calibration.mixed_open and open_risk_scores:
            q_refine = min(q_refine, float(np.quantile(open_risk_scores, 0.50)))
        if np.isfinite(q_accept):
            q_refine = max(q_accept + 0.05, q_refine)

        quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
        calibration_phy_values = {
            key: [
                float(record["channel_state"][key])
                for record in conformal_records + threshold_records
                if isinstance(record["channel_state"].get(key), (int, float))
            ]
            for key in _PHY_DIAGNOSTIC_KEYS
        }
        return q_accept, q_refine, {
            "q_accept": q_accept if np.isfinite(q_accept) else None,
            "q_refine": q_refine,
            "selected_threshold": selected_threshold,
            "conformal_nonconformity_threshold": float(calibrator.threshold),
            "conformal_probability_cutoff": float(1.0 - calibrator.threshold),
            "split_sizes": {
                "conformal": len(conformal_samples),
                "threshold": len(threshold_samples),
                "certificate": len(certificate_samples),
            },
            "risk_q": np.quantile(risk_scores, quantiles).tolist() if risk_scores else [],
            "correct_risk_q": np.quantile(correct_risk_scores, quantiles).tolist() if correct_risk_scores else [],
            "open_risk_q": np.quantile(open_risk_scores, quantiles).tolist() if open_risk_scores else [],
            "phy_q": {
                key: np.quantile(values, quantiles).tolist() if values else []
                for key, values in calibration_phy_values.items()
            },
        }

    def _stage_records(
        self,
        samples: list[SemanticSample],
        channel: WirelessChannel,
        layer_names: tuple[str, ...],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for sample in samples:
            layers = self.parser.parse(sample)
            symbols = self.encoder.encode(layers, layer_names)
            observation = self._calibration_transmit(channel, symbols)
            y_hat, probabilities, latent = self.decoder.decode(observation.received)
            _, prototype_distance = self.decoder.prototype_book.nearest(latent)
            risk_score, features = self.detector.score(
                probabilities=probabilities,
                latent=latent,
                prototype_distance=prototype_distance,
                reconstruction_error=0.0,
                channel_state=observation.state,
                task=sample.task,
                domain=sample.domain,
            )
            feature_dict = features.as_dict()
            feature_dict["confidence"] = (
                float(np.max(probabilities)) if probabilities.size else 0.0
            )
            supported, violation = self.receiver._channel_support(observation.state)
            feature_dict["channel_supported"] = float(supported)
            feature_dict["channel_support_violation"] = float(violation)
            open_exposure = self._is_open_exposure(sample)
            records.append(
                {
                    "probabilities": probabilities,
                    "label": sample.y,
                    "risk_score": risk_score,
                    "features": feature_dict,
                    "channel_supported": supported,
                    "channel_state": observation.state,
                    "open_exposure": open_exposure,
                    "unsafe": bool(open_exposure or y_hat != sample.y),
                }
            )
        return records

    def _selective_items(
        self,
        records: list[dict[str, object]],
        calibrator: ConformalCalibrator,
    ) -> list[SelectiveCalibrationItem]:
        items = []
        for record in records:
            probabilities = np.asarray(record["probabilities"], dtype=np.float64)
            prediction_set = (
                calibrator.prediction_set(probabilities)
                if self.config.ablation.use_conformal
                else {int(np.argmax(probabilities))}
            )
            features = dict(record["features"])
            supported = bool(record["channel_supported"])
            items.append(
                SelectiveCalibrationItem(
                    risk_score=float(record["risk_score"]),
                    unsafe=bool(record["unsafe"]),
                    eligible=self.receiver.acceptance_eligible(
                        prediction_set,
                        features,
                        supported,
                    ),
                )
            )
        return items

    def _certificate_prerequisite_failure(
        self,
        certificate_samples: list[SemanticSample],
    ) -> str:
        if not self.config.calibration.certification_enabled:
            return ""
        if len(certificate_samples) < self.config.calibration.minimum_certification_samples:
            return "independent certificate split is too small"
        semantic_open_regimes = {
            BenchmarkRegime.SOURCE_OPEN,
            BenchmarkRegime.CLASS_OPEN,
            BenchmarkRegime.TASK_OPEN,
            BenchmarkRegime.FULL_OPEN,
        }
        if (
            self.deployment_regime in semantic_open_regimes
            and not self.config.calibration.mixed_open
        ):
            return "open-regime certification requires mixed-open calibration"
        if (
            self.deployment_regime in semantic_open_regimes
            and not any(self._is_open_exposure(sample) for sample in certificate_samples)
        ):
            return "certificate split contains no semantic-open examples"
        return ""

    def _certify_composed_policy(
        self,
        certificate_samples: list[SemanticSample],
        channel: WirelessChannel,
    ) -> ReliabilityCertificate | None:
        if not self.config.calibration.certification_enabled:
            return None
        prerequisite = self._certificate_prerequisite_failure(certificate_samples)
        if prerequisite:
            return ReliabilityCertificate.unavailable(
                self.config.calibration.target_open_outage,
                1.0 - self.config.calibration.certification_alpha,
                prerequisite,
            )

        outcomes: list[tuple[bool, bool]] = []
        harq = SemanticHARQ(
            self.encoder,
            channel,
            self.receiver,
            max_refinements=self.config.calibration.max_refinements,
        )
        for sample in certificate_samples:
            layers = self.parser.parse(sample)
            codec = self.codec_library.route(
                sample.domain,
                sample.task,
                self.config.channel.kind.value,
            )
            if self.config.ablation.use_scheduler:
                action = self.scheduler.schedule(
                    base_risk=0.5 + codec.risk_bias,
                    codec_ids=(codec.codec_id,),
                )
            else:
                action = ResourceAction(
                    codec_id="fixed",
                    layers=("core",),
                    repetitions=1,
                )
            if self.config.ablation.use_harq:
                output = harq.run(layers, action, sample.task, sample.domain)
            else:
                observation = self._transmit_repeated(
                    channel,
                    self.encoder.encode(layers, action.layers),
                    action.repetitions,
                    action.power,
                )
                output = self.receiver.receive(
                    observation.received,
                    action,
                    observation.state,
                    sample.task,
                    sample.domain,
                )
            outcomes.append(
                (
                    output.decision == Decision.ACCEPT,
                    bool(self._is_open_exposure(sample) or output.y_hat != sample.y),
                )
            )

        thresholds = [
            policy.q_accept
            for policy in self.receiver._stage_policies.values()
            if np.isfinite(policy.q_accept)
        ]
        representative_threshold = max(thresholds) if thresholds else 0.0
        return certify_fixed_policy(
            outcomes,
            target_outage=self.config.calibration.target_open_outage,
            alpha=self.config.calibration.certification_alpha,
            minimum_accepts=self._minimum_certified_accepts(),
            threshold=representative_threshold,
        )

    def _minimum_certified_accepts(self) -> int:
        return max(
            int(self.config.calibration.minimum_certified_accepts),
            minimum_zero_error_accepts(
                self.config.calibration.target_open_outage,
                self.config.calibration.certification_alpha,
            ),
        )

    @staticmethod
    def _certificate_debug(
        certificate: ReliabilityCertificate | None,
    ) -> dict[str, object]:
        if certificate is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "valid": certificate.valid,
            "target_outage": certificate.target_outage,
            "upper_bound": certificate.upper_bound,
            "confidence": certificate.confidence,
            "calibration_samples": certificate.calibration_samples,
            "accepted_samples": certificate.accepted_samples,
            "unsafe_accepted": certificate.unsafe_accepted,
            "reason": certificate.reason,
        }

    def _policy_calibration_layer_sets(self) -> tuple[tuple[str, ...], ...]:
        if self.config.calibration.stage_aware:
            return (
                ("core",),
                ("core", "refinement"),
                ("core", "evidence"),
                ("core", "refinement", "evidence"),
            )
        return (("core", "refinement", "evidence"),)

    def _partition_calibration_samples(
        self,
        samples: list[SemanticSample],
    ) -> tuple[
        list[SemanticSample],
        list[SemanticSample],
        list[SemanticSample],
        list[SemanticSample],
    ]:
        """Create disjoint model, conformal, selection, and certificate splits."""

        if not samples:
            raise ValueError("Calibration requires at least one sample.")
        if not self.config.calibration.certification_enabled:
            return samples, samples, samples, []
        if len(samples) < 4:
            # Tiny smoke tests can still fit the prototype and conformal
            # objects, but cannot produce a reliability certificate.
            return samples, samples, samples, []

        model_fraction = float(np.clip(self.config.calibration.model_fit_fraction, 0.05, 0.80))
        conformal_fraction = float(np.clip(self.config.calibration.conformal_fraction, 0.05, 0.40))
        threshold_fraction = float(np.clip(self.config.calibration.threshold_fraction, 0.05, 0.40))
        if model_fraction + conformal_fraction + threshold_fraction >= 0.95:
            raise ValueError(
                "Calibration split fractions must leave at least 5% for independent certification."
            )
        split_rng = np.random.default_rng(self.config.seed + 7919)
        indices = split_rng.permutation(len(samples)).tolist()
        n_model = max(1, int(len(samples) * model_fraction))
        n_conformal = max(1, int(len(samples) * conformal_fraction))
        n_threshold = max(1, int(len(samples) * threshold_fraction))
        if n_model + n_conformal + n_threshold >= len(samples):
            n_model = max(1, len(samples) - 3)
            n_conformal = 1
            n_threshold = 1
        model_end = n_model
        conformal_end = model_end + n_conformal
        threshold_end = conformal_end + n_threshold

        def take(values: list[int]) -> list[SemanticSample]:
            return [samples[index] for index in values]

        return (
            take(indices[:model_end]),
            take(indices[model_end:conformal_end]),
            take(indices[conformal_end:threshold_end]),
            take(indices[threshold_end:]),
        )

    def run(self, samples: list[SemanticSample], channel: WirelessChannel) -> ExperimentResult:
        metrics = MetricsAccumulator()
        harq = SemanticHARQ(self.encoder, channel, self.receiver, max_refinements=self.config.calibration.max_refinements)
        traces = []

        for idx, sample in enumerate(samples):
            adaptation_harm = 0.0
            adaptation = None
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

            self._collect_verified_adaptation_feedback(sample)
            if (
                self.config.ablation.use_adaptation
                and len(self.adaptation_proposal_buffer) >= self.config.adaptation.min_buffer
                and len(self.adaptation_validation_buffer) >= self.config.adaptation.min_buffer
                and idx > 0
                and idx % self.config.adaptation.min_buffer == 0
            ):
                adaptation = self.adapter.propose_and_gate(
                    list(self.adaptation_proposal_buffer),
                    list(self.adaptation_validation_buffer),
                )
                # Never reuse gate data. The next candidate and its gate are
                # built from fresh verified observations, as required by the
                # sequential concentration argument.
                self.adaptation_proposal_buffer.clear()
                self.adaptation_validation_buffer.clear()
                adaptation_harm = adaptation.harm
                if adaptation.accepted:
                    # The existing threshold certificates describe the
                    # incumbent decoder.  Until an online recertification
                    # split is collected, the adapted decoder may refine or
                    # reject but is not allowed to accept.
                    self.receiver.clear_stage_policies()

            calibration_error = 0.0
            if output.certificate is not None:
                calibration_error = max(
                    0.0,
                    output.certificate.upper_bound - output.certificate.target_outage,
                )
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
                    "prediction_set_size": len(output.prediction_set),
                    "certificate": self._trace_certificate(output.certificate),
                    "features": output.features,
                    "harq_refinement_rounds": int(output.features.get("harq_refinement_rounds", 0.0)),
                    "harq_transmissions": int(output.features.get("harq_transmissions", 1.0)),
                    "harq_hit_max_refinements": bool(int(output.features.get("harq_hit_max_refinements", 0.0))),
                    "adaptation": (
                        {
                            "accepted": adaptation.accepted,
                            "previous_risk": adaptation.previous_risk,
                            "candidate_risk": adaptation.candidate_risk,
                            "candidate_harm": adaptation.candidate_harm,
                            "excess_risk_upper": adaptation.excess_risk_upper,
                            "alpha_t": adaptation.alpha_t,
                            "validation_samples": adaptation.validation_samples,
                            "reason": adaptation.reason,
                        }
                        if adaptation is not None
                        else None
                    ),
                }
            )

        summary = metrics.summarize(
            adaptation_harm_rate=self.adapter.adaptation_harm_rate,
            certified_accept_rate=self.adapter.certified_accept_rate,
        )
        return ExperimentResult(metrics=summary, decisions=dict(metrics.decision_counts), traces=traces)

    def _collect_verified_adaptation_feedback(self, sample: SemanticSample) -> None:
        if not self.config.ablation.use_adaptation:
            return
        label = sample.context.get(self.config.adaptation.verified_feedback_key)
        received = sample.context.get(self.config.adaptation.verified_received_key)
        if label is None or received is None:
            return
        try:
            pair = (np.asarray(received, dtype=np.float64), int(label))
        except (TypeError, ValueError):
            return
        total = len(self.adaptation_proposal_buffer) + len(self.adaptation_validation_buffer)
        if total % 2 == 0:
            self.adaptation_proposal_buffer.append(pair)
        else:
            self.adaptation_validation_buffer.append(pair)

    @staticmethod
    def _trace_certificate(
        certificate: ReliabilityCertificate | None,
    ) -> dict[str, object] | None:
        if certificate is None:
            return None
        return {
            "valid": certificate.valid,
            "target_outage": certificate.target_outage,
            "upper_bound": certificate.upper_bound,
            "confidence": certificate.confidence,
            "accepted_samples": certificate.accepted_samples,
            "unsafe_accepted": certificate.unsafe_accepted,
            "method": certificate.method,
            "reason": certificate.reason,
        }

    def _calibration_transmit(self, channel: WirelessChannel, symbols: np.ndarray):
        repetitions = 3 if channel.config.kind == ChannelKind.INTERFERENCE else 1
        power = 1.5 if channel.config.kind == ChannelKind.INTERFERENCE else 1.0
        return self._transmit_repeated(channel, symbols, repetitions, power)

    def _transmit_repeated(self, channel: WirelessChannel, symbols: np.ndarray, repetitions: int, power: float = 1.0):
        transmit_repeated = getattr(channel, "transmit_repeated", None)
        if transmit_repeated is not None:
            return transmit_repeated(symbols, repetitions, power)
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
        return (
            ("core",),
        )


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
    calibration_channel_config = config.channel
    evaluation_channel_config = bench.channel_config()
    if calibration_channel_config.backend.value == "sionna" and calibration_channel_config.sionna_seed is None:
        calibration_channel_config = replace(
            calibration_channel_config,
            sionna_seed=config.seed + 100,
        )
    if evaluation_channel_config.backend.value == "sionna" and evaluation_channel_config.sionna_seed is None:
        evaluation_channel_config = replace(
            evaluation_channel_config,
            sionna_seed=config.seed + 200,
        )
    calibration_channel = build_channel(
        calibration_channel_config,
        np.random.default_rng(config.seed + 100),
    )
    evaluation_channel = build_channel(
        evaluation_channel_config,
        np.random.default_rng(config.seed + 200),
    )
    system = OpenSemComSystem(
        replace(config, channel=calibration_channel_config),
        deployment_regime=regime,
    )
    calibration_stream = bench.calibration_samples(calibration_samples)
    evaluation_stream = bench.samples(samples, users=users)
    _validate_sample_cohorts(calibration_stream, evaluation_stream)
    system.calibrate(calibration_stream, calibration_channel)
    return system.run(
        evaluation_stream,
        evaluation_channel,
    )


def _validate_sample_cohorts(
    calibration_samples: list[SemanticSample],
    evaluation_samples: list[SemanticSample],
) -> None:
    """Fail when the same source artifact appears in calibration and eval."""

    def identity(sample: SemanticSample) -> tuple[str, str]:
        source = str(sample.context.get("source_path", ""))
        artifact_index = str(sample.context.get("artifact_index", ""))
        return source, artifact_index

    calibration_list = [
        identity(sample)
        for sample in calibration_samples
        if identity(sample)[0]
    ]
    evaluation_list = [
        identity(sample)
        for sample in evaluation_samples
        if identity(sample)[0]
    ]
    if len(calibration_list) != len(set(calibration_list)):
        raise ValueError("Duplicate source artifact inside the calibration cohort.")
    if len(evaluation_list) != len(set(evaluation_list)):
        raise ValueError("Duplicate source artifact inside the evaluation cohort.")
    calibration_ids = set(calibration_list)
    evaluation_ids = set(evaluation_list)
    overlap = calibration_ids & evaluation_ids
    if overlap:
        example = next(iter(overlap))
        raise ValueError(
            "Source leakage between calibration and evaluation cohorts: "
            f"{example}"
        )

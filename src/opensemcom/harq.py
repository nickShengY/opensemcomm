"""Semantic HARQ and progressive refinement."""

from __future__ import annotations

import numpy as np

from opensemcom.channels import WirelessChannel
from opensemcom.receiver import SelectiveSemanticReceiver
from opensemcom.semantic import LayeredSemanticEncoder
from opensemcom.types import Decision, ReceiverOutput, ResourceAction, SemanticLayers


class SemanticHARQ:
    """Transmit progressively richer semantic payloads when refinement is requested."""

    FULL_LAYERS = ("core", "refinement", "evidence")

    def __init__(
        self,
        encoder: LayeredSemanticEncoder,
        channel: WirelessChannel,
        receiver: SelectiveSemanticReceiver,
        max_refinements: int = 3,
    ):
        self.encoder = encoder
        self.channel = channel
        self.receiver = receiver
        self.max_refinements = max(0, int(max_refinements))

    def run(
        self,
        layers: SemanticLayers,
        action: ResourceAction,
        task: str,
        domain: str = "",
        max_refinements: int | None = None,
    ) -> ReceiverOutput:
        refinement_budget = self.max_refinements if max_refinements is None else max(0, int(max_refinements))
        current_action = action
        observation = self._transmit_repeated(
            self.encoder.encode(layers, current_action.layers),
            current_action.repetitions,
            current_action.power,
        )
        output = self.receiver.receive(observation.received, current_action, observation.state, task, domain)
        refinement_rounds = 0
        full_payload_rounds = int(current_action.layers == self.FULL_LAYERS)

        transitions: list[dict[str, object]] = []
        while output.decision == Decision.REFINE and refinement_rounds < refinement_budget:
            previous_output = output
            previous_action = current_action
            current_action = self._next_action(current_action)
            observation = self._transmit_repeated(
                self.encoder.encode(layers, current_action.layers),
                current_action.repetitions,
                current_action.power,
            )
            output = self.receiver.receive(observation.received, current_action, observation.state, task, domain)
            refinement_rounds += 1
            transitions.append(self._transition_record(previous_output, previous_action, output, current_action, refinement_rounds))
            if current_action.layers == self.FULL_LAYERS:
                full_payload_rounds += 1

        hit_max_refinements = output.decision == Decision.REFINE and refinement_rounds >= refinement_budget
        final_decision = Decision.REJECT_OPEN if hit_max_refinements else output.decision
        return self._with_metadata(
            output,
            current_action,
            final_decision,
            refinement_rounds,
            refinement_budget,
            full_payload_rounds,
            hit_max_refinements,
            transitions,
        )

    def _transition_record(
        self,
        current_output: ReceiverOutput,
        current_action: ResourceAction,
        next_output: ReceiverOutput,
        next_action: ResourceAction,
        refinement_round: int,
    ) -> dict[str, object]:
        current_accept, current_refine = self._thresholds_for_action(current_action)
        next_accept, next_refine = self._thresholds_for_action(next_action)
        current_margin = float(current_output.risk_score) - current_accept
        next_margin = float(next_output.risk_score) - next_accept
        return {
            "refinement_round": refinement_round,
            "current_stage_index": self._stage_index(current_action),
            "next_stage_index": self._stage_index(next_action),
            "current_stage": self._stage_name(current_action),
            "next_stage": self._stage_name(next_action),
            "transition_type": self._transition_type(current_action, next_action),
            "current_risk_score": float(current_output.risk_score),
            "next_risk_score": float(next_output.risk_score),
            "current_q_accept": current_accept,
            "next_q_accept": next_accept,
            "current_q_refine": current_refine,
            "next_q_refine": next_refine,
            "raw_score_change": float(current_output.risk_score) - float(next_output.risk_score),
            "current_acceptance_margin": current_margin,
            "next_acceptance_margin": next_margin,
            "margin_change": current_margin - next_margin,
            "current_action": current_output.decision.value,
            "next_action": next_output.decision.value,
        }

    def _thresholds_for_action(self, action: ResourceAction) -> tuple[float, float]:
        accessor = getattr(self.receiver, "thresholds_for_action", None)
        if accessor is None:
            return float("nan"), float("nan")
        q_accept, q_refine = accessor(action)
        return float(q_accept), float(q_refine)

    @staticmethod
    def _stage_name(action: ResourceAction) -> str:
        return "+".join(action.layers)

    @staticmethod
    def _stage_index(action: ResourceAction) -> int:
        return {("core",): 0, ("core", "refinement"): 1, SemanticHARQ.FULL_LAYERS: 2}.get(
            action.layers, len(action.layers) - 1
        )

    @staticmethod
    def _transition_type(current: ResourceAction, next_action: ResourceAction) -> str:
        current_index = SemanticHARQ._stage_index(current)
        next_index = SemanticHARQ._stage_index(next_action)
        if current_index != next_index:
            return f"semantic_expansion_{current_index}_to_{next_index}"
        if current.layers == SemanticHARQ.FULL_LAYERS:
            return "strengthened_complete_payload_2_to_2"
        return "repeated_payload"

    def _next_action(self, action: ResourceAction) -> ResourceAction:
        next_layers = self._next_layers(action.layers)
        if next_layers != action.layers:
            return self._advance_layers(action, next_layers)
        return self._strengthen_full_payload(action)

    def _next_layers(self, current_layers: tuple[str, ...]) -> tuple[str, ...]:
        if current_layers == ("core",):
            return ("core", "refinement")
        if current_layers == ("core", "refinement"):
            return self.FULL_LAYERS
        if current_layers == self.FULL_LAYERS:
            return self.FULL_LAYERS
        return self.FULL_LAYERS

    def _advance_layers(self, action: ResourceAction, next_layers: tuple[str, ...]) -> ResourceAction:
        repetitions = max(1, int(action.repetitions))
        layer_count = len(next_layers)
        return ResourceAction(
            power=max(action.power, 1.0 + 0.25 * (layer_count - 1)),
            bandwidth=max(action.bandwidth, 1.0 + 0.20 * (layer_count - 1)),
            latency=max(action.latency, (1.0 + 0.40 * (layer_count - 1)) * repetitions),
            energy=max(action.energy, (1.0 + 0.35 * (layer_count - 1)) * repetitions),
            compute=max(action.compute, 1.0 + 0.30 * (layer_count - 1)),
            semantic_rate=min(action.semantic_rate, 1.0 / max(layer_count, 1)),
            codec_id=action.codec_id,
            layers=next_layers,
            repetitions=repetitions,
        )

    def _strengthen_full_payload(self, action: ResourceAction) -> ResourceAction:
        repetitions = max(1, int(action.repetitions)) + 1
        return ResourceAction(
            power=action.power + 0.25,
            bandwidth=action.bandwidth,
            latency=max(action.latency + 1.0, 1.8 * repetitions),
            energy=max(action.energy + 0.75, 1.7 * repetitions),
            compute=action.compute + 0.10,
            semantic_rate=action.semantic_rate,
            codec_id=action.codec_id,
            layers=self.FULL_LAYERS,
            repetitions=repetitions,
        )

    def _with_metadata(
        self,
        output: ReceiverOutput,
        action: ResourceAction,
        decision: Decision,
        refinement_rounds: int,
        refinement_budget: int,
        full_payload_rounds: int,
        hit_max_refinements: bool,
        transitions: list[dict[str, object]],
    ) -> ReceiverOutput:
        features = dict(output.features)
        features["harq_refinement_rounds"] = float(refinement_rounds)
        features["harq_transmissions"] = float(refinement_rounds + 1)
        features["harq_max_refinements"] = float(refinement_budget)
        features["harq_full_payload_rounds"] = float(full_payload_rounds)
        features["harq_hit_max_refinements"] = float(hit_max_refinements)
        return ReceiverOutput(
            y_hat=output.y_hat,
            probabilities=output.probabilities,
            prediction_set=output.prediction_set,
            risk_score=output.risk_score,
            decision=decision,
            features=features,
            action=action,
            refinement_transitions=tuple(transitions),
        )

    def _transmit_repeated(self, symbols, repetitions: int, power: float = 1.0):
        transmit_repeated = getattr(self.channel, "transmit_repeated", None)
        if transmit_repeated is not None:
            return transmit_repeated(symbols, repetitions, power)
        repetitions = max(1, int(repetitions))
        amplitude = float(np.sqrt(max(power, 1e-9)))
        transmitted = symbols * amplitude
        first = self.channel.transmit(transmitted)
        first = type(first)(received=first.received / amplitude, state={**first.state, "tx_power": float(power)})
        if repetitions == 1:
            return first
        received = [first.received]
        states = [first.state]
        for _ in range(repetitions - 1):
            obs = self.channel.transmit(transmitted)
            received.append(obs.received / amplitude)
            states.append({**obs.state, "tx_power": float(power)})
        state = {}
        for key in states[0]:
            values = [s.get(key) for s in states if isinstance(s.get(key), (int, float))]
            state[key] = float(sum(values) / len(values)) if values else states[0][key]
        state["repetitions"] = float(repetitions)
        return type(first)(received=sum(received) / repetitions, state=state)

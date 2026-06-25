"""Semantic HARQ and progressive refinement."""

from __future__ import annotations

import numpy as np

from opensemcom.channels import WirelessChannel
from opensemcom.receiver import SelectiveSemanticReceiver
from opensemcom.semantic import LayeredSemanticEncoder
from opensemcom.types import Decision, ReceiverOutput, ResourceAction, SemanticLayers


class SemanticHARQ:
    """Transmit extra semantic layers when the selective receiver asks for refinement."""

    def __init__(self, encoder: LayeredSemanticEncoder, channel: WirelessChannel, receiver: SelectiveSemanticReceiver):
        self.encoder = encoder
        self.channel = channel
        self.receiver = receiver

    def run(self, layers: SemanticLayers, action: ResourceAction, task: str, domain: str = "") -> ReceiverOutput:
        selected = action.layers
        observation = self._transmit_repeated(self.encoder.encode(layers, selected), action.repetitions, action.power)
        output = self.receiver.receive(observation.received, action, observation.state, task, domain)
        if output.decision != Decision.REFINE:
            return output

        refined_layers = ("core", "refinement", "evidence")
        refined_action = ResourceAction(
            power=action.power,
            bandwidth=action.bandwidth,
            latency=action.latency + 1.0,
            energy=action.energy + 0.5,
            compute=action.compute + 0.5,
            semantic_rate=action.semantic_rate / 2.0,
            codec_id=action.codec_id,
            layers=refined_layers,
            repetitions=action.repetitions,
        )
        observation = self._transmit_repeated(
            self.encoder.encode(layers, refined_layers),
            refined_action.repetitions,
            refined_action.power,
        )
        refined = self.receiver.receive(observation.received, refined_action, observation.state, task, domain)
        if refined.decision == Decision.REFINE:
            return ReceiverOutput(
                y_hat=refined.y_hat,
                probabilities=refined.probabilities,
                prediction_set=refined.prediction_set,
                risk_score=refined.risk_score,
                decision=Decision.SEMANTIC_HARQ,
                features=refined.features,
                action=refined_action,
            )
        return refined

    def _transmit_repeated(self, symbols, repetitions: int, power: float = 1.0):
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

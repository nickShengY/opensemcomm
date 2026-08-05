"""Risk-certified safe adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np

from opensemcom.config import AdaptationConfig
from opensemcom.semantic import PrototypeSemanticDecoder
from opensemcom.types import Array


@dataclass(frozen=True)
class AdaptationResult:
    accepted: bool
    previous_risk: float
    candidate_risk: float
    epsilon: float
    harm: float


class SafeAdapter:
    """Lightweight bias adapter with a high-probability non-degradation gate."""

    def __init__(self, decoder: PrototypeSemanticDecoder, config: AdaptationConfig):
        self.decoder = decoder
        self.config = config
        self.candidates = 0
        self.accepted = 0
        self.harm_events = 0

    def propose_and_gate(self, buffer: list[tuple[Array, int]]) -> AdaptationResult:
        self.candidates += 1
        if len(buffer) < self.config.min_buffer:
            return AdaptationResult(False, 1.0, 1.0, 1.0, 0.0)
        previous_risk = self.decoder.risk(buffer)
        bias_delta = self._candidate_bias(buffer)
        candidate = self.decoder.candidate_with_bias(bias_delta)
        candidate_risk = candidate.risk(buffer)
        epsilon = sqrt(log(4.0 * max(self.config.horizon, 1) / self.config.alpha) / (2.0 * len(buffer)))
        passes = candidate_risk + epsilon <= previous_risk - epsilon - self.config.kappa
        harm = max(0.0, candidate_risk - previous_risk)
        if passes:
            self.decoder.apply_bias(bias_delta)
            self.accepted += 1
        if harm > 0.0:
            self.harm_events += 1
        return AdaptationResult(passes, previous_risk, candidate_risk, epsilon, harm)

    def _candidate_bias(self, buffer: list[tuple[Array, int]]) -> Array:
        errors = []
        for received, y in buffer:
            y_hat, probabilities, latent = self.decoder.decode(received)
            if y_hat != y and float(np.max(probabilities)) >= self.config.pseudo_label_threshold:
                target = self.decoder.prototype_book.centroids[y % len(self.decoder.prototype_book.centroids)]
                errors.append(target - latent)
        if not errors:
            return np.zeros_like(self.decoder.adapter_bias)
        return self.config.update_strength * np.mean(np.asarray(errors), axis=0)

    @property
    def certified_accept_rate(self) -> float:
        return self.accepted / max(self.candidates, 1)

    @property
    def adaptation_harm_rate(self) -> float:
        return self.harm_events / max(self.candidates, 1)

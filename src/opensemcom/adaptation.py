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
    alpha_t: float = 0.0
    validation_samples: int = 0
    excess_risk_upper: float = 1.0
    reason: str = ""
    candidate_harm: float = 0.0


class SafeAdapter:
    """Lightweight bias adapter with a high-probability non-degradation gate."""

    def __init__(self, decoder: PrototypeSemanticDecoder, config: AdaptationConfig):
        self.decoder = decoder
        self.config = config
        self.candidates = 0
        self.accepted = 0
        self.harm_events = 0

    def propose_and_gate(
        self,
        proposal_buffer: list[tuple[Array, int]],
        validation_buffer: list[tuple[Array, int]] | None = None,
    ) -> AdaptationResult:
        """Propose on one verified split and gate on an independent split.

        The per-update confidence levels follow
        ``alpha_t = alpha / (t(t+1))``.  Their infinite sum is ``alpha``, so a
        union bound controls the probability of ever accepting a harmful
        update over an unbounded deployment horizon.
        """

        self.candidates += 1
        update_index = self.candidates
        alpha_t = self.config.alpha / (update_index * (update_index + 1))
        if len(proposal_buffer) < self.config.min_buffer:
            return AdaptationResult(
                False,
                1.0,
                1.0,
                1.0,
                0.0,
                alpha_t=alpha_t,
                validation_samples=len(validation_buffer or []),
                reason="insufficient verified proposal samples",
            )
        if validation_buffer is None or len(validation_buffer) < self.config.min_buffer:
            return AdaptationResult(
                False,
                1.0,
                1.0,
                1.0,
                0.0,
                alpha_t=alpha_t,
                validation_samples=len(validation_buffer or []),
                reason="independent verified validation split is required",
            )
        previous_risk = self.decoder.risk(validation_buffer)
        bias_delta = self._candidate_bias(proposal_buffer)
        candidate = self.decoder.candidate_with_bias(bias_delta)
        candidate_risk = candidate.risk(validation_buffer)
        # The paired loss difference lies in [-1, 1].  Hoeffding therefore
        # gives P(E[d] > mean(d)+eps) <= alpha_t with this radius.
        epsilon = sqrt(2.0 * log(1.0 / alpha_t) / len(validation_buffer))
        empirical_excess = candidate_risk - previous_risk
        excess_upper = empirical_excess + epsilon
        passes = excess_upper <= -self.config.kappa
        candidate_harm = max(0.0, candidate_risk - previous_risk)
        realized_harm = candidate_harm if passes else 0.0
        if passes:
            self.decoder.apply_bias(bias_delta)
            self.accepted += 1
        if realized_harm > 0.0:
            self.harm_events += 1
        return AdaptationResult(
            passes,
            previous_risk,
            candidate_risk,
            epsilon,
            realized_harm,
            alpha_t=alpha_t,
            validation_samples=len(validation_buffer),
            excess_risk_upper=excess_upper,
            reason="" if passes else "candidate did not pass the sequential non-degradation gate",
            candidate_harm=candidate_harm,
        )

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

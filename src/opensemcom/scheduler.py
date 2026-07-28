"""Risk-aware wireless semantic scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opensemcom.config import ResourceWeights
from opensemcom.risk import ResourceCostModel
from opensemcom.types import ResourceAction, ResourceBudget


@dataclass(frozen=True)
class SchedulingCandidate:
    action: ResourceAction
    expected_risk: float


class RiskAwareScheduler:
    """Greedy constrained scheduler over power, bandwidth, codec, and layers."""

    def __init__(self, budget: ResourceBudget, weights: ResourceWeights):
        self.budget = budget
        self.cost_model = ResourceCostModel(weights)
        self.resource_penalty = max(0.0, float(weights.scheduler_resource_penalty))

    def schedule(self, base_risk: float, codec_ids: tuple[str, ...] = ("default",)) -> ResourceAction:
        candidates = self._candidates(codec_ids)
        feasible = [c for c in candidates if self._feasible(c.action)]
        if not feasible:
            feasible = candidates
        scored = [
            SchedulingCandidate(
                action=c.action,
                expected_risk=max(
                    0.0,
                    base_risk * c.expected_risk + self.resource_penalty * self.cost_model.cost(c.action),
                ),
            )
            for c in feasible
        ]
        return min(scored, key=lambda c: c.expected_risk).action

    def _candidates(self, codec_ids: tuple[str, ...]) -> list[SchedulingCandidate]:
        layer_options = [
            ("core",),
            ("core", "refinement"),
            ("core", "evidence"),
            ("core", "refinement", "evidence"),
        ]
        candidates: list[SchedulingCandidate] = []
        for codec_id in codec_ids:
            for layers in layer_options:
                n = len(layers)
                repetitions = 3 if codec_id == "robust" else 1
                action = ResourceAction(
                    power=1.0 + 0.25 * (n - 1),
                    bandwidth=1.0 + 0.20 * (n - 1),
                    latency=(1.0 + 0.40 * (n - 1)) * repetitions,
                    energy=(1.0 + 0.35 * (n - 1)) * repetitions,
                    compute=1.0 + 0.30 * (n - 1),
                    semantic_rate=1.0 / n,
                    codec_id=codec_id,
                    layers=layers,
                    repetitions=repetitions,
                )
                risk_factor = 1.0 / np.sqrt(n * repetitions)
                candidates.append(SchedulingCandidate(action=action, expected_risk=float(risk_factor)))
        return candidates

    def _feasible(self, action: ResourceAction) -> bool:
        return (
            action.power <= self.budget.max_power
            and action.bandwidth <= self.budget.max_bandwidth
            and action.latency <= self.budget.max_latency
            and action.energy <= self.budget.max_energy
            and action.compute <= self.budget.max_compute
        )

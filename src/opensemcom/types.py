"""Shared data records for OpenSemCom."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class Decision(str, Enum):
    """Reliability-control actions emitted by the semantic receiver."""

    ACCEPT = "accept"
    REFINE = "refine"
    SEMANTIC_HARQ = "semantic-HARQ"
    ADAPT = "adapt"
    REJECT_OPEN = "reject/open"
    FALLBACK = "fallback"


class ChannelKind(str, Enum):
    """Supported wireless channel stressors."""

    AWGN = "awgn"
    RAYLEIGH = "rayleigh"
    RICIAN = "rician"
    MIMO = "mimo"
    BLOCKAGE = "blockage"
    DOPPLER = "doppler"
    BURST = "burst"
    INTERFERENCE = "interference"


class ChannelBackend(str, Enum):
    """Supported channel simulation backends."""

    NUMPY = "numpy"
    SIONNA = "sionna"


@dataclass(frozen=True)
class ResourceAction:
    """Wireless and edge resource allocation for one semantic transmission."""

    power: float = 1.0
    bandwidth: float = 1.0
    latency: float = 1.0
    energy: float = 1.0
    compute: float = 1.0
    semantic_rate: float = 1.0
    codec_id: str = "default"
    layers: tuple[str, ...] = ("core",)
    repetitions: int = 1


@dataclass(frozen=True)
class ResourceBudget:
    """Expected resource limits from the constrained risk objective."""

    max_power: float = 2.0
    max_bandwidth: float = 2.0
    max_latency: float = 4.0
    max_energy: float = 4.0
    max_compute: float = 4.0


@dataclass(frozen=True)
class ReliabilityCertificate:
    """Finite-sample certificate attached to one selective decision policy.

    A valid certificate bounds the probability of an unsafe accepted decision
    for the fixed policy and deployment distribution described by
    ``assumptions``. It is deliberately explicit so that an empirical risk
    score cannot be mistaken for a statistical guarantee.
    """

    valid: bool
    target_outage: float
    upper_bound: float
    confidence: float
    calibration_samples: int
    accepted_samples: int
    unsafe_accepted: int
    threshold: float
    method: str = "clopper-pearson"
    reason: str = ""
    assumptions: tuple[str, ...] = ()

    @classmethod
    def unavailable(
        cls,
        target_outage: float,
        confidence: float,
        reason: str,
    ) -> "ReliabilityCertificate":
        return cls(
            valid=False,
            target_outage=float(target_outage),
            upper_bound=1.0,
            confidence=float(confidence),
            calibration_samples=0,
            accepted_samples=0,
            unsafe_accepted=0,
            threshold=float("-inf"),
            reason=reason,
        )


@dataclass(frozen=True)
class SemanticSample:
    """One source/task/channel instance in OpenSemCom-Bench."""

    x: Array
    y: int
    task: str
    domain: str
    is_unknown: bool
    channel_state: dict[str, float] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticLayers:
    """Layered semantic payloads."""

    core: Array
    refinement: Array
    evidence: Array
    fallback: Array
    voi: Array

    def select(self, layer_names: tuple[str, ...]) -> Array:
        parts: list[Array] = []
        for name in layer_names:
            if name == "core":
                parts.append(self.core)
            elif name == "refinement":
                parts.append(self.refinement)
            elif name == "evidence":
                parts.append(self.evidence)
            elif name == "fallback":
                parts.append(self.fallback)
            else:
                raise ValueError(f"Unknown semantic layer: {name}")
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


@dataclass(frozen=True)
class ReceiverOutput:
    """Receiver output tuple: prediction, set, risk score, and decision."""

    y_hat: int
    probabilities: Array
    prediction_set: set[int]
    risk_score: float
    decision: Decision
    features: dict[str, float]
    action: ResourceAction
    certificate: ReliabilityCertificate | None = None


@dataclass(frozen=True)
class RiskBreakdown:
    """Terms in the unified open semantic risk."""

    task_loss: float
    unknown_acceptance: float
    task_mismatch: float
    adaptation_harm: float
    calibration_error: float
    resource_cost: float

    @property
    def total_without_weights(self) -> float:
        return (
            self.task_loss
            + self.unknown_acceptance
            + self.task_mismatch
            + self.adaptation_harm
            + self.calibration_error
            + self.resource_cost
        )


@dataclass(frozen=True)
class ExperimentResult:
    """Aggregated metrics and per-sample traces."""

    metrics: dict[str, float]
    decisions: dict[str, int]
    traces: list[dict[str, Any]]

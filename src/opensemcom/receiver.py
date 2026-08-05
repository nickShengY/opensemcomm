"""Selective semantic receiver."""

from __future__ import annotations

from dataclasses import dataclass

from opensemcom.calibration import ConformalCalibrator
from opensemcom.config import CalibrationConfig
from opensemcom.risk import OpenRiskDetector
from opensemcom.semantic import PrototypeSemanticDecoder
from opensemcom.types import Array, Decision, ReceiverOutput, ResourceAction


@dataclass(frozen=True)
class StageDecisionPolicy:
    """Thresholds fitted for one semantic payload stage."""

    calibrator: ConformalCalibrator
    q_accept: float
    q_refine: float


class SelectiveSemanticReceiver:
    """Produces point prediction, prediction set, risk score, and decision."""

    def __init__(
        self,
        decoder: PrototypeSemanticDecoder,
        detector: OpenRiskDetector,
        calibrator: ConformalCalibrator,
        calibration_config: CalibrationConfig,
        use_detector: bool = True,
        use_conformal: bool = True,
    ):
        self.decoder = decoder
        self.detector = detector
        self.calibrator = calibrator
        self.config = calibration_config
        self.use_detector = use_detector
        self.use_conformal = use_conformal
        self.q_accept = calibration_config.accept_quantile
        self.q_refine = calibration_config.refine_quantile
        self._stage_policies: dict[tuple[str, ...], StageDecisionPolicy] = {}

    def clear_stage_policies(self) -> None:
        self._stage_policies.clear()

    def set_stage_policy(
        self,
        layers: tuple[str, ...],
        calibrator: ConformalCalibrator,
        q_accept: float,
        q_refine: float,
    ) -> None:
        self._stage_policies[tuple(layers)] = StageDecisionPolicy(calibrator, q_accept, q_refine)

    def _policy_for_action(self, action: ResourceAction) -> StageDecisionPolicy:
        return self._stage_policies.get(
            tuple(action.layers),
            StageDecisionPolicy(self.calibrator, self.q_accept, self.q_refine),
        )
    def receive(
        self,
        received: Array,
        action: ResourceAction,
        channel_state: dict[str, float],
        task: str,
        domain: str = "",
        reconstruction_error: float = 0.0,
        adaptation_risk: float = 0.0,
    ) -> ReceiverOutput:
        y_hat, probabilities, latent = self.decoder.decode(received)
        _, prototype_distance = self.decoder.prototype_book.nearest(latent)
        if self.use_detector:
            risk_score, features = self.detector.score(
                probabilities=probabilities,
                latent=latent,
                prototype_distance=prototype_distance,
                reconstruction_error=reconstruction_error,
                channel_state=channel_state,
                task=task,
                domain=domain,
                adaptation_risk=adaptation_risk,
            )
            feature_dict = features.as_dict()
            feature_dict["confidence"] = float(max(probabilities)) if probabilities.size else 0.0
        else:
            risk_score = 0.0
            feature_dict = {
                "prediction": 0.0,
                "prototype": 0.0,
                "energy": 0.0,
                "selective": 0.0,
                "unknown": 0.0,
                "openmax": 0.0,
                "vim": 0.0,
                "mahalanobis": 0.0,
                "reconstruction": 0.0,
                "channel": 0.0,
                "task": 0.0,
                "domain": 0.0,
                "adaptation": 0.0,
                "confidence": float(max(probabilities)) if probabilities.size else 0.0,
            }
        for key, value in channel_state.items():
            if key.startswith("phy_"):
                feature_dict[key] = float(value)
        policy = self._policy_for_action(action)
        prediction_set = policy.calibrator.prediction_set(probabilities) if self.use_conformal else {int(y_hat)}
        decision = self._decision(
            risk_score,
            prediction_set,
            feature_dict,
            q_accept=policy.q_accept,
            q_refine=policy.q_refine,
        )
        return ReceiverOutput(
            y_hat=y_hat,
            probabilities=probabilities,
            prediction_set=prediction_set,
            risk_score=risk_score,
            decision=decision,
            features=feature_dict,
            action=action,
        )

    def _decision(
        self,
        risk_score: float,
        prediction_set: set[int],
        features: dict[str, float] | None = None,
        q_accept: float | None = None,
        q_refine: float | None = None,
    ) -> Decision:
        q_accept = self.q_accept if q_accept is None else q_accept
        q_refine = self.q_refine if q_refine is None else q_refine
        if not prediction_set:
            return Decision.REJECT_OPEN
        features = features or {}
        open_exposure = (
            features.get("domain", 0.0) >= 1.0
            or features.get("task", 0.0) >= 1.0
            or features.get("unknown", 0.0) >= 0.50
            or (
                self.detector.weights.openmax > 0.0
                and features.get("openmax", 0.0) >= 0.90
            )
            or (
                self.detector.weights.vim > 0.0
                and features.get("vim", 0.0) >= 0.90
            )
        )
        if len(prediction_set) > 1:
            if risk_score <= q_refine:
                return Decision.REFINE
            return Decision.REJECT_OPEN
        if open_exposure:
            if risk_score <= q_refine:
                return Decision.REFINE
            return Decision.REJECT_OPEN
        if risk_score <= q_accept and features.get("confidence", 0.0) >= self.config.min_accept_confidence:
            return Decision.ACCEPT
        if risk_score <= q_refine:
            return Decision.REFINE
        return Decision.REJECT_OPEN

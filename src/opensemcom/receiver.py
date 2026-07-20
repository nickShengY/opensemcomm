"""Selective semantic receiver."""

from __future__ import annotations

from opensemcom.calibration import ConformalCalibrator
from opensemcom.config import CalibrationConfig
from opensemcom.risk import OpenRiskDetector
from opensemcom.semantic import PrototypeSemanticDecoder
from opensemcom.types import Array, Decision, ReceiverOutput, ResourceAction


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
        prediction_set = self.calibrator.prediction_set(probabilities) if self.use_conformal else {int(y_hat)}
        decision = self._decision(risk_score, prediction_set, feature_dict)
        return ReceiverOutput(
            y_hat=y_hat,
            probabilities=probabilities,
            prediction_set=prediction_set,
            risk_score=risk_score,
            decision=decision,
            features=feature_dict,
            action=action,
        )

    def _decision(self, risk_score: float, prediction_set: set[int], features: dict[str, float] | None = None) -> Decision:
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
            if risk_score <= self.q_refine:
                return Decision.REFINE
            return Decision.REJECT_OPEN
        if open_exposure:
            if risk_score <= self.q_refine:
                return Decision.REFINE
            return Decision.REJECT_OPEN
        if risk_score <= self.q_accept and features.get("confidence", 0.0) >= self.config.min_accept_confidence:
            return Decision.ACCEPT
        if risk_score <= self.q_refine:
            return Decision.REFINE
        return Decision.REJECT_OPEN

"""DINO fixed-transmission receiver adapter."""

from __future__ import annotations

import numpy as np

from opensemcom.types import Decision, ReceiverOutput, ResourceAction


class StaticFeatureReceiver:
    """A supervised classifier with confidence-based OOD risk for one-shot baselines."""

    method_name = "static"

    def __init__(self, accept_quantile: float = 0.95):
        self.accept_quantile = float(np.clip(accept_quantile, 0.0, 1.0))
        self.classifier = None
        self.accept_threshold = 1.0

    def fit(self, payloads: list[np.ndarray], labels: list[int], is_unknown: list[bool]) -> None:
        known = [
            (np.asarray(payload, dtype=np.float64), int(label))
            for payload, label, unknown in zip(payloads, labels, is_unknown)
            if not unknown
        ]
        if not known:
            raise ValueError(f"{self.method_name} receiver requires at least one known calibration row.")
        x, y = zip(*known)
        unique = sorted(set(y))
        if len(unique) == 1:
            self.classifier = _ConstantClassifier(unique[0])
            return
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=19),
        )
        self.classifier.fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))

    def calibrate(self, payloads: list[np.ndarray], labels: list[int], is_unknown: list[bool]) -> None:
        if self.classifier is None:
            raise RuntimeError(f"{self.method_name} receiver must be fitted before calibration.")
        risks = []
        for payload, label, unknown in zip(payloads, labels, is_unknown):
            if unknown:
                continue
            y_hat, probabilities = self._predict(payload)
            if y_hat == int(label):
                risks.append(1.0 - float(np.max(probabilities)))
        if not risks:
            for payload, unknown in zip(payloads, is_unknown):
                if not unknown:
                    _, probabilities = self._predict(payload)
                    risks.append(1.0 - float(np.max(probabilities)))
        self.accept_threshold = float(np.quantile(risks, self.accept_quantile)) if risks else 0.0

    def receive(self, received: np.ndarray, channel_state: dict[str, float]) -> ReceiverOutput:
        y_hat, probabilities = self._predict(received)
        confidence = float(np.max(probabilities)) if probabilities.size else 0.0
        risk = float(np.clip(1.0 - confidence, 0.0, 1.0))
        decision = Decision.ACCEPT if risk <= self.accept_threshold else Decision.REJECT_OPEN
        features = {
            "confidence": confidence,
            "baseline_ood_risk": risk,
            "baseline_accept_threshold": self.accept_threshold,
            "harq_refinement_rounds": 0.0,
            "harq_transmissions": 1.0,
            "harq_full_payload_rounds": 0.0,
            "harq_hit_max_refinements": 0.0,
        }
        features.update({key: float(value) for key, value in channel_state.items() if key.startswith("phy_")})
        return ReceiverOutput(
            y_hat=y_hat,
            probabilities=probabilities,
            prediction_set={y_hat},
            risk_score=risk,
            decision=decision,
            features=features,
            action=ResourceAction(codec_id=f"{self.method_name}-static", layers=("static",), repetitions=1),
        )

    def _predict(self, payload: np.ndarray) -> tuple[int, np.ndarray]:
        if self.classifier is None:
            raise RuntimeError(f"{self.method_name} receiver must be fitted before receiving.")
        x = np.asarray(payload, dtype=np.float64).reshape(1, -1)
        probabilities = np.asarray(self.classifier.predict_proba(x)[0], dtype=np.float64)
        classes = np.asarray(self.classifier.classes_, dtype=np.int64)
        return int(classes[int(np.argmax(probabilities))]), probabilities


class _ConstantClassifier:
    def __init__(self, label: int):
        self.classes_ = np.asarray([label], dtype=np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.ones((len(x), 1), dtype=np.float64)


class DinoReceiver(StaticFeatureReceiver):
    method_name = "dino"

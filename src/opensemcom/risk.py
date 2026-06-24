"""Open semantic risk and channel-task-aware risk detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opensemcom.config import DetectorWeights, ResourceWeights, RiskWeights
from opensemcom.types import Array, Decision, ResourceAction, RiskBreakdown, SemanticSample


class ResourceCostModel:
    def __init__(self, weights: ResourceWeights):
        self.weights = weights

    def cost(self, action: ResourceAction) -> float:
        return (
            self.weights.power * action.power
            + self.weights.bandwidth * action.bandwidth
            + self.weights.latency * action.latency
            + self.weights.energy * action.energy
            + self.weights.compute * action.compute
        )


class OpenSemanticRisk:
    """Unified open semantic risk from the research plan."""

    def __init__(self, risk_weights: RiskWeights, resource_cost: ResourceCostModel):
        self.weights = risk_weights
        self.resource_cost = resource_cost

    def breakdown(
        self,
        sample: SemanticSample,
        y_hat: int,
        decision: Decision,
        action: ResourceAction,
        known_tasks: tuple[str, ...],
        adaptation_harm: float,
        calibration_error: float,
    ) -> RiskBreakdown:
        accepted = decision == Decision.ACCEPT
        task_loss = float(y_hat != sample.y) if accepted else 0.0
        unknown_acceptance = float(sample.is_unknown and accepted)
        task_mismatch = float(sample.task not in known_tasks and accepted and task_loss > 0.0)
        return RiskBreakdown(
            task_loss=task_loss,
            unknown_acceptance=unknown_acceptance,
            task_mismatch=task_mismatch,
            adaptation_harm=float(max(0.0, adaptation_harm)),
            calibration_error=float(max(0.0, calibration_error)),
            resource_cost=self.resource_cost.cost(action),
        )

    def total(self, breakdown: RiskBreakdown) -> float:
        return (
            breakdown.task_loss
            + self.weights.beta_unknown * breakdown.unknown_acceptance
            + self.weights.beta_task * breakdown.task_mismatch
            + self.weights.beta_adapt * breakdown.adaptation_harm
            + self.weights.beta_calibration * breakdown.calibration_error
            + self.weights.beta_resource * breakdown.resource_cost
        )


@dataclass
class RiskFeatures:
    prediction: float
    prototype: float
    energy: float
    selective: float
    unknown: float
    openmax: float
    vim: float
    mahalanobis: float
    reconstruction: float
    channel: float
    task: float
    domain: float
    adaptation: float

    def as_dict(self) -> dict[str, float]:
        return {
            "prediction": self.prediction,
            "prototype": self.prototype,
            "energy": self.energy,
            "selective": self.selective,
            "unknown": self.unknown,
            "openmax": self.openmax,
            "vim": self.vim,
            "mahalanobis": self.mahalanobis,
            "reconstruction": self.reconstruction,
            "channel": self.channel,
            "task": self.task,
            "domain": self.domain,
            "adaptation": self.adaptation,
        }


class OpenRiskDetector:
    """Combines semantic, channel, task, and adaptation evidence."""

    def __init__(self, weights: DetectorWeights, train_tasks: tuple[str, ...], train_domains: tuple[str, ...] = ()):
        self.weights = weights
        self.train_tasks = train_tasks
        self.train_domains = train_domains
        self.class_means: dict[int, Array] = {}
        self.inv_covariance: Array | None = None
        self.mahalanobis_low = 0.0
        self.mahalanobis_high = 1.0
        self.openmax_low = 0.0
        self.openmax_high = 1.0
        self.vim_low = 0.0
        self.vim_high = 1.0
        self.unknown_classifier = None
        self.selective_classifier = None
        self.vim_basis: Array | None = None
        self.global_mean: Array | None = None

    def fit_calibration(self, latents: list[tuple[Array, int] | tuple[Array, int, bool]]) -> None:
        grouped: dict[int, list[Array]] = {}
        vectors = []
        open_targets = []
        known_vectors = []
        for item in latents:
            latent, label, is_open = _unpack_calibration_item(item)
            if is_open:
                open_targets.append(1)
            else:
                open_targets.append(0)
                known_vectors.append(np.asarray(latent, dtype=np.float64))
            if is_open:
                continue
            grouped.setdefault(int(label), []).append(np.asarray(latent, dtype=np.float64))
            vectors.append(np.asarray(latent, dtype=np.float64))
        if len(vectors) < 2:
            return
        self._fit_unknown_classifier(latents)
        self.class_means = {label: np.mean(values, axis=0) for label, values in grouped.items()}
        x = np.asarray(vectors, dtype=np.float64)
        self.global_mean = np.mean(x, axis=0)
        centered = x - np.mean(x, axis=0, keepdims=True)
        covariance = np.cov(centered, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.asarray([[float(covariance)]], dtype=np.float64)
        dim = covariance.shape[0]
        regularizer = max(float(np.trace(covariance)) / max(dim, 1) * 1e-3, 1e-6)
        self.inv_covariance = np.linalg.pinv(covariance + np.eye(dim) * regularizer)
        distances = [
            self._mahalanobis_distance(latent)
            for latent, _, is_open in map(_unpack_calibration_item, latents)
            if not is_open
        ]
        self.mahalanobis_low = float(np.quantile(distances, 0.50))
        self.mahalanobis_high = float(np.quantile(distances, 0.95))
        if self.mahalanobis_high <= self.mahalanobis_low:
            self.mahalanobis_high = self.mahalanobis_low + 1.0
        openmax_distances = [self._openmax_distance(latent, label) for latent, label, is_open in map(_unpack_calibration_item, latents) if not is_open]
        self.openmax_low, self.openmax_high = _robust_range(openmax_distances)
        self._fit_vim(np.asarray(known_vectors, dtype=np.float64))

    def fit_selective(
        self,
        items: list[tuple[Array, RiskFeatures, bool]],
    ) -> None:
        if len(items) < 8:
            return
        labels = np.asarray([int(unsafe) for _, _, unsafe in items], dtype=np.int64)
        if len(set(labels.tolist())) < 2:
            return
        x_values = np.asarray([self._selective_input(probabilities, features) for probabilities, features, _ in items], dtype=np.float64)
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError as exc:
            raise RuntimeError("Selective risk head requires scikit-learn in the scratch environment.") from exc
        positives = max(int(np.sum(labels == 1)), 1)
        negatives = max(int(np.sum(labels == 0)), 1)
        weights = np.asarray([
            (len(labels) / (2 * positives)) if label == 1 else (len(labels) / (2 * negatives))
            for label in labels
        ], dtype=np.float64)
        model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=2,
            random_state=23,
            subsample=0.9,
        )
        model.fit(x_values, labels, sample_weight=weights)
        self.selective_classifier = model

    def score(
        self,
        probabilities: Array,
        latent: Array,
        prototype_distance: float,
        reconstruction_error: float,
        channel_state: dict[str, float],
        task: str,
        domain: str = "",
        adaptation_risk: float = 0.0,
    ) -> tuple[float, RiskFeatures]:
        pred = 1.0 - float(np.max(probabilities))
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        energy = entropy / max(float(np.log(max(probabilities.size, 2))), 1e-12)
        chan = self._channel_shift_score(channel_state)
        task_score = 0.0 if task in self.train_tasks else 1.0
        domain_score = 0.0 if not self.train_domains or domain in self.train_domains else 1.0
        proto = float(prototype_distance / (1.0 + prototype_distance))
        recon = float(reconstruction_error / (1.0 + reconstruction_error))
        maha = self._mahalanobis_score(latent)
        unknown = self._unknown_score(latent)
        openmax = self._openmax_score(latent, int(np.argmax(probabilities)))
        vim = self._vim_score(latent)
        features = RiskFeatures(
            prediction=pred,
            prototype=proto,
            energy=max(0.0, min(1.0, energy)),
            selective=0.0,
            unknown=unknown,
            openmax=openmax,
            vim=vim,
            mahalanobis=maha,
            reconstruction=recon,
            channel=chan,
            task=task_score,
            domain=domain_score,
            adaptation=max(0.0, adaptation_risk),
        )
        selective = self._selective_score(probabilities, features)
        features.selective = selective
        base_score = (
            self.weights.prediction * features.prediction
            + self.weights.prototype * features.prototype
            + self.weights.energy * features.energy
            + self.weights.unknown * features.unknown
            + self.weights.openmax * features.openmax
            + self.weights.vim * features.vim
            + self.weights.mahalanobis * features.mahalanobis
            + self.weights.reconstruction * features.reconstruction
            + self.weights.channel * features.channel
            + self.weights.task * features.task
            + self.weights.domain * features.domain
            + self.weights.adaptation * features.adaptation
        )
        selective_weight = float(np.clip(self.weights.selective, 0.0, 1.0))
        score = (1.0 - selective_weight) * base_score + selective_weight * selective
        return float(score), features

    def _mahalanobis_score(self, latent: Array) -> float:
        if self.inv_covariance is None or not self.class_means:
            return 0.0
        distance = self._mahalanobis_distance(np.asarray(latent, dtype=np.float64))
        scaled = (distance - self.mahalanobis_low) / max(self.mahalanobis_high - self.mahalanobis_low, 1e-9)
        return float(np.clip(scaled, 0.0, 1.0))

    def _unknown_score(self, latent: Array) -> float:
        if self.unknown_classifier is None:
            return 0.0
        probability = self.unknown_classifier.predict_proba(np.asarray(latent, dtype=np.float64).reshape(1, -1))[0]
        classes = list(self.unknown_classifier.classes_)
        if 1 not in classes:
            return 0.0
        return float(np.clip(probability[classes.index(1)], 0.0, 1.0))

    def _selective_score(self, probabilities: Array, features: RiskFeatures) -> float:
        if self.selective_classifier is None:
            return 0.0
        probability = self.selective_classifier.predict_proba(
            self._selective_input(probabilities, features).reshape(1, -1)
        )[0]
        classes = list(self.selective_classifier.classes_)
        if 1 not in classes:
            return 0.0
        return float(np.clip(probability[classes.index(1)], 0.0, 1.0))

    def _selective_input(self, probabilities: Array, features: RiskFeatures) -> Array:
        probs = np.asarray(probabilities, dtype=np.float64)
        sorted_probs = np.sort(probs)
        confidence = float(sorted_probs[-1]) if sorted_probs.size else 0.0
        second = float(sorted_probs[-2]) if sorted_probs.size > 1 else 0.0
        margin = confidence - second
        return np.asarray(
            [
                features.prediction,
                features.prototype,
                features.energy,
                features.unknown,
                features.openmax,
                features.vim,
                features.mahalanobis,
                features.reconstruction,
                features.channel,
                features.task,
                features.domain,
                features.adaptation,
                confidence,
                margin,
            ],
            dtype=np.float64,
        )

    def _openmax_score(self, latent: Array, label: int) -> float:
        distance = self._openmax_distance(np.asarray(latent, dtype=np.float64), label)
        scaled = (distance - self.openmax_low) / max(self.openmax_high - self.openmax_low, 1e-9)
        return float(np.clip(scaled, 0.0, 1.0))

    def _vim_score(self, latent: Array) -> float:
        if self.vim_basis is None or self.global_mean is None:
            return 0.0
        centered = np.asarray(latent, dtype=np.float64) - self.global_mean
        projected = self.vim_basis @ (self.vim_basis.T @ centered)
        residual = float(np.linalg.norm(centered - projected))
        scaled = (residual - self.vim_low) / max(self.vim_high - self.vim_low, 1e-9)
        return float(np.clip(scaled, 0.0, 1.0))

    def _mahalanobis_distance(self, latent: Array) -> float:
        if self.inv_covariance is None or not self.class_means:
            return 0.0
        z = np.asarray(latent, dtype=np.float64)
        distances = []
        for mean in self.class_means.values():
            delta = z - mean
            distances.append(float(delta @ self.inv_covariance @ delta.T))
        return float(np.sqrt(max(min(distances), 0.0)))

    def _openmax_distance(self, latent: Array, label: int) -> float:
        if label in self.class_means:
            mean = self.class_means[label]
        elif self.class_means:
            mean = min(self.class_means.values(), key=lambda value: float(np.linalg.norm(latent - value)))
        else:
            return 0.0
        return float(np.linalg.norm(np.asarray(latent, dtype=np.float64) - mean))

    def _fit_unknown_classifier(self, latents: list[tuple[Array, int] | tuple[Array, int, bool]]) -> None:
        x_values = []
        y_values = []
        for item in latents:
            latent, _, is_open = _unpack_calibration_item(item)
            x_values.append(np.asarray(latent, dtype=np.float64))
            y_values.append(int(is_open))
        if len(set(y_values)) < 2:
            return
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError("Unknown-aware open detector requires scikit-learn in the scratch environment.") from exc
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=17),
        )
        model.fit(np.asarray(x_values, dtype=np.float64), np.asarray(y_values, dtype=np.int64))
        self.unknown_classifier = model

    def _fit_vim(self, known_vectors: Array) -> None:
        if known_vectors.ndim != 2 or known_vectors.shape[0] < 3:
            return
        centered = known_vectors - np.mean(known_vectors, axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        rank = max(1, min(vt.shape[0] - 1, int(np.ceil(vt.shape[0] * 0.8))))
        self.vim_basis = vt[:rank].T
        residuals = []
        for latent in known_vectors:
            residuals.append(self._vim_raw_residual(latent))
        self.vim_low, self.vim_high = _robust_range(residuals)

    def _vim_raw_residual(self, latent: Array) -> float:
        if self.vim_basis is None or self.global_mean is None:
            return 0.0
        centered = np.asarray(latent, dtype=np.float64) - self.global_mean
        projected = self.vim_basis @ (self.vim_basis.T @ centered)
        return float(np.linalg.norm(centered - projected))

    def _channel_shift_score(self, state: dict[str, float]) -> float:
        snr_penalty = max(0.0, (10.0 - state.get("effective_snr_db", 10.0)) / 20.0)
        interference = state.get("interference_power", 0.0)
        blockage = state.get("blockage_probability", 0.0)
        doppler = min(1.0, state.get("doppler_hz", 0.0) / 200.0)
        burst = state.get("burst_probability", 0.0)
        csi = min(1.0, state.get("csi_error", 0.0) * 5.0)
        return float(np.clip(snr_penalty + interference + blockage + doppler + burst + csi, 0.0, 1.0))


def _unpack_calibration_item(item: tuple[Array, int] | tuple[Array, int, bool]) -> tuple[Array, int, bool]:
    if len(item) == 2:
        latent, label = item
        return latent, int(label), False
    latent, label, is_open = item
    return latent, int(label), bool(is_open)


def _robust_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = float(np.quantile(values, 0.50))
    high = float(np.quantile(values, 0.95))
    if high <= low:
        high = low + 1.0
    return low, high

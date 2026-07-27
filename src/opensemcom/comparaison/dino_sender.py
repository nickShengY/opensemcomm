"""DINO fixed-payload sender adapter."""

from __future__ import annotations

import numpy as np


class StaticFeatureSender:
    """Projects any backbone feature into the fixed value budget for one PHY block."""

    method_name = "static"

    def __init__(self, payload_values: int, seed: int):
        if payload_values <= 0:
            raise ValueError("payload_values must be positive.")
        self.payload_values = int(payload_values)
        self.seed = int(seed)
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, features: list[np.ndarray]) -> None:
        if not features:
            raise ValueError(f"{self.method_name} sender requires calibration features.")
        x = np.asarray([np.asarray(value, dtype=np.float64).reshape(-1) for value in features])
        if x.ndim != 2:
            raise ValueError("Feature vectors must have a consistent shape.")
        self.mean = np.mean(x, axis=0)
        centered = x - self.mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[: min(self.payload_values, vt.shape[0])]
        projected = centered @ components.T
        scale = np.quantile(np.abs(projected), 0.995, axis=0) if projected.size else np.empty(0)
        self.components = components
        self.scale = np.maximum(scale, 1e-6)

    def encode(self, feature: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None or self.scale is None:
            raise RuntimeError(f"{self.method_name} sender must be fitted before encoding.")
        x = np.asarray(feature, dtype=np.float64).reshape(-1)
        if x.shape != self.mean.shape:
            raise ValueError(f"Expected feature shape {self.mean.shape}, got {x.shape}.")
        values = ((x - self.mean) @ self.components.T) / self.scale
        payload = np.zeros(self.payload_values, dtype=np.float64)
        payload[: values.size] = np.clip(values, -1.0, 1.0)
        return payload


class DinoSender(StaticFeatureSender):
    method_name = "dino"

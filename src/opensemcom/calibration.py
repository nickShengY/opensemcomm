"""Conformal reliability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opensemcom.types import Array


@dataclass
class ConformalCalibrator:
    """Split conformal calibrator using probability nonconformity scores."""

    delta: float = 0.1
    threshold: float = 1.0
    fitted: bool = False

    def fit(self, probabilities: list[Array], labels: list[int]) -> None:
        if not probabilities:
            self.threshold = 1.0
            self.fitted = True
            return
        scores = []
        for probs, y in zip(probabilities, labels):
            if y < len(probs):
                scores.append(1.0 - float(probs[y]))
            else:
                scores.append(1.0)
        n = len(scores)
        quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - self.delta)) / max(n, 1))
        self.threshold = float(np.quantile(scores, quantile_level, method="higher"))
        self.fitted = True

    def prediction_set(self, probabilities: Array) -> set[int]:
        if not self.fitted:
            threshold = self.threshold
        else:
            threshold = self.threshold
        return {int(i) for i, p in enumerate(probabilities) if (1.0 - float(p)) <= threshold}

    def calibration_error(self, covered: list[bool]) -> float:
        if not covered:
            return 1.0
        coverage = sum(covered) / len(covered)
        return abs(coverage - (1.0 - self.delta))

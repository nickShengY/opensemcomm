import numpy as np

from opensemcom.calibration import ConformalCalibrator
from opensemcom.config import ResourceWeights, RiskWeights
from opensemcom.risk import OpenSemanticRisk, ResourceCostModel
from opensemcom.types import Decision, ResourceAction, SemanticSample


def test_conformal_prediction_set_contains_confident_label():
    calibrator = ConformalCalibrator(delta=0.1)
    calibrator.fit([np.array([0.9, 0.1]), np.array([0.8, 0.2])], [0, 0])
    prediction_set = calibrator.prediction_set(np.array([0.85, 0.15]))
    assert 0 in prediction_set


def test_open_risk_penalizes_unknown_acceptance():
    risk = OpenSemanticRisk(RiskWeights(), ResourceCostModel(ResourceWeights()))
    sample = SemanticSample(
        x=np.zeros(4),
        y=3,
        task="classification",
        domain="urban-day",
        is_unknown=True,
    )
    breakdown = risk.breakdown(
        sample=sample,
        y_hat=0,
        decision=Decision.ACCEPT,
        action=ResourceAction(),
        known_tasks=("classification",),
        adaptation_harm=0.0,
        calibration_error=0.0,
    )
    assert breakdown.unknown_acceptance == 1.0
    assert risk.total(breakdown) > 1.0

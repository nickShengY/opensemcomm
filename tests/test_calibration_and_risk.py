import numpy as np

from opensemcom.calibration import ConformalCalibrator
from opensemcom.config import ResourceWeights, RiskWeights
from opensemcom.metrics import MetricsAccumulator
from opensemcom.risk import OpenSemanticRisk, ResourceCostModel
from opensemcom.types import (
    Decision,
    ReceiverOutput,
    ResourceAction,
    RiskBreakdown,
    SemanticSample,
)


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


def test_goodput_and_coverage_use_evaluated_samples_not_channel_uses():
    metrics = MetricsAccumulator()
    action = ResourceAction(layers=("core", "refinement", "evidence"), repetitions=3)
    breakdown = RiskBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    for decision in (Decision.ACCEPT, Decision.REJECT_OPEN):
        sample = SemanticSample(
            x=np.zeros(4),
            y=0,
            task="classification",
            domain="urban-day",
            is_unknown=False,
        )
        output = ReceiverOutput(
            y_hat=0,
            probabilities=np.asarray([1.0]),
            prediction_set={0},
            risk_score=0.0,
            decision=decision,
            features={"channel_supported": 1.0},
            action=action,
        )
        metrics.add(sample, output, breakdown, total_risk=0.0)

    summary = metrics.summarize()

    assert summary["coverage"] == 0.5
    assert summary["prediction_set_coverage"] == 1.0
    assert summary["semantic_goodput"] == 0.5
    assert summary["goodput_per_channel_use"] < summary["semantic_goodput"]

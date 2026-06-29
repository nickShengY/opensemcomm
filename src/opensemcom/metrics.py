"""Evaluation metrics for OpenSemCom."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from opensemcom.types import Decision, ReceiverOutput, RiskBreakdown, SemanticSample


@dataclass
class MetricsAccumulator:
    risks: list[float] = field(default_factory=list)
    resource_costs: list[float] = field(default_factory=list)
    task_errors: list[float] = field(default_factory=list)
    accepted_errors: list[float] = field(default_factory=list)
    accepted_open_errors: list[float] = field(default_factory=list)
    covered: list[bool] = field(default_factory=list)
    prediction_set_sizes: list[int] = field(default_factory=list)
    channel_uses: int = 0
    correct_accepted: int = 0
    accepted_samples: int = 0
    total_bandwidth: float = 0.0
    total_power: float = 0.0
    total_energy: float = 0.0
    total_latency: float = 0.0
    total_compute: float = 0.0
    total_repetitions: int = 0
    decision_counts: Counter[str] = field(default_factory=Counter)
    ood_scores: list[tuple[float, bool]] = field(default_factory=list)

    def add(
        self,
        sample: SemanticSample,
        output: ReceiverOutput,
        breakdown: RiskBreakdown,
        total_risk: float,
        ood_label: bool | None = None,
    ) -> None:
        accepted = output.decision == Decision.ACCEPT
        error = float(output.y_hat != sample.y)
        self.risks.append(total_risk)
        self.resource_costs.append(breakdown.resource_cost)
        self.task_errors.append(error)
        if accepted:
            self.accepted_samples += 1
            self.accepted_errors.append(error)
            self.accepted_open_errors.append(float(error > 0.0 or breakdown.unknown_acceptance > 0.0 or breakdown.task_mismatch > 0.0))
            if error == 0.0:
                self.correct_accepted += 1
        self.covered.append(sample.y in output.prediction_set)
        self.prediction_set_sizes.append(len(output.prediction_set))
        self.channel_uses += max(1, len(output.action.layers)) * max(1, output.action.repetitions)
        self.total_bandwidth += output.action.bandwidth
        self.total_power += output.action.power
        self.total_energy += output.action.energy
        self.total_latency += output.action.latency
        self.total_compute += output.action.compute
        self.total_repetitions += max(1, output.action.repetitions)
        self.decision_counts[output.decision.value] += 1
        self.ood_scores.append((output.risk_score, sample.is_unknown if ood_label is None else ood_label))

    def summarize(self, adaptation_harm_rate: float = 0.0, certified_accept_rate: float = 0.0) -> dict[str, float]:
        open_risk = mean(self.risks)
        num_samples = max(len(self.risks), 1)
        total_resource_cost = float(np.sum(self.resource_costs))
        return {
            "open_semantic_risk": open_risk,
            "semantic_outage": mean([float(e > 0.0) for e in self.task_errors]),
            "open_semantic_outage": mean(self.accepted_open_errors),
            "accuracy": 1.0 - mean(self.task_errors),
            "coverage": mean([float(x) for x in self.covered]),
            "prediction_set_size": mean([float(x) for x in self.prediction_set_sizes]),
            "semantic_goodput": self.correct_accepted / max(self.channel_uses, 1),
            "accepted_samples": float(self.accepted_samples),
            "correct_accepted_samples": float(self.correct_accepted),
            "total_bandwidth": self.total_bandwidth,
            "avg_bandwidth": self.total_bandwidth / num_samples,
            "bandwidth_per_accepted": self.total_bandwidth / self.accepted_samples if self.accepted_samples else 0.0,
            "bandwidth_per_correct_accepted": self.total_bandwidth / self.correct_accepted if self.correct_accepted else 0.0,
            "goodput_per_bandwidth": self.correct_accepted / max(self.total_bandwidth, 1e-9),
            "total_resource_cost": total_resource_cost,
            "avg_resource_cost": mean(self.resource_costs),
            "resource_cost_per_correct_accepted": total_resource_cost / self.correct_accepted if self.correct_accepted else 0.0,
            "goodput_per_resource_cost": self.correct_accepted / max(total_resource_cost, 1e-9),
            "total_power": self.total_power,
            "avg_power": self.total_power / num_samples,
            "risk_per_joule": open_risk / max(self.total_energy, 1e-9),
            "total_energy": self.total_energy,
            "avg_energy": self.total_energy / num_samples,
            "risk_latency_product": open_risk * (self.total_latency / num_samples),
            "total_latency": self.total_latency,
            "avg_latency": self.total_latency / num_samples,
            "total_compute": self.total_compute,
            "avg_compute": self.total_compute / num_samples,
            "total_repetitions": float(self.total_repetitions),
            "avg_repetitions": self.total_repetitions / num_samples,
            "adaptation_harm_rate": adaptation_harm_rate,
            "certified_accept_rate": certified_accept_rate,
            "auroc_ood": auroc(self.ood_scores),
            "fpr95": fpr_at_tpr(self.ood_scores, target_tpr=0.95),
            "open_set_f1": open_set_f1(self.ood_scores),
        }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def auroc(scores_and_labels: list[tuple[float, bool]]) -> float:
    positives = [(s, y) for s, y in scores_and_labels if y]
    negatives = [(s, y) for s, y in scores_and_labels if not y]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = 0
    for ps, _ in positives:
        for ns, _ in negatives:
            wins += float(ps > ns) + 0.5 * float(ps == ns)
            total += 1
    return wins / max(total, 1)


def fpr_at_tpr(scores_and_labels: list[tuple[float, bool]], target_tpr: float) -> float:
    if not scores_and_labels:
        return 1.0
    thresholds = sorted({s for s, _ in scores_and_labels}, reverse=True)
    best_fpr = 1.0
    for threshold in thresholds:
        tp = sum(1 for s, y in scores_and_labels if y and s >= threshold)
        fn = sum(1 for s, y in scores_and_labels if y and s < threshold)
        fp = sum(1 for s, y in scores_and_labels if not y and s >= threshold)
        tn = sum(1 for s, y in scores_and_labels if not y and s < threshold)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if tpr >= target_tpr:
            best_fpr = min(best_fpr, fpr)
    return best_fpr


def open_set_f1(scores_and_labels: list[tuple[float, bool]]) -> float:
    if not scores_and_labels:
        return 0.0
    threshold = float(np.median([s for s, _ in scores_and_labels]))
    tp = sum(1 for s, y in scores_and_labels if y and s >= threshold)
    fp = sum(1 for s, y in scores_and_labels if not y and s >= threshold)
    fn = sum(1 for s, y in scores_and_labels if y and s < threshold)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


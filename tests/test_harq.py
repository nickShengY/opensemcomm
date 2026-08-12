import numpy as np
import pytest

from opensemcom.channels import ChannelObservation
from opensemcom.harq import SemanticHARQ
from opensemcom.types import Decision, ReceiverOutput, ResourceAction, SemanticLayers


class DummyEncoder:
    def encode(self, layers: SemanticLayers, selected_layers: tuple[str, ...]):
        return layers.select(selected_layers)


class IdentityChannel:
    def transmit(self, symbols):
        return ChannelObservation(received=np.asarray(symbols, dtype=np.float64), state={"snr_db": 12.0})


class SequenceReceiver:
    def __init__(self, decisions: list[Decision]):
        self.decisions = list(decisions)
        self.actions: list[tuple[str, ...]] = []

    def receive(self, received, action, channel_state, task, domain=""):
        self.actions.append(action.layers)
        decision = self.decisions.pop(0)
        return ReceiverOutput(
            y_hat=0,
            probabilities=np.asarray([1.0], dtype=np.float64),
            prediction_set={0},
            risk_score=0.1,
            decision=decision,
            features={"confidence": 1.0},
            action=action,
        )


class TransitionReceiver(SequenceReceiver):
    def __init__(self):
        super().__init__([Decision.REFINE, Decision.REFINE, Decision.ACCEPT])
        self.risks = [0.80, 0.55, 0.20]

    def receive(self, received, action, channel_state, task, domain=""):
        output = super().receive(received, action, channel_state, task, domain)
        return ReceiverOutput(
            y_hat=output.y_hat,
            probabilities=output.probabilities,
            prediction_set=output.prediction_set,
            risk_score=self.risks.pop(0),
            decision=output.decision,
            features=output.features,
            action=action,
        )

    def thresholds_for_action(self, action):
        thresholds = {
            ("core",): (0.30, 0.70),
            ("core", "refinement"): (0.40, 0.65),
            ("core", "refinement", "evidence"): (0.25, 0.55),
        }
        return thresholds[action.layers]


def make_layers() -> SemanticLayers:
    return SemanticLayers(
        core=np.asarray([1.0, 0.0], dtype=np.float64),
        refinement=np.asarray([0.5], dtype=np.float64),
        evidence=np.asarray([0.25], dtype=np.float64),
        fallback=np.asarray([1.0, 0.0, 0.5, 0.25], dtype=np.float64),
        voi=np.asarray([1.0, 0.8, 0.6, 0.4], dtype=np.float64),
    )


def test_harq_progresses_layers_until_accept():
    receiver = SequenceReceiver([Decision.REFINE, Decision.REFINE, Decision.ACCEPT])
    harq = SemanticHARQ(DummyEncoder(), IdentityChannel(), receiver, max_refinements=3)

    output = harq.run(make_layers(), ResourceAction(layers=("core",), repetitions=1), task="classification")

    assert receiver.actions == [
        ("core",),
        ("core", "refinement"),
        ("core", "refinement", "evidence"),
    ]
    assert output.decision == Decision.ACCEPT
    assert output.action.layers == ("core", "refinement", "evidence")
    assert output.features["harq_refinement_rounds"] == 2.0
    assert output.features["harq_transmissions"] == 3.0


def test_harq_rejects_when_budget_is_exhausted():
    receiver = SequenceReceiver([Decision.REFINE, Decision.REFINE, Decision.REFINE])
    harq = SemanticHARQ(DummyEncoder(), IdentityChannel(), receiver, max_refinements=2)

    output = harq.run(make_layers(), ResourceAction(layers=("core",), repetitions=1), task="classification")

    assert receiver.actions == [
        ("core",),
        ("core", "refinement"),
        ("core", "refinement", "evidence"),
    ]
    assert output.decision == Decision.REJECT_OPEN
    assert output.features["harq_hit_max_refinements"] == 1.0
    assert output.features["harq_max_refinements"] == 2.0


def test_harq_zero_refinement_budget_rejects_a_refinement_request():
    receiver = SequenceReceiver([Decision.REFINE])
    harq = SemanticHARQ(DummyEncoder(), IdentityChannel(), receiver, max_refinements=0)

    output = harq.run(make_layers(), ResourceAction(layers=("core",), repetitions=1), task="classification")

    assert receiver.actions == [("core",)]
    assert output.decision == Decision.REJECT_OPEN
    assert output.features["harq_refinement_rounds"] == 0.0
    assert output.features["harq_hit_max_refinements"] == 1.0

def test_harq_logs_stage_relative_score_transitions_without_affecting_actions():
    receiver = TransitionReceiver()
    harq = SemanticHARQ(DummyEncoder(), IdentityChannel(), receiver, max_refinements=3)

    output = harq.run(make_layers(), ResourceAction(layers=("core",), repetitions=1), task="classification")

    assert output.decision == Decision.ACCEPT
    assert len(output.refinement_transitions) == 2
    first, second = output.refinement_transitions
    assert first["current_stage"] == "core"
    assert first["next_stage"] == "core+refinement"
    assert first["current_stage_index"] == 0
    assert first["next_stage_index"] == 1
    assert first["transition_type"] == "semantic_expansion_0_to_1"
    assert first["raw_score_change"] == 0.25
    assert first["current_acceptance_margin"] == 0.50
    assert first["next_acceptance_margin"] == pytest.approx(0.15)
    assert first["margin_change"] == pytest.approx(0.35)
    assert second["next_stage"] == "core+refinement+evidence"

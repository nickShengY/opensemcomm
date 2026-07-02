import numpy as np

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

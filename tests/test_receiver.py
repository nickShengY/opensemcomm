from types import SimpleNamespace

from opensemcom.config import CalibrationConfig, DetectorWeights
from opensemcom.receiver import SelectiveSemanticReceiver
from opensemcom.types import Decision


def make_receiver(vim_weight: float) -> SelectiveSemanticReceiver:
    receiver = SelectiveSemanticReceiver.__new__(SelectiveSemanticReceiver)
    receiver.detector = SimpleNamespace(weights=DetectorWeights(vim=vim_weight))
    receiver.config = CalibrationConfig()
    receiver.q_accept = 0.30
    receiver.q_refine = 0.40
    return receiver


def test_disabled_vim_does_not_block_acceptance():
    receiver = make_receiver(vim_weight=0.0)

    decision = receiver._decision(0.20, {0}, {"confidence": 1.0, "vim": 1.0})

    assert decision == Decision.ACCEPT


def test_enabled_vim_still_requests_refinement():
    receiver = make_receiver(vim_weight=0.10)

    decision = receiver._decision(0.20, {0}, {"confidence": 1.0, "vim": 1.0})

    assert decision == Decision.REFINE
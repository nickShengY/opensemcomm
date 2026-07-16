from __future__ import annotations

import numpy as np

from opensemcom.cli.communication_control_suite import (
    TrainedReceiver,
    sha256_file,
)


def test_trained_receiver_checkpoint_round_trip(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(36, 8)).astype(np.float32)
    y = np.asarray([0, 1, 2] * 12, dtype=np.int64)
    is_open = np.asarray(([0] * 24) + ([1] * 12), dtype=np.int64)
    receiver = TrainedReceiver(
        input_dim=8,
        n_classes=3,
        has_open_class=False,
        seed=5,
        hidden_dim=16,
        epochs=2,
        device="cpu",
    )
    receiver.fit(x, y, is_open)
    expected = receiver.score(x)

    checkpoint = tmp_path / "receiver.pt"
    receiver.save_checkpoint(checkpoint)
    restored = TrainedReceiver.load_checkpoint(checkpoint, device="cpu")
    actual = restored.score(x)

    assert checkpoint.stat().st_size > 0
    assert len(sha256_file(checkpoint)) == 64
    np.testing.assert_array_equal(actual.pred, expected.pred)
    np.testing.assert_allclose(actual.risk, expected.risk, rtol=0.0, atol=1e-7)


def test_checkpoint_rejects_unknown_format(tmp_path):
    import torch

    checkpoint = tmp_path / "invalid.pt"
    torch.save({"format_version": 999, "model_type": "other"}, checkpoint)

    try:
        TrainedReceiver.load_checkpoint(checkpoint)
    except ValueError as exc:
        assert "Unsupported receiver checkpoint" in str(exc)
    else:
        raise AssertionError("Invalid checkpoint format was accepted")

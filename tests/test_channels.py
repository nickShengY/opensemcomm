import importlib.util

import numpy as np
import pytest

from opensemcom.channels import WirelessChannel, build_channel
from opensemcom.config import ChannelConfig
from opensemcom.types import ChannelBackend, ChannelKind


def test_awgn_channel_preserves_shape():
    rng = np.random.default_rng(0)
    channel = WirelessChannel(ChannelConfig(kind=ChannelKind.AWGN, snr_db=30.0), rng)
    x = np.ones(8)
    obs = channel.transmit(x)
    assert obs.received.shape == x.shape
    assert "effective_snr_db" in obs.state


def test_interference_channel_reports_shift_terms():
    rng = np.random.default_rng(1)
    channel = WirelessChannel(ChannelConfig(kind=ChannelKind.INTERFERENCE, interference_power=0.2), rng)
    obs = channel.transmit(np.ones(4))
    assert obs.state["interference_power"] == 0.2
    assert obs.received.shape == (4,)


def test_build_channel_defaults_to_numpy_backend():
    rng = np.random.default_rng(2)
    channel = build_channel(ChannelConfig(), rng)
    assert isinstance(channel, WirelessChannel)


SIONNA_AVAILABLE = importlib.util.find_spec("sionna") is not None


@pytest.mark.skipif(SIONNA_AVAILABLE, reason="Sionna is installed in this environment.")
def test_sionna_backend_requires_optional_dependency():
    rng = np.random.default_rng(3)
    with pytest.raises(RuntimeError, match="Sionna backend requested"):
        build_channel(ChannelConfig(backend=ChannelBackend.SIONNA), rng)


@pytest.mark.skipif(not SIONNA_AVAILABLE, reason="Sionna is not installed.")
def test_sionna_awgn_coded_link_round_trip():
    channel = build_channel(
        ChannelConfig(
            backend=ChannelBackend.SIONNA,
            kind=ChannelKind.AWGN,
            snr_db=40.0,
            sionna_seed=7,
        ),
        np.random.default_rng(3),
    )
    symbols = np.asarray([-0.8, -0.2, 0.0, 0.3, 0.7], dtype=np.float64)

    observation = channel.transmit(symbols)

    assert observation.received.shape == symbols.shape
    assert np.allclose(observation.received, symbols, atol=1.0 / 255.0 + 1e-12)
    assert observation.state["coding_rate"] == 0.5
    assert observation.state["bits_per_symbol"] == 2.0


@pytest.mark.skipif(not SIONNA_AVAILABLE, reason="Sionna is not installed.")
def test_sionna_repetitions_use_chase_combining():
    channel = build_channel(
        ChannelConfig(
            backend=ChannelBackend.SIONNA,
            kind=ChannelKind.AWGN,
            snr_db=6.0,
            sionna_seed=13,
        ),
        np.random.default_rng(4),
    )

    observation = channel.transmit_repeated(np.linspace(-0.7, 0.7, 16), repetitions=2)

    assert observation.received.shape == (16,)
    assert observation.state["repetitions"] == 2.0
    assert observation.state["harq_chase_combining"] == 1.0
@pytest.mark.skipif(not SIONNA_AVAILABLE, reason="Sionna is not installed.")
def test_sionna_interference_channel_reports_interference_power():
    channel = build_channel(
        ChannelConfig(
            backend=ChannelBackend.SIONNA,
            kind=ChannelKind.INTERFERENCE,
            snr_db=8.0,
            interference_power=0.25,
            sionna_seed=21,
        ),
        np.random.default_rng(5),
    )

    observation = channel.transmit(np.linspace(-0.5, 0.5, 12))

    assert observation.received.shape == (12,)
    assert observation.state["interference_power"] == 0.25

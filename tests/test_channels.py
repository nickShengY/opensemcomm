import numpy as np

from opensemcom.channels import WirelessChannel
from opensemcom.config import ChannelConfig
from opensemcom.types import ChannelKind


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

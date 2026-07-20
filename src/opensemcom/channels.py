"""Wireless channel models for OpenSemCom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from opensemcom.config import ChannelConfig
from opensemcom.types import Array, ChannelBackend, ChannelKind

if TYPE_CHECKING:
    import torch


def snr_to_noise_std(signal: Array, snr_db: float) -> float:
    power = float(np.mean(np.square(signal))) if signal.size else 1.0
    snr_linear = 10.0 ** (snr_db / 10.0)
    return float(np.sqrt(max(power, 1e-9) / max(snr_linear, 1e-9)))


@dataclass
class ChannelObservation:
    received: Array
    state: dict[str, float]


class WirelessChannel:
    """Composable wireless channel with common 6G stressors."""

    def __init__(self, config: ChannelConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def transmit(self, symbols: Array) -> ChannelObservation:
        x = np.asarray(symbols, dtype=np.float64)
        h_gain = 1.0
        y = x.copy()

        if self.config.kind == ChannelKind.RAYLEIGH:
            h_gain = float(self.rng.rayleigh(self.config.fading_scale))
            y = h_gain * y
        elif self.config.kind == ChannelKind.RICIAN:
            los = 1.0
            scatter = float(self.rng.normal(0.0, self.config.fading_scale))
            h_gain = abs(los + scatter)
            y = h_gain * y
        elif self.config.kind == ChannelKind.MIMO:
            y, h_gain = self._mimo_transmit(y)
        elif self.config.kind == ChannelKind.BLOCKAGE:
            blocked = self.rng.random() < self.config.blockage_probability
            h_gain = 0.15 if blocked else 1.0
            y = h_gain * y
        elif self.config.kind == ChannelKind.DOPPLER:
            phase = np.linspace(0.0, self.config.doppler_hz * 0.01, y.size)
            y = y * np.cos(phase)
            h_gain = float(np.mean(np.abs(np.cos(phase)))) if y.size else 1.0
        elif self.config.kind == ChannelKind.INTERFERENCE:
            interference = self.rng.normal(0.0, np.sqrt(self.config.interference_power), size=y.shape)
            y = y + interference

        if self.config.burst_probability > 0.0 and self.rng.random() < self.config.burst_probability:
            burst = self.rng.normal(0.0, 2.0, size=y.shape)
            y = y + burst

        noise_std = snr_to_noise_std(y, self.config.snr_db)
        y = y + self.rng.normal(0.0, noise_std, size=y.shape)

        if self.config.csi_error > 0.0:
            h_gain = max(0.0, h_gain + float(self.rng.normal(0.0, self.config.csi_error)))

        effective_snr = 10.0 * np.log10((h_gain**2 + 1e-9) / (noise_std**2 + 1e-9))
        state = {
            "snr_db": float(self.config.snr_db),
            "effective_snr_db": float(effective_snr),
            "gain": float(h_gain),
            "interference_power": float(self.config.interference_power),
            "blockage_probability": float(self.config.blockage_probability),
            "doppler_hz": float(self.config.doppler_hz),
            "burst_probability": float(self.config.burst_probability),
            "csi_error": float(self.config.csi_error),
        }
        return ChannelObservation(received=y, state=state)

    def _mimo_transmit(self, symbols: Array) -> tuple[Array, float]:
        if symbols.size == 0:
            return symbols, 1.0
        rx = max(1, self.config.mimo_rx)
        h = self.rng.normal(0.0, self.config.fading_scale, size=(rx, symbols.size))
        branches = h * symbols.reshape(1, -1)
        combined = branches.mean(axis=0)
        gain = float(np.mean(np.abs(h)))
        return combined, gain


class SionnaChannel(WirelessChannel):
    """Minimal Sionna-backed channel wrapper preserving the OpenSemCom contract."""

    def __init__(self, config: ChannelConfig, rng: np.random.Generator):
        super().__init__(config, rng)
        self._torch, awgn_cls = _load_sionna_awgn()
        self._awgn = awgn_cls(precision="double", device=config.sionna_device)

    def transmit(self, symbols: Array) -> ChannelObservation:
        if self.config.kind != ChannelKind.AWGN:
            raise NotImplementedError(
                "The minimal Sionna backend currently supports only AWGN. "
                "Use backend='numpy' for Rayleigh, interference, blockage, Doppler, or MIMO."
            )

        x = np.asarray(symbols, dtype=np.float64)
        noise_std = snr_to_noise_std(x, self.config.snr_db)
        noise_variance = float(2.0 * (noise_std**2))

        x_tensor = self._torch.from_numpy(x).to(device=self.config.sionna_device, dtype=self._torch.float64)
        x_complex = self._torch.complex(x_tensor, self._torch.zeros_like(x_tensor))
        y_complex = self._awgn(x_complex, noise_variance)
        y = y_complex.real.detach().cpu().numpy().astype(np.float64, copy=False)

        effective_snr = 10.0 * np.log10(1.0 / max(noise_std**2, 1e-9))
        state = {
            "snr_db": float(self.config.snr_db),
            "effective_snr_db": float(effective_snr),
            "gain": 1.0,
            "interference_power": 0.0,
            "blockage_probability": 0.0,
            "doppler_hz": 0.0,
            "burst_probability": 0.0,
            "csi_error": 0.0,
        }
        return ChannelObservation(received=y, state=state)


def build_channel(config: ChannelConfig, rng: np.random.Generator) -> WirelessChannel:
    """Construct the configured channel backend."""

    if config.backend == ChannelBackend.SIONNA:
        return SionnaChannel(config, rng)
    return WirelessChannel(config, rng)


def shifted_channel(base: ChannelConfig, kind: ChannelKind, snr_delta: float = 0.0) -> ChannelConfig:
    """Create a channel-open variant without mutating the base config."""

    return ChannelConfig(
        backend=base.backend,
        kind=kind,
        snr_db=base.snr_db + snr_delta,
        fading_scale=base.fading_scale,
        interference_power=max(base.interference_power, 0.15 if kind == ChannelKind.INTERFERENCE else base.interference_power),
        blockage_probability=max(base.blockage_probability, 0.25 if kind == ChannelKind.BLOCKAGE else base.blockage_probability),
        doppler_hz=max(base.doppler_hz, 80.0 if kind == ChannelKind.DOPPLER else base.doppler_hz),
        burst_probability=max(base.burst_probability, 0.10 if kind == ChannelKind.BURST else base.burst_probability),
        csi_error=max(base.csi_error, 0.05),
        mimo_rx=base.mimo_rx,
        sionna_device=base.sionna_device,
    )


def _load_sionna_awgn() -> tuple[type["torch"], type]:
    try:
        import torch
        from sionna.phy.channel import AWGN
    except ImportError as exc:
        raise RuntimeError(
            "Sionna backend requested, but Sionna/PyTorch are not available. "
            "Install a Sionna-compatible environment first."
        ) from exc
    return torch, AWGN

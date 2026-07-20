"""Sionna physical-layer link for OpenSemCom semantic payloads.

This module is deliberately the only place where the runtime calls Sionna
PHY APIs. It converts an OpenSemCom floating-point payload to bits, transports
it through a coded digital link, and reconstructs the floating-point payload
expected by the semantic receiver.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from opensemcom.config import ChannelConfig
from opensemcom.types import Array, ChannelKind

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PhyObservation:
    """Decoded semantic payload plus physical-link metadata."""

    received: Array
    state: dict[str, float]


class SionnaDigitalLink:
    """Coded QAM link with 5G LDPC decoding and optional receive diversity."""

    _SUPPORTED_KINDS = {
        ChannelKind.AWGN,
        ChannelKind.RAYLEIGH,
        ChannelKind.RICIAN,
        ChannelKind.MIMO,
    }

    def __init__(self, config: ChannelConfig):
        self.config = config
        if config.kind not in self._SUPPORTED_KINDS:
            supported = ", ".join(kind.value for kind in sorted(self._SUPPORTED_KINDS, key=lambda item: item.value))
            raise NotImplementedError(
                f"The coded Sionna link supports {supported}. "
                "OFDM/time-selective channels and interference are separate link models."
            )
        self._validate_config()
        self._torch, self._sionna = _load_sionna_components()
        if config.sionna_seed is not None:
            self._sionna.config.seed = int(config.sionna_seed)

        precision = "double"
        device = config.sionna_device
        self._encoder = self._sionna.LDPC5GEncoder(
            config.sionna_ldpc_info_bits,
            config.sionna_ldpc_codeword_bits,
            num_bits_per_symbol=config.sionna_bits_per_symbol,
            precision=precision,
            device=device,
        )
        self._decoder = self._sionna.LDPC5GDecoder(
            self._encoder,
            hard_out=True,
            return_infobits=True,
            precision=precision,
            device=device,
        )
        self._mapper = self._sionna.Mapper(
            "qam",
            config.sionna_bits_per_symbol,
            precision=precision,
            device=device,
        )
        self._demapper = self._sionna.Demapper(
            "app",
            "qam",
            config.sionna_bits_per_symbol,
            precision=precision,
            device=device,
        )
        self._awgn = self._sionna.AWGN(precision=precision, device=device)
        self._flat_fading = self._sionna.GenerateFlatFadingChannel(
            num_tx_ant=1,
            num_rx_ant=max(1, config.mimo_rx) if config.kind == ChannelKind.MIMO else 1,
            precision=precision,
            device=device,
        )

    def transmit(self, symbols: Array, repetitions: int = 1) -> PhyObservation:
        """Send one semantic payload, Chase-combining identical retransmissions."""

        x = np.asarray(symbols, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return PhyObservation(received=x, state=self._state(gain=1.0, repetitions=1))

        payload_bits = _quantize_to_bits(x, self.config.sionna_quantization_bits)
        padded_bits, payload_bit_count = self._pad_blocks(payload_bits)
        bits = self._torch.from_numpy(padded_bits).to(
            device=self.config.sionna_device,
            dtype=self._torch.float64,
        )
        codewords = self._encoder(bits)
        transmitted = self._mapper(codewords)
        noise_variance = float(10.0 ** (-self.config.snr_db / 10.0))
        llrs, gain = self._transmit_codewords(transmitted, noise_variance, repetitions)
        decoded_bits = self._decoder(llrs).detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1)
        recovered = _bits_to_values(
            decoded_bits[:payload_bit_count],
            self.config.sionna_quantization_bits,
        ).reshape(x.shape)
        return PhyObservation(received=recovered, state=self._state(gain=gain, repetitions=repetitions))

    def _transmit_codewords(self, transmitted, noise_variance: float, repetitions: int):
        repetitions = max(1, int(repetitions))
        combined_llrs = None
        gain_values: list[float] = []
        for _ in range(repetitions):
            equalized, effective_noise, gain = self._propagate(transmitted, noise_variance)
            llrs = self._demapper(equalized, effective_noise)
            combined_llrs = llrs if combined_llrs is None else combined_llrs + llrs
            gain_values.append(gain)
        return combined_llrs, float(np.mean(gain_values))

    def _propagate(self, transmitted, noise_variance: float):
        if self.config.kind == ChannelKind.AWGN:
            return self._awgn(transmitted, noise_variance), noise_variance, 1.0

        batch_size = transmitted.shape[0]
        h = self._flat_fading(batch_size)
        if self.config.kind == ChannelKind.RICIAN:
            k_factor = self.config.sionna_rician_k_factor
            h = np.sqrt(k_factor / (k_factor + 1.0)) + h * np.sqrt(1.0 / (k_factor + 1.0))

        # Treat ChannelKind.MIMO as one transmitted stream with receive diversity.
        h = h * self.config.fading_scale
        h = h[..., 0]
        y = self._awgn(h.unsqueeze(-1) * transmitted.unsqueeze(1), noise_variance)
        channel_power = self._torch.sum(self._torch.abs(h) ** 2, dim=1)
        equalized = self._torch.sum(self._torch.conj(h).unsqueeze(-1) * y, dim=1) / channel_power
        effective_noise = noise_variance / channel_power
        gain = float(self._torch.mean(self._torch.sqrt(channel_power)).detach().cpu())
        return equalized, effective_noise, gain

    def _pad_blocks(self, bits: Array) -> tuple[Array, int]:
        k = self.config.sionna_ldpc_info_bits
        blocks = max(1, int(np.ceil(bits.size / k)))
        padded = np.zeros((blocks, k), dtype=np.uint8)
        padded.reshape(-1)[: bits.size] = bits
        return padded, int(bits.size)

    def _state(self, gain: float, repetitions: int) -> dict[str, float]:
        effective_snr = self.config.snr_db + 20.0 * np.log10(max(gain, 1e-9)) + 10.0 * np.log10(max(1, int(repetitions)))
        return {
            "snr_db": float(self.config.snr_db),
            "effective_snr_db": float(effective_snr),
            "gain": float(gain),
            "interference_power": 0.0,
            "blockage_probability": 0.0,
            "doppler_hz": 0.0,
            "burst_probability": 0.0,
            "csi_error": 0.0,
            "repetitions": float(max(1, int(repetitions))),
            "coding_rate": float(self.config.sionna_ldpc_info_bits / self.config.sionna_ldpc_codeword_bits),
            "bits_per_symbol": float(self.config.sionna_bits_per_symbol),
            "quantization_bits": float(self.config.sionna_quantization_bits),
            "harq_chase_combining": float(max(1, int(repetitions)) > 1),
            "mimo_rx": float(max(1, self.config.mimo_rx) if self.config.kind == ChannelKind.MIMO else 1),
        }

    def _validate_config(self) -> None:
        if self.config.sionna_bits_per_symbol not in (2, 4, 6, 8):
            raise ValueError("sionna_bits_per_symbol must be one of 2, 4, 6, or 8 for square QAM.")
        if not 1 <= self.config.sionna_quantization_bits <= 16:
            raise ValueError("sionna_quantization_bits must be between 1 and 16.")
        if self.config.sionna_ldpc_info_bits <= 0 or self.config.sionna_ldpc_codeword_bits <= self.config.sionna_ldpc_info_bits:
            raise ValueError("Sionna LDPC requires 0 < sionna_ldpc_info_bits < sionna_ldpc_codeword_bits.")
        if self.config.sionna_ldpc_codeword_bits % self.config.sionna_bits_per_symbol:
            raise ValueError("sionna_ldpc_codeword_bits must be divisible by sionna_bits_per_symbol.")
        if self.config.fading_scale <= 0.0:
            raise ValueError("fading_scale must be positive for the Sionna fading link.")
        if self.config.sionna_rician_k_factor < 0.0:
            raise ValueError("sionna_rician_k_factor must be non-negative.")


def _quantize_to_bits(values: Array, quantization_bits: int) -> Array:
    """Uniformly quantize normalized semantic values in [-1, 1] to bits."""

    levels = (1 << quantization_bits) - 1
    integers = np.rint(np.clip((values + 1.0) * 0.5, 0.0, 1.0) * levels).astype(np.uint32)
    shifts = np.arange(quantization_bits - 1, -1, -1, dtype=np.uint32)
    return ((integers[:, None] >> shifts) & 1).astype(np.uint8, copy=False).reshape(-1)


def _bits_to_values(bits: Array, quantization_bits: int) -> Array:
    if bits.size % quantization_bits:
        raise ValueError("Decoded bit count is not aligned with the semantic quantizer.")
    levels = (1 << quantization_bits) - 1
    weights = 1 << np.arange(quantization_bits - 1, -1, -1, dtype=np.uint32)
    integers = (bits.reshape(-1, quantization_bits).astype(np.uint32) * weights).sum(axis=1)
    return integers.astype(np.float64) * (2.0 / levels) - 1.0


def _load_sionna_components():
    try:
        import sionna.phy as sionna_phy
        import torch
        from sionna.phy.channel import AWGN, GenerateFlatFadingChannel
        from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
        from sionna.phy.mapping import Demapper, Mapper
    except ImportError as exc:
        raise RuntimeError(
            "Sionna backend requested, but Sionna/PyTorch are not available. "
            "Install the project with `.[sionna]` in a Sionna-compatible environment."
        ) from exc

    return torch, SimpleNamespace(
        config=sionna_phy.config,
        AWGN=AWGN,
        GenerateFlatFadingChannel=GenerateFlatFadingChannel,
        LDPC5GEncoder=LDPC5GEncoder,
        LDPC5GDecoder=LDPC5GDecoder,
        Mapper=Mapper,
        Demapper=Demapper,
    )

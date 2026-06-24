"""Reliability-card semantic codec library."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityCard:
    codec_id: str
    domains: tuple[str, ...]
    tasks: tuple[str, ...]
    channels: tuple[str, ...]
    risk_bias: float
    open_semantic_outage: float
    compute_cost: float
    latency_cost: float
    energy_cost: float
    calibration_threshold: float


class CodecLibrary:
    def __init__(self, cards: list[ReliabilityCard] | None = None):
        self.cards = cards or [
            ReliabilityCard(
                codec_id="default",
                domains=("urban-day", "urban-night", "rain", "sensor-shift"),
                tasks=("classification", "retrieval", "hazard"),
                channels=("awgn", "rayleigh", "rician", "mimo", "interference", "blockage", "doppler", "burst"),
                risk_bias=0.0,
                open_semantic_outage=0.10,
                compute_cost=1.0,
                latency_cost=1.0,
                energy_cost=1.0,
                calibration_threshold=0.90,
            ),
            ReliabilityCard(
                codec_id="robust",
                domains=("urban-day", "urban-night", "rain", "sensor-shift", "unknown-domain"),
                tasks=("classification", "retrieval", "hazard"),
                channels=("rayleigh", "mimo", "interference", "blockage", "doppler", "burst"),
                risk_bias=-0.05,
                open_semantic_outage=0.07,
                compute_cost=1.4,
                latency_cost=1.3,
                energy_cost=1.3,
                calibration_threshold=0.93,
            ),
        ]

    def route(self, domain: str, task: str, channel: str) -> ReliabilityCard:
        def score(card: ReliabilityCard) -> tuple[int, float]:
            support = int(domain in card.domains) + int(task in card.tasks) + int(channel in card.channels)
            return support, -card.open_semantic_outage

        return max(self.cards, key=score)

    def ids(self) -> tuple[str, ...]:
        return tuple(card.codec_id for card in self.cards)

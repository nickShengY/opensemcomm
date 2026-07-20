"""Configuration objects for the OpenSemCom prototype."""

from __future__ import annotations

from dataclasses import dataclass, field

from opensemcom.types import ChannelBackend, ChannelKind, ResourceBudget


@dataclass(frozen=True)
class RiskWeights:
    beta_unknown: float = 3.0
    beta_task: float = 2.0
    beta_adapt: float = 2.0
    beta_calibration: float = 1.0
    beta_resource: float = 0.1


@dataclass(frozen=True)
class ResourceWeights:
    power: float = 0.25
    bandwidth: float = 0.20
    latency: float = 0.20
    energy: float = 0.20
    compute: float = 0.15


@dataclass(frozen=True)
class DetectorWeights:
    prediction: float = 0.15
    prototype: float = 0.10
    energy: float = 0.10
    selective: float = 0.0
    unknown: float = 0.0
    openmax: float = 0.0
    vim: float = 0.0
    reconstruction: float = 0.10
    channel: float = 0.15
    task: float = 0.10
    domain: float = 0.20
    adaptation: float = 0.05
    mahalanobis: float = 0.15


@dataclass(frozen=True)
class CalibrationConfig:
    delta: float = 0.1
    epsilon_task: float = 0.0
    accept_quantile: float = 0.45
    refine_quantile: float = 0.65
    max_refinements: int = 3
    min_accept_confidence: float = 0.0
    mixed_open: bool = False
    open_split: str = "open-calibration"
    open_fraction: float = 0.5
    target_open_outage: float = 0.05


@dataclass(frozen=True)
class AdaptationConfig:
    alpha: float = 0.05
    horizon: int = 1000
    min_buffer: int = 16
    kappa: float = 0.01
    pseudo_label_threshold: float = 0.75
    update_strength: float = 0.05


@dataclass(frozen=True)
class AblationConfig:
    use_detector: bool = True
    use_conformal: bool = True
    use_harq: bool = True
    use_adaptation: bool = True
    use_scheduler: bool = True


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 32
    latent_dim: int = 16
    num_known_classes: int = 6
    num_unknown_classes: int = 2
    projection: str = "random"
    classifier: str = "prototype"
    channel_augmentations: int = 1
    torch_epochs: int = 80
    torch_hidden_dim: int = 256
    torch_lr: float = 0.001
    torch_device: str = "auto"
    tasks: tuple[str, ...] = ("classification", "retrieval")
    train_tasks: tuple[str, ...] = ("classification",)
    train_domains: tuple[str, ...] = ("urban-day",)


@dataclass(frozen=True)
class ChannelConfig:
    backend: ChannelBackend = ChannelBackend.NUMPY
    kind: ChannelKind = ChannelKind.AWGN
    snr_db: float = 12.0
    fading_scale: float = 1.0
    interference_power: float = 0.0
    blockage_probability: float = 0.0
    doppler_hz: float = 0.0
    burst_probability: float = 0.0
    csi_error: float = 0.0
    mimo_rx: int = 2
    sionna_device: str = "cpu"
    sionna_seed: int | None = None
    sionna_bits_per_symbol: int = 2
    sionna_quantization_bits: int = 8
    sionna_ldpc_info_bits: int = 256
    sionna_ldpc_codeword_bits: int = 512
    sionna_rician_k_factor: float = 5.0


@dataclass(frozen=True)
class OpenSemComConfig:
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    detector_weights: DetectorWeights = field(default_factory=DetectorWeights)
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    resource_weights: ResourceWeights = field(default_factory=ResourceWeights)
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    ablation: AblationConfig = field(default_factory=AblationConfig)


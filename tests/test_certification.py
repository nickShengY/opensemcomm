from math import comb

import numpy as np

from opensemcom.adaptation import SafeAdapter
from opensemcom.certification import (
    ChannelSupportProfile,
    SelectiveCalibrationItem,
    certify_fixed_policy,
    clopper_pearson_upper,
    minimum_zero_error_accepts,
    select_accept_threshold,
)
from opensemcom.config import AdaptationConfig, ModelConfig
from opensemcom.semantic import PrototypeSemanticDecoder


def test_exact_binomial_bound_has_expected_five_percent_boundary():
    assert minimum_zero_error_accepts(0.05, 0.05) == 59
    assert minimum_zero_error_accepts(0.05, 0.05 / 8) == 99
    assert clopper_pearson_upper(0, 58, 0.05) > 0.05
    assert clopper_pearson_upper(0, 59, 0.05) <= 0.05

    upper = clopper_pearson_upper(3, 100, 0.05)
    cdf = sum(
        comb(100, index)
        * upper**index
        * (1.0 - upper) ** (100 - index)
        for index in range(4)
    )
    assert abs(cdf - 0.05) < 1e-10


def test_composed_policy_certificate_counts_only_accepted_outcomes():
    outcomes = [(True, False)] * 80 + [(False, True)] * 20

    certificate = certify_fixed_policy(
        outcomes,
        target_outage=0.05,
        alpha=0.05,
        minimum_accepts=20,
    )

    assert certificate.valid
    assert certificate.accepted_samples == 80
    assert certificate.unsafe_accepted == 0
    assert certificate.upper_bound < 0.05
    assert certificate.method == "clopper-pearson-composed-policy"


def test_one_unsafe_accept_can_invalidate_a_small_five_percent_certificate():
    outcomes = [(True, False)] * 79 + [(True, True)]

    certificate = certify_fixed_policy(
        outcomes,
        target_outage=0.05,
        alpha=0.05,
        minimum_accepts=20,
    )

    assert not certificate.valid
    assert certificate.upper_bound > 0.05


def test_threshold_selection_is_policy_selection_not_a_certificate():
    items = [
        SelectiveCalibrationItem(risk_score=index / 100.0, unsafe=index >= 30, eligible=True)
        for index in range(40)
    ]

    threshold = select_accept_threshold(
        items,
        target_outage=0.10,
        minimum_accepts=20,
        safety_factor=0.5,
    )

    assert threshold == 0.30


def test_channel_support_guard_rejects_an_unseen_channel_family():
    profile = ChannelSupportProfile(relative_margin=0.05)
    profile.fit(
        [
            {
                "channel_kind_code": 0.0,
                "snr_db": 12.0,
                "gain": 1.0,
            }
            for _ in range(32)
        ]
    )

    supported, _ = profile.evaluate(
        {"channel_kind_code": 0.0, "snr_db": 12.1, "gain": 1.0}
    )
    shifted, violation = profile.evaluate(
        {"channel_kind_code": 1.0, "snr_db": 12.1, "gain": 1.0}
    )

    assert supported
    assert not shifted
    assert violation > 0.0

    missing_diagnostics, _ = profile.evaluate(
        {"channel_kind_code": 0.0, "snr_db": 12.1}
    )
    assert not missing_diagnostics


def test_adaptation_requires_an_independent_verified_validation_split():
    decoder = PrototypeSemanticDecoder(
        ModelConfig(input_dim=4, latent_dim=4, num_known_classes=2),
        np.random.default_rng(4),
    )
    adapter = SafeAdapter(
        decoder,
        AdaptationConfig(min_buffer=2),
    )
    proposal = [(np.ones(4), 0), (np.zeros(4), 1)]

    result = adapter.propose_and_gate(proposal)

    assert not result.accepted
    assert "independent verified validation" in result.reason
    assert result.alpha_t == 0.025

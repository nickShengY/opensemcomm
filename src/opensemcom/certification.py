"""Finite-sample reliability certificates and empirical channel support guards.

The certificate in this module is intentionally narrower than a generic
"confidence score."  It bounds the unsafe-acceptance rate of one fixed
selective policy on an independent certificate split.  Model fitting,
conformal calibration, and threshold selection must all happen before that
split is inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, lgamma, log
from typing import Iterable

import numpy as np

from opensemcom.types import ReliabilityCertificate


CERTIFICATE_ASSUMPTIONS = (
    "certificate examples and channel realizations are i.i.d. deployment draws",
    "the certified policy is fixed before the certificate split is inspected",
    "unsafe acceptance is a bounded binary loss",
    "the decoder, thresholds, and support guard remain unchanged after certification",
)


def minimum_zero_error_accepts(target_outage: float, alpha: float) -> int:
    """Minimum zero-error accepted examples needed for a CP certificate."""

    if not 0.0 < target_outage < 1.0:
        raise ValueError("target_outage must lie strictly between zero and one")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return int(ceil(log(alpha) / log(1.0 - target_outage)))


@dataclass(frozen=True)
class SelectiveCalibrationItem:
    """One independent example used to select or certify an accept threshold."""

    risk_score: float
    unsafe: bool
    eligible: bool


def select_accept_threshold(
    items: Iterable[SelectiveCalibrationItem],
    target_outage: float,
    minimum_accepts: int,
    safety_factor: float = 0.5,
) -> float | None:
    """Select the largest empirical-safe threshold on a non-certificate split.

    This function performs policy selection only.  Its output is not a
    certificate and must be evaluated once on an untouched certificate split.
    """

    eligible = sorted(
        (item for item in items if item.eligible),
        key=lambda item: item.risk_score,
    )
    if not eligible:
        return None
    target = float(np.clip(target_outage * safety_factor, 0.0, 1.0))
    minimum_accepts = max(1, int(minimum_accepts))
    unsafe = 0
    selected: float | None = None
    for index, item in enumerate(eligible, start=1):
        unsafe += int(item.unsafe)
        if index >= minimum_accepts and unsafe / index <= target:
            selected = float(item.risk_score)
    return selected


def certify_accept_threshold(
    items: Iterable[SelectiveCalibrationItem],
    threshold: float | None,
    target_outage: float,
    alpha: float,
    minimum_accepts: int,
) -> ReliabilityCertificate:
    """Certify one fixed threshold with an exact one-sided binomial bound."""

    confidence = 1.0 - float(alpha)
    if threshold is None:
        return ReliabilityCertificate.unavailable(
            target_outage,
            confidence,
            "threshold selection found no eligible operating point",
        )
    items = list(items)
    accepted = [
        item
        for item in items
        if item.eligible and item.risk_score <= threshold
    ]
    unsafe = sum(int(item.unsafe) for item in accepted)
    minimum_accepts = max(1, int(minimum_accepts))
    if len(accepted) < minimum_accepts:
        return ReliabilityCertificate(
            valid=False,
            target_outage=float(target_outage),
            upper_bound=1.0,
            confidence=confidence,
            calibration_samples=len(items),
            accepted_samples=len(accepted),
            unsafe_accepted=unsafe,
            threshold=float(threshold),
            reason=(
                f"only {len(accepted)} certificate examples were accepted; "
                f"{minimum_accepts} are required"
            ),
            assumptions=CERTIFICATE_ASSUMPTIONS,
        )
    upper = clopper_pearson_upper(unsafe, len(accepted), alpha)
    valid = upper <= target_outage
    return ReliabilityCertificate(
        valid=valid,
        target_outage=float(target_outage),
        upper_bound=float(upper),
        confidence=confidence,
        calibration_samples=len(items),
        accepted_samples=len(accepted),
        unsafe_accepted=unsafe,
        threshold=float(threshold),
        reason="" if valid else "one-sided unsafe-acceptance bound exceeds the target",
        assumptions=CERTIFICATE_ASSUMPTIONS,
    )


def certify_fixed_policy(
    outcomes: Iterable[tuple[bool, bool]],
    target_outage: float,
    alpha: float,
    minimum_accepts: int,
    threshold: float = 0.0,
) -> ReliabilityCertificate:
    """Certify the final outcomes of a fixed, possibly multi-stage policy.

    Each pair is ``(accepted, unsafe)``.  This is the appropriate certificate
    for the composed accept/refine/reject policy because it evaluates the
    actual route taken through all refinement stages.
    """

    outcomes = list(outcomes)
    accepted = [unsafe for is_accepted, unsafe in outcomes if is_accepted]
    unsafe_count = sum(int(unsafe) for unsafe in accepted)
    confidence = 1.0 - float(alpha)
    minimum_accepts = max(1, int(minimum_accepts))
    if len(accepted) < minimum_accepts:
        return ReliabilityCertificate(
            valid=False,
            target_outage=float(target_outage),
            upper_bound=1.0,
            confidence=confidence,
            calibration_samples=len(outcomes),
            accepted_samples=len(accepted),
            unsafe_accepted=unsafe_count,
            threshold=float(threshold),
            method="clopper-pearson-composed-policy",
            reason=(
                f"only {len(accepted)} certificate examples were accepted; "
                f"{minimum_accepts} are required"
            ),
            assumptions=CERTIFICATE_ASSUMPTIONS,
        )
    upper = clopper_pearson_upper(unsafe_count, len(accepted), alpha)
    valid = upper <= target_outage
    return ReliabilityCertificate(
        valid=valid,
        target_outage=float(target_outage),
        upper_bound=float(upper),
        confidence=confidence,
        calibration_samples=len(outcomes),
        accepted_samples=len(accepted),
        unsafe_accepted=unsafe_count,
        threshold=float(threshold),
        method="clopper-pearson-composed-policy",
        reason="" if valid else "one-sided unsafe-acceptance bound exceeds the target",
        assumptions=CERTIFICATE_ASSUMPTIONS,
    )


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """Exact one-sided Clopper-Pearson upper confidence limit.

    ``successes`` denotes unsafe accepted decisions.  The implementation uses
    a monotone binary search over the binomial CDF and requires no SciPy
    dependency.
    """

    successes = int(successes)
    trials = int(trials)
    alpha = float(alpha)
    if trials <= 0:
        return 1.0
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if successes == trials:
        return 1.0
    if successes == 0:
        return float(1.0 - alpha ** (1.0 / trials))

    lower = successes / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        cdf = _binomial_cdf(successes, trials, midpoint)
        # P_p[X <= k] decreases monotonically with p.
        if cdf > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) / 2.0)


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return float(k >= n)
    logs = [
        (
            lgamma(n + 1)
            - lgamma(index + 1)
            - lgamma(n - index + 1)
            + index * log(probability)
            + (n - index) * log1m(probability)
        )
        for index in range(k + 1)
    ]
    maximum = max(logs)
    return float(exp(maximum) * sum(exp(value - maximum) for value in logs))


def log1m(value: float) -> float:
    """Stable log(1-value) for probabilities away from one."""

    return float(np.log1p(-value))


@dataclass
class ChannelSupportProfile:
    """Empirical channel envelope fixed before policy certification.

    This guard is a conservative domain-of-use check, not a claim that every
    point inside the envelope has the same distribution.  The end-to-end
    accepted-outage certificate is fitted with this guard already active.
    """

    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    relative_margin: float = 0.10
    absolute_margin: float = 1e-6
    bounds: dict[str, tuple[float, float]] | None = None

    _KEYS = (
        "channel_kind_code",
        "snr_db",
        "effective_snr_db",
        "gain",
        "interference_power",
        "blockage_probability",
        "doppler_hz",
        "burst_probability",
        "csi_error",
        "phy_payload_bit_error_rate",
        "phy_ldpc_block_error_rate",
        "phy_payload_mse",
        "phy_quantization_mse",
    )

    def fit(self, states: Iterable[dict[str, float]]) -> None:
        collected: dict[str, list[float]] = {}
        for state in states:
            for key in self._KEYS:
                value = state.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    collected.setdefault(key, []).append(float(value))
        bounds: dict[str, tuple[float, float]] = {}
        lower_q = float(np.clip(self.lower_quantile, 0.0, 1.0))
        upper_q = float(np.clip(self.upper_quantile, lower_q, 1.0))
        for key, values in collected.items():
            lower = float(np.quantile(values, lower_q))
            upper = float(np.quantile(values, upper_q))
            width = max(upper - lower, abs(lower), abs(upper), 1.0)
            margin = max(self.absolute_margin, self.relative_margin * width)
            bounds[key] = (lower - margin, upper + margin)
        self.bounds = bounds

    @property
    def fitted(self) -> bool:
        return bool(self.bounds)

    def evaluate(self, state: dict[str, float]) -> tuple[bool, float]:
        if not self.bounds:
            return False, 1.0
        maximum_violation = 0.0
        for key, (lower, upper) in self.bounds.items():
            value = state.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                return False, 1.0
            scale = max(upper - lower, self.absolute_margin, 1e-12)
            if value < lower:
                maximum_violation = max(maximum_violation, (lower - value) / scale)
            elif value > upper:
                maximum_violation = max(maximum_violation, (value - upper) / scale)
        return maximum_violation <= 0.0, float(maximum_violation)

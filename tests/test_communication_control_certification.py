from pathlib import Path
from types import SimpleNamespace

import numpy as np

from opensemcom.cli.communication_control_suite import (
    KNOWN_CLASSES,
    Row,
    Scored,
    certify_progressive_policy,
    certify_single_policy,
    eval_single,
    make_fullopen_split,
    select_progressive_policy,
    select_single_policy,
    split_audit,
)


def scored(risk: np.ndarray, predictions: np.ndarray | None = None) -> Scored:
    predictions = (
        np.zeros(len(risk), dtype=np.int64)
        if predictions is None
        else predictions
    )
    return Scored(
        pred=np.asarray(predictions, dtype=np.int64),
        risk=np.asarray(risk, dtype=np.float64),
    )


def test_headline_single_policy_is_selected_then_certified_on_disjoint_data():
    selection = scored(np.linspace(0.0, 1.0, 100))
    y_selection = np.zeros(100, dtype=np.int64)
    policy = select_single_policy(
        selection,
        y_selection,
        np.zeros(100, dtype=bool),
        target=0.05,
        budget=2.0,
        accept_cost=1.0,
    )
    certificate = scored(np.linspace(0.0, 1.0, 80))

    certified = certify_single_policy(
        policy,
        certificate,
        np.zeros(80, dtype=np.int64),
        np.zeros(80, dtype=bool),
        target=0.05,
        accept_cost=1.0,
    )

    assert certified["certificate_valid"]
    assert certified["certificate_accepted"] == 80
    assert certified["certificate_upper_bound"] < 0.05
    assert certified["threshold"] == certified["selected_threshold"]


def test_invalid_headline_certificate_deploys_reject_all_threshold():
    policy = {"threshold": 1.0}
    predictions = np.zeros(80, dtype=np.int64)
    predictions[0] = 1
    certificate = scored(np.linspace(0.0, 1.0, 80), predictions)

    certified = certify_single_policy(
        policy,
        certificate,
        np.zeros(80, dtype=np.int64),
        np.zeros(80, dtype=bool),
        target=0.05,
        accept_cost=1.0,
    )

    assert not certified["certificate_valid"]
    assert certified["certificate_upper_bound"] > 0.05
    assert certified["threshold"] == float("-inf")

    lower_risk_evaluation = scored(np.asarray([-1.0, -2.0]))
    metrics = eval_single(
        lower_risk_evaluation,
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=bool),
        certified["threshold"],
        accept_cost=1.0,
    )
    assert metrics["accepted"] == 0


def test_progressive_certificate_covers_final_composed_route():
    risks = np.linspace(0.0, 1.0, 100)
    core_selection = scored(risks)
    refine_selection = scored(risks[::-1])
    selection_policy = select_progressive_policy(
        core_selection,
        refine_selection,
        np.zeros(100, dtype=np.int64),
        np.zeros(100, dtype=bool),
        target=0.05,
        budget=2.0,
    )
    core_certificate = scored(np.linspace(0.0, 1.0, 80))
    refine_certificate = scored(np.linspace(1.0, 0.0, 80))

    certified = certify_progressive_policy(
        selection_policy,
        core_certificate,
        refine_certificate,
        np.zeros(80, dtype=np.int64),
        np.zeros(80, dtype=bool),
        target=0.05,
    )

    assert certified["certificate_valid"]
    assert certified["certificate_method"] == "clopper-pearson-composed-policy"
    assert certified["q1"] == certified["selected_q1"]


def test_fullopen_split_preserves_every_declared_severity_when_eval_is_limited():
    rows = []
    source_index = 0
    for label in range(KNOWN_CLASSES):
        for _ in range(128):
            source = f"/known/{source_index}.npy"
            rows.append(
                Row(
                    key=(source, "0"),
                    paths={"dino": Path(source)},
                    raw_source_path=source,
                    dataset="cifar10",
                    label=label,
                    task="classification",
                    domain="cifar10",
                    is_unknown=False,
                    split="eval",
                    regime="closed-id",
                )
            )
            source_index += 1
    for open_index in range(4000):
        source = f"/open/{open_index}.npy"
        rows.append(
            Row(
                key=(source, "0"),
                paths={"dino": Path(source)},
                raw_source_path=source,
                dataset="cifar100",
                label=KNOWN_CLASSES,
                task="classification",
                domain="cifar100",
                is_unknown=True,
                split="eval",
                regime="full-open",
            )
        )
    args = SimpleNamespace(
        train_known_per_class=64,
        cal_known_per_class=32,
        train_open=512,
        cal_open=2000,
        eval_size=1024,
        certificate_fraction=0.70,
    )

    for name, fraction in (
        ("mild", 0.25),
        ("medium", 0.50),
        ("hard", 0.75),
        ("extreme", 0.91),
    ):
        split = make_fullopen_split(rows, 20260730, args, fraction)
        audit = split_audit(f"full-open-{name}", 20260730, split)

        for cohort in ("policy_selection", "certificate", "eval"):
            total = audit["rows"][cohort]
            tolerance = max(0.01, 1.0 / total)
            assert abs(audit["open_fraction"][cohort] - fraction) <= tolerance

"""Evaluate and independently certify exact DeepSense top-k beam policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from opensemcom.cli.communication_control_suite import (
    ChannelContext,
    Row,
    SplitData,
    load_channel_context,
    load_rows,
    make_features,
    parse_manifest_specs,
    row_key_sha256,
    shuffled,
    unique_source_rows,
    validate_split_disjoint,
)
from opensemcom.certification import clopper_pearson_upper, minimum_zero_error_accepts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepSense exact beam top-k experiments.")
    parser.add_argument("--feature-manifest", action="append", required=True, help="name=manifest.csv")
    parser.add_argument("--deepsense-scenario-root", default="data/deepsense6g/Scenario1")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--targets", default="0.05,0.10")
    parser.add_argument("--certification-alpha", type=float, default=0.05)
    parser.add_argument(
        "--certificate-family-size",
        type=int,
        default=3,
        help="Bonferroni family size; defaults to the three compared methods",
    )
    parser.add_argument("--selection-safety-factor", type=float, default=0.50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.certificate_family_size < 1:
        raise ValueError("--certificate-family-size must be at least one")
    specs = parse_manifest_specs(args.feature_manifest)
    rows, manifest_summary = load_rows(specs)
    channel = load_channel_context(Path(args.deepsense_scenario_root))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for seed in seeds:
        split = make_exact_split(rows, seed)
        y_train = labels(split["train"])
        y_selection = labels(split["policy_selection"])
        y_certificate = labels(split["certificate"])
        y_eval = labels(split["eval"])
        n_classes = int(
            max(
                np.max(y_train),
                np.max(y_selection),
                np.max(y_certificate),
                np.max(y_eval),
            )
        ) + 1
        feature_sets = {
            "dino_logistic": (
                make_features(split["train"], ("dino",), channel, False),
                make_features(split["policy_selection"], ("dino",), channel, False),
                make_features(split["certificate"], ("dino",), channel, False),
                make_features(split["eval"], ("dino",), channel, False),
                "logistic",
            ),
            "ensemble_logistic": (
                make_features(split["train"], tuple(specs), channel, False),
                make_features(split["policy_selection"], tuple(specs), channel, False),
                make_features(split["certificate"], tuple(specs), channel, False),
                make_features(split["eval"], tuple(specs), channel, False),
                "logistic",
            ),
            "opensemcom_channel_mlp": (
                make_features(split["train"], tuple(specs), channel, True),
                make_features(split["policy_selection"], tuple(specs), channel, True),
                make_features(split["certificate"], tuple(specs), channel, True),
                make_features(split["eval"], tuple(specs), channel, True),
                "mlp",
            ),
        }
        for method, (
            x_train,
            x_selection,
            x_certificate,
            x_eval,
            model_type,
        ) in feature_sets.items():
            model = fit_prob_model(x_train, y_train, n_classes, model_type, seed)
            selection_prob = predict_full_proba(model, x_selection, n_classes)
            certificate_prob = predict_full_proba(model, x_certificate, n_classes)
            eval_prob = predict_full_proba(model, x_eval, n_classes)
            for target in targets:
                selected_threshold = select_topk_threshold(
                    selection_prob,
                    y_selection,
                    target,
                    k=5,
                    safety_factor=args.selection_safety_factor,
                )
                certificate = certify_topk_threshold(
                    certificate_prob,
                    y_certificate,
                    selected_threshold,
                    target,
                    k=5,
                    alpha=(
                        args.certification_alpha
                        / args.certificate_family_size
                    ),
                )
                certificate["certificate_family_alpha"] = args.certification_alpha
                certificate["certificate_family_size"] = args.certificate_family_size
                threshold = (
                    selected_threshold
                    if certificate["certificate_valid"]
                    else float("-inf")
                )
                metrics = eval_topk(eval_prob, y_eval, threshold, k=5)
                summary_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "target_outage": target,
                        "selected_threshold": selected_threshold,
                        "threshold": threshold,
                        **certificate,
                        **metrics,
                    }
                )

    write_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), summary_rows)
    output_prefix.with_name(output_prefix.name + "_manifest_summary.json").write_text(
        json.dumps(
            {
                **manifest_summary,
                "deepsense_feature_rows": len(
                    [r for r in rows if r.dataset == "deepsense6g"]
                ),
                "certification_family_alpha": args.certification_alpha,
                "certificate_family_size": args.certificate_family_size,
                "certification_policy_alpha": (
                    args.certification_alpha / args.certificate_family_size
                ),
                "selection_safety_factor": args.selection_safety_factor,
                "split_audits": [
                    {
                        "seed": seed,
                        "rows": {
                            name: len(values)
                            for name, values in make_exact_split(rows, seed).items()
                        },
                        "row_key_sha256": {
                            name: row_key_sha256(values)
                            for name, values in make_exact_split(rows, seed).items()
                        },
                    }
                    for seed in seeds
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(summary_rows), "output_prefix": str(output_prefix)}, indent=2))


def make_exact_split(rows: list[Row], seed: int) -> dict[str, list[Row]]:
    rng = np.random.default_rng(seed)
    ds = unique_source_rows(
        r
        for r in rows
        if r.dataset == "deepsense6g" and r.task == "beam-prediction"
    )
    raw_labels = sorted({r.label for r in ds})
    label_rank = {label: idx for idx, label in enumerate(raw_labels)}
    remapped = [Row(r.key, r.paths, r.raw_source_path, r.dataset, label_rank[r.label], r.task, r.domain, False, r.split, r.regime) for r in ds]
    by_label: dict[int, list[Row]] = {}
    for row in remapped:
        by_label.setdefault(row.label, []).append(row)
    train: list[Row] = []
    policy_selection: list[Row] = []
    certificate: list[Row] = []
    eval_rows: list[Row] = []
    for values in by_label.values():
        values = shuffled(values, rng)
        n = len(values)
        train_end = max(1, int(0.50 * n))
        selection_end = max(train_end + 1, int(0.65 * n))
        certificate_end = max(selection_end + 1, int(0.85 * n))
        train += values[:train_end]
        policy_selection += values[train_end:selection_end]
        certificate += values[selection_end:certificate_end]
        eval_rows += values[certificate_end:]
    split = {
        "train": shuffled(train, rng),
        "policy_selection": shuffled(policy_selection, rng),
        "certificate": shuffled(certificate, rng),
        "eval": shuffled(eval_rows, rng),
    }
    validate_split_disjoint(
        SplitData(
            split["train"],
            split["policy_selection"],
            split["certificate"],
            split["eval"],
        )
    )
    return split


def labels(rows: list[Row]) -> np.ndarray:
    return np.asarray([r.label for r in rows], dtype=np.int64)


def fit_prob_model(x: np.ndarray, y: np.ndarray, n_classes: int, model_type: str, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=8e-4,
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        )
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
    return make_pipeline(StandardScaler(), clf).fit(x, y)


def predict_full_proba(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    probs = model.predict_proba(x)
    classes = model[-1].classes_
    full = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    for idx, cls in enumerate(classes):
        full[:, int(cls)] = probs[:, idx]
    row_sum = np.maximum(full.sum(axis=1, keepdims=True), 1e-12)
    return full / row_sum


def select_topk_threshold(
    probs: np.ndarray,
    y: np.ndarray,
    target: float,
    k: int,
    safety_factor: float = 0.5,
) -> float:
    risks = topk_risk(probs, k)
    best = float(np.min(risks) - 1e-6)
    best_goodput = -1.0
    for threshold in np.unique(np.quantile(risks, np.linspace(0.0, 1.0, 101))):
        metrics = eval_topk(probs, y, float(threshold), k)
        if (
            metrics["accepted_open_outage"] <= target * safety_factor
            and metrics["semantic_goodput"] > best_goodput
        ):
            best = float(threshold)
            best_goodput = metrics["semantic_goodput"]
    return best


def certify_topk_threshold(
    probs: np.ndarray,
    y: np.ndarray,
    threshold: float,
    target: float,
    k: int,
    alpha: float = 0.05,
) -> dict:
    metrics = eval_topk(probs, y, threshold, k)
    accepted = int(metrics["accepted"])
    unsafe = int(metrics["accepted_unsafe"])
    required = minimum_zero_error_accepts(target, alpha)
    upper = (
        clopper_pearson_upper(unsafe, accepted, alpha)
        if accepted
        else 1.0
    )
    valid = accepted >= required and upper <= target
    return {
        "certificate_valid": valid,
        "certificate_method": "clopper-pearson-fixed-topk-policy",
        "certificate_alpha": alpha,
        "certificate_confidence": 1.0 - alpha,
        "certificate_target_outage": target,
        "certificate_upper_bound": upper,
        "certificate_samples": len(y),
        "certificate_accepted": accepted,
        "certificate_unsafe": unsafe,
        "certificate_minimum_accepts": required,
        "certificate_reason": (
            ""
            if valid
            else (
                f"accepted {accepted} certificate examples; {required} required"
                if accepted < required
                else "exact one-sided accepted-outage bound exceeds target"
            )
        ),
    }


def eval_topk(probs: np.ndarray, y: np.ndarray, threshold: float, k: int) -> dict:
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top3 = order[:, : min(3, probs.shape[1])]
    top5 = order[:, : min(k, probs.shape[1])]
    risk = topk_risk(probs, k)
    selected = risk <= threshold
    top5_correct = np.asarray([label in row for label, row in zip(y, top5)], dtype=bool)
    accepted_correct = np.logical_and(selected, top5_correct)
    accepted_unsafe = np.logical_and(selected, ~top5_correct)
    accepted = int(np.sum(selected))
    return {
        "top1_accuracy": float(np.mean(top1 == y)),
        "top3_accuracy": float(np.mean([label in row for label, row in zip(y, top3)])),
        "top5_accuracy": float(np.mean(top5_correct)),
        "semantic_goodput": float(np.mean(accepted_correct)),
        "coverage": float(np.mean(selected)),
        "accepted_open_outage": float(np.sum(accepted_unsafe) / max(accepted, 1)),
        "accepted": accepted,
        "accepted_correct": int(np.sum(accepted_correct)),
        "accepted_unsafe": int(np.sum(accepted_unsafe)),
    }


def topk_risk(probs: np.ndarray, k: int) -> np.ndarray:
    order = np.sort(probs, axis=1)[:, ::-1]
    return 1.0 - np.sum(order[:, : min(k, probs.shape[1])], axis=1)


def reject_threshold(risks: np.ndarray) -> float:
    return float(np.min(risks) - 1e-6) if len(risks) else -1.0


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

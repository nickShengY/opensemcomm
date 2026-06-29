"""Evaluate exact DeepSense beam top-k prediction on feature-backed rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from opensemcom.cli.communication_control_suite import (
    ChannelContext,
    Row,
    load_channel_context,
    load_rows,
    make_features,
    parse_manifest_specs,
    shuffled,
    wilson_upper,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepSense exact beam top-k experiments.")
    parser.add_argument("--feature-manifest", action="append", required=True, help="name=manifest.csv")
    parser.add_argument("--deepsense-scenario-root", default="data/deepsense6g/Scenario1")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--targets", default="0.05,0.10")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        y_cal = labels(split["cal"])
        y_eval = labels(split["eval"])
        n_classes = int(max(np.max(y_train), np.max(y_cal), np.max(y_eval))) + 1
        feature_sets = {
            "dino_logistic": (
                make_features(split["train"], ("dino",), channel, False),
                make_features(split["cal"], ("dino",), channel, False),
                make_features(split["eval"], ("dino",), channel, False),
                "logistic",
            ),
            "ensemble_logistic": (
                make_features(split["train"], tuple(specs), channel, False),
                make_features(split["cal"], tuple(specs), channel, False),
                make_features(split["eval"], tuple(specs), channel, False),
                "logistic",
            ),
            "opensemcom_channel_mlp": (
                make_features(split["train"], tuple(specs), channel, True),
                make_features(split["cal"], tuple(specs), channel, True),
                make_features(split["eval"], tuple(specs), channel, True),
                "mlp",
            ),
        }
        for method, (x_train, x_cal, x_eval, model_type) in feature_sets.items():
            model = fit_prob_model(x_train, y_train, n_classes, model_type, seed)
            cal_prob = predict_full_proba(model, x_cal, n_classes)
            eval_prob = predict_full_proba(model, x_eval, n_classes)
            for target in targets:
                threshold = select_topk_threshold(cal_prob, y_cal, target, k=5)
                metrics = eval_topk(eval_prob, y_eval, threshold, k=5)
                summary_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "target_outage": target,
                        "threshold": threshold,
                        **metrics,
                    }
                )

    write_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), summary_rows)
    output_prefix.with_name(output_prefix.name + "_manifest_summary.json").write_text(
        json.dumps({**manifest_summary, "deepsense_feature_rows": len([r for r in rows if r.dataset == "deepsense6g"])}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(summary_rows), "output_prefix": str(output_prefix)}, indent=2))


def make_exact_split(rows: list[Row], seed: int) -> dict[str, list[Row]]:
    rng = np.random.default_rng(seed)
    ds = [r for r in rows if r.dataset == "deepsense6g" and r.task == "beam-prediction"]
    raw_labels = sorted({r.label for r in ds})
    label_rank = {label: idx for idx, label in enumerate(raw_labels)}
    remapped = [Row(r.key, r.paths, r.raw_source_path, r.dataset, label_rank[r.label], r.task, r.domain, False, r.split, r.regime) for r in ds]
    by_label: dict[int, list[Row]] = {}
    for row in remapped:
        by_label.setdefault(row.label, []).append(row)
    train: list[Row] = []
    cal: list[Row] = []
    eval_rows: list[Row] = []
    for values in by_label.values():
        values = shuffled(values, rng)
        n = len(values)
        train += values[: max(1, int(0.50 * n))]
        cal += values[max(1, int(0.50 * n)) : max(2, int(0.75 * n))]
        eval_rows += values[max(2, int(0.75 * n)) :]
    return {"train": shuffled(train, rng), "cal": shuffled(cal, rng), "eval": shuffled(eval_rows, rng)}


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


def select_topk_threshold(probs: np.ndarray, y: np.ndarray, target: float, k: int) -> float:
    risks = topk_risk(probs, k)
    best = float(np.min(risks) - 1e-6)
    best_goodput = -1.0
    for threshold in np.unique(np.quantile(risks, np.linspace(0.0, 1.0, 101))):
        metrics = eval_topk(probs, y, float(threshold), k)
        if wilson_upper(metrics["accepted_unsafe"], metrics["accepted"]) <= target and metrics["semantic_goodput"] > best_goodput:
            best = float(threshold)
            best_goodput = metrics["semantic_goodput"]
    return best


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


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

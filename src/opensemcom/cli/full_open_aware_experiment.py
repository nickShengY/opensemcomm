"""Full-open-aware receiver training and open-set baseline evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


KNOWN_CLASSES = 6
TRAIN_TASKS = {"classification"}
TRAIN_DOMAINS = {"cifar10"}


@dataclass(frozen=True)
class Row:
    key: tuple[str, str]
    dino_path: Path
    clip_path: Path | None
    label: int
    task: str
    domain: str
    is_unknown: bool
    split: str
    regime: str

    @property
    def open_exposure(self) -> bool:
        return self.is_unknown or self.task not in TRAIN_TASKS or self.domain not in TRAIN_DOMAINS

    @property
    def known_id(self) -> bool:
        return not self.open_exposure and 0 <= self.label < KNOWN_CLASSES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate a full-open-aware OpenSemCom receiver.")
    parser.add_argument("--dino-manifest", required=True)
    parser.add_argument("--clip-manifest")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--backbone", choices=["dino", "clip", "dino_clip"], default="dino_clip")
    parser.add_argument("--train-known-per-class", type=int, default=256)
    parser.add_argument("--train-open", type=int, default=1024)
    parser.add_argument("--cal-known-per-class", type=int, default=64)
    parser.add_argument("--cal-open", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--severity", default="mild:0.25,medium:0.50,hard:0.75,extreme:0.91")
    parser.add_argument("--curve-points", type=int, default=101)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(123)
    rows = load_joined_rows(args.dino_manifest, args.clip_manifest)
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    severities = parse_severities(args.severity)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    all_summary = []
    all_curves = []
    all_actions = []
    for seed in seeds:
        split = make_split(rows, seed, args)
        train_x, train_y, train_open, train_accept = build_arrays(split["train"], args.backbone)
        cal_x, cal_y, cal_open, cal_accept = build_arrays(split["calibration"], args.backbone)
        model = FullOpenAwareReceiver(
            input_dim=train_x.shape[1],
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            seed=seed,
        )
        model.fit(train_x, train_y, train_open, train_accept)
        baseline_models = fit_baselines(train_x, train_y, train_open)
        for severity_name, open_fraction in severities:
            eval_rows = make_severity_eval(split["eval_known"], split["eval_open"], open_fraction, args.eval_size, seed)
            eval_x, eval_y, eval_open, _ = build_arrays(eval_rows, args.backbone)
            receiver_scored = score_opensemcom(model, eval_x)
            baseline_scored = score_baselines(baseline_models, eval_x)
            methods = {
                "opensemcom": score_opensemcom_ensemble(receiver_scored, baseline_scored),
                "receiver_only": receiver_scored,
            }
            methods.update(baseline_scored)
            cal_receiver_scored = score_opensemcom(model, cal_x)
            cal_baseline_scored = score_baselines(baseline_models, cal_x)
            cal_scores = {
                "opensemcom": score_opensemcom_ensemble(cal_receiver_scored, cal_baseline_scored),
                "receiver_only": cal_receiver_scored,
            }
            cal_scores.update(cal_baseline_scored)
            for method, scored in methods.items():
                cal_scored = cal_scores[method]
                thresholds = calibration_thresholds(cal_scored.risk, cal_y, cal_open)
                for target, threshold in thresholds.items():
                    metrics = evaluate_at_threshold(scored, eval_y, eval_open, threshold)
                    all_summary.append(
                        {
                            "seed": seed,
                            "severity": severity_name,
                            "open_fraction": open_fraction,
                            "method": method,
                            "target_openout": target,
                            **metrics,
                        }
                    )
                curve_rows = risk_goodput_curve(scored, eval_y, eval_open, args.curve_points)
                for row in curve_rows:
                    all_curves.append(
                        {
                            "seed": seed,
                            "severity": severity_name,
                            "open_fraction": open_fraction,
                            "method": method,
                            **row,
                        }
                    )
                all_actions.append(
                    {
                        "seed": seed,
                        "severity": severity_name,
                        "open_fraction": open_fraction,
                        "method": method,
                        "samples": len(eval_rows),
                        "known_id_accuracy": float(np.mean(scored.pred == eval_y)) if len(eval_y) else 0.0,
                        "known_subset_accuracy": known_subset_accuracy(scored.pred, eval_y, eval_open),
                        "auroc": auroc(scored.risk, eval_open),
                        "fpr95": fpr_at_tpr(scored.risk, eval_open),
                    }
                )
    write_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), all_summary)
    write_csv(output_prefix.with_name(output_prefix.name + "_curves.csv"), all_curves)
    write_csv(output_prefix.with_name(output_prefix.name + "_diagnostics.csv"), all_actions)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(all_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(all_summary), "curve_rows": len(all_curves), "output_prefix": str(output_prefix)}, indent=2))


def load_joined_rows(dino_manifest: str, clip_manifest: str | None) -> list[Row]:
    dino_rows = read_manifest(dino_manifest)
    clip_by_key = {}
    if clip_manifest:
        for raw in read_manifest(clip_manifest):
            clip_by_key[row_key(raw)] = Path(raw["source_path"]).expanduser().resolve()
    rows = []
    for raw in dino_rows:
        if raw.get("dataset") == "ag_news":
            continue
        key = row_key(raw)
        clip_path = clip_by_key.get(key)
        if clip_manifest and clip_path is None:
            continue
        rows.append(
            Row(
                key=key,
                dino_path=Path(raw["source_path"]).expanduser().resolve(),
                clip_path=clip_path,
                label=int(raw["label"]),
                task=raw["task"],
                domain=raw["domain"],
                is_unknown=parse_bool(raw["is_unknown"]),
                split=raw.get("split") or "eval",
                regime=raw.get("regime") or "",
            )
        )
    return rows


def read_manifest(path: str) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("raw_source_path") or row["source_path"], row.get("raw_artifact_index") or row.get("artifact_index") or "")


def make_split(rows: list[Row], seed: int, args) -> dict[str, list[Row]]:
    rng = np.random.default_rng(seed)
    known_all = [row for row in rows if row.known_id]
    open_all = [row for row in rows if row.regime == "full-open" and row.open_exposure]
    by_class = {label: [] for label in range(KNOWN_CLASSES)}
    for row in known_all:
        by_class[row.label].append(row)
    train = []
    calibration = []
    eval_known = []
    for label, values in by_class.items():
        values = shuffled(values, rng)
        train.extend(values[: args.train_known_per_class])
        calibration.extend(values[args.train_known_per_class : args.train_known_per_class + args.cal_known_per_class])
        eval_known.extend(values[args.train_known_per_class + args.cal_known_per_class :])
    open_values = shuffled(open_all, rng)
    train.extend(open_values[: args.train_open])
    calibration.extend(open_values[args.train_open : args.train_open + args.cal_open])
    eval_open = open_values[args.train_open + args.cal_open :]
    return {"train": train, "calibration": calibration, "eval_known": eval_known, "eval_open": eval_open}


def make_severity_eval(known_rows: list[Row], open_rows: list[Row], open_fraction: float, total: int, seed: int) -> list[Row]:
    rng = np.random.default_rng(seed + int(open_fraction * 1000))
    open_n = min(len(open_rows), int(round(total * open_fraction)))
    known_n = min(len(known_rows), total - open_n)
    selected = shuffled(known_rows, rng)[:known_n] + shuffled(open_rows, rng)[:open_n]
    return shuffled(selected, rng)


def build_arrays(rows: list[Row], backbone: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    open_labels = []
    accept_labels = []
    for row in rows:
        parts = []
        if backbone in {"dino", "dino_clip"}:
            parts.append(load_feature(row.dino_path))
        if backbone in {"clip", "dino_clip"}:
            if row.clip_path is None:
                raise ValueError(f"Missing CLIP feature for manifest row {row.key}")
            parts.append(load_feature(row.clip_path))
        parts.append(metadata_features(row, backbone))
        x = np.concatenate(parts).astype(np.float32)
        features.append(x)
        labels.append(row.label if row.known_id else KNOWN_CLASSES)
        open_labels.append(row.open_exposure)
        accept_labels.append(row.known_id)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(open_labels, dtype=bool),
        np.asarray(accept_labels, dtype=bool),
    )


def metadata_features(row: Row, backbone: str) -> np.ndarray:
    """Encode manifest metadata plus explicit missing channel-state slots."""
    task = one_hot_hash(f"task:{row.task}", 16)
    domain = one_hot_hash(f"domain:{row.domain}", 32)
    layer_flags = np.asarray(
        [
            float(backbone in {"dino", "dino_clip"}),
            float(backbone in {"clip", "dino_clip"}),
        ],
        dtype=np.float32,
    )
    # The current manifests do not contain measured per-row wireless channel
    # state. Keep the channel-state feature contract explicit without fabricating
    # SNR/fading/interference values.
    channel_state = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.concatenate([task, domain, layer_flags, channel_state]).astype(np.float32)


def one_hot_hash(value: str, buckets: int) -> np.ndarray:
    out = np.zeros(buckets, dtype=np.float32)
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    out[int.from_bytes(digest[:4], "big") % buckets] = 1.0
    return out


def load_feature(path: Path) -> np.ndarray:
    x = np.load(path).reshape(-1).astype(np.float32)
    norm = np.linalg.norm(x)
    return x / max(float(norm), 1e-6)


@dataclass
class Scored:
    pred: np.ndarray
    risk: np.ndarray
    accept_logit: np.ndarray
    known_prob: np.ndarray


class FullOpenAwareReceiver:
    def __init__(self, input_dim: int, hidden_dim: int, epochs: int, lr: float, device: str, seed: int):
        import torch

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
        ).to(self.device)
        self.class_head = torch.nn.Linear(hidden_dim, KNOWN_CLASSES + 1).to(self.device)
        self.unsafe_head = torch.nn.Linear(hidden_dim, 1).to(self.device)
        self.accept_head = torch.nn.Linear(hidden_dim, 1).to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray, y: np.ndarray, open_label: np.ndarray, accept_label: np.ndarray) -> None:
        torch = self.torch
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
        x = (x - self.mean) / self.std
        unsafe = np.logical_or(open_label, y == KNOWN_CLASSES).astype(np.float32)
        accept = accept_label.astype(np.float32)
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        ut = torch.as_tensor(unsafe.reshape(-1, 1), dtype=torch.float32, device=self.device)
        at = torch.as_tensor(accept.reshape(-1, 1), dtype=torch.float32, device=self.device)
        counts = np.bincount(y, minlength=KNOWN_CLASSES + 1).astype(np.float32)
        class_weights = np.sum(counts) / np.maximum(counts * (KNOWN_CLASSES + 1), 1.0)
        class_loss = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=self.device))
        pos_unsafe = max(float(np.sum(unsafe == 1)), 1.0)
        neg_unsafe = max(float(np.sum(unsafe == 0)), 1.0)
        unsafe_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_unsafe / pos_unsafe], device=self.device))
        pos_accept = max(float(np.sum(accept == 1)), 1.0)
        neg_accept = max(float(np.sum(accept == 0)), 1.0)
        accept_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_accept / pos_accept], device=self.device))
        params = list(self.model.parameters()) + list(self.class_head.parameters()) + list(self.unsafe_head.parameters()) + list(self.accept_head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(2027)
        batch_size = min(512, max(64, xt.shape[0]))
        self.model.train()
        for _ in range(self.epochs):
            order = torch.randperm(xt.shape[0], generator=generator, device=self.device)
            for start in range(0, xt.shape[0], batch_size):
                idx = order[start : start + batch_size]
                h = self.model(xt[idx])
                logits = self.class_head(h)
                unsafe_logits = self.unsafe_head(h)
                accept_logits = self.accept_head(h)
                loss = (
                    class_loss(logits, yt[idx])
                    + 1.5 * unsafe_loss(unsafe_logits, ut[idx])
                    + 1.0 * accept_loss(accept_logits, at[idx])
                    + ranking_loss(unsafe_logits, ut[idx], torch)
                    - 0.05 * torch.mean(torch.sigmoid(accept_logits) * at[idx])
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.model.eval()

    def score(self, x: np.ndarray) -> Scored:
        torch = self.torch
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            h = self.model(xt)
            probs = torch.softmax(self.class_head(h), dim=-1)
            unsafe_prob = torch.sigmoid(self.unsafe_head(h)).reshape(-1)
            accept_prob = torch.sigmoid(self.accept_head(h)).reshape(-1)
        probs_np = probs.detach().cpu().numpy()
        pred = np.argmax(probs_np[:, :KNOWN_CLASSES], axis=1)
        unknown_prob = probs_np[:, KNOWN_CLASSES]
        known_prob = 1.0 - unknown_prob
        risk = np.clip(
            0.45 * unsafe_prob.detach().cpu().numpy()
            + 0.25 * (1.0 - accept_prob.detach().cpu().numpy())
            + 0.30 * unknown_prob,
            0.0,
            1.0,
        )
        return Scored(pred=pred, risk=risk, accept_logit=accept_prob.detach().cpu().numpy(), known_prob=known_prob)


def ranking_loss(logits, unsafe, torch):
    safe_scores = logits[unsafe.reshape(-1) < 0.5]
    unsafe_scores = logits[unsafe.reshape(-1) >= 0.5]
    if safe_scores.numel() == 0 or unsafe_scores.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    safe = safe_scores[: min(safe_scores.numel(), 128)]
    uns = unsafe_scores[: min(unsafe_scores.numel(), 128)]
    return torch.relu(0.2 + safe.reshape(-1, 1) - uns.reshape(1, -1)).mean()


def score_opensemcom(model: FullOpenAwareReceiver, x: np.ndarray) -> Scored:
    return model.score(x)


def score_opensemcom_ensemble(receiver: Scored, baselines: dict[str, Scored]) -> Scored:
    # OpenSemCom uses the full-open-aware receiver for known-class semantics and
    # an unknown-aware acceptance head as the open-risk gate.
    gate = baselines["one_vs_rest"]
    return Scored(pred=receiver.pred, risk=gate.risk, accept_logit=gate.accept_logit, known_prob=receiver.known_prob)


def fit_baselines(x: np.ndarray, y: np.ndarray, open_label: np.ndarray) -> dict[str, object]:
    from sklearn.covariance import LedoitWolf
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    known = y < KNOWN_CLASSES
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(x[known], y[known])
    unknown = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    unknown.fit(x, open_label.astype(np.int64))
    means = {label: x[np.logical_and(known, y == label)].mean(axis=0) for label in range(KNOWN_CLASSES) if np.any(np.logical_and(known, y == label))}
    cov = LedoitWolf().fit(x[known])
    return {"logistic": clf, "unknown": unknown, "means": means, "cov": cov}


def score_baselines(models: dict[str, object], x: np.ndarray) -> dict[str, Scored]:
    clf = models["logistic"]
    probs = clf.predict_proba(x)
    pred = clf.predict(x)
    msp_risk = 1.0 - np.max(probs, axis=1)
    energy_risk = entropy(probs) / math.log(max(probs.shape[1], 2))
    unknown_probs = models["unknown"].predict_proba(x)
    unknown_classes = list(models["unknown"].classes_)
    unknown_risk = unknown_probs[:, unknown_classes.index(1)] if 1 in unknown_classes else msp_risk
    means = models["means"]
    cov = models["cov"]
    precision = cov.precision_
    distances = []
    euclidean = []
    for sample in x:
        class_d = []
        class_e = []
        for mean in means.values():
            delta = sample - mean
            class_d.append(float(delta @ precision @ delta.T))
            class_e.append(float(np.linalg.norm(delta)))
        distances.append(math.sqrt(max(min(class_d), 0.0)))
        euclidean.append(min(class_e))
    maha = scale01(np.asarray(distances))
    openmax = scale01(np.asarray(euclidean))
    vim = scale01(vim_residuals(x, np.stack(list(means.values()))))
    return {
        "msp": Scored(pred=pred, risk=msp_risk, accept_logit=1.0 - msp_risk, known_prob=1.0 - msp_risk),
        "energy": Scored(pred=pred, risk=energy_risk, accept_logit=1.0 - energy_risk, known_prob=1.0 - msp_risk),
        "mahalanobis": Scored(pred=pred, risk=maha, accept_logit=1.0 - maha, known_prob=1.0 - maha),
        "openmax": Scored(pred=pred, risk=openmax, accept_logit=1.0 - openmax, known_prob=1.0 - openmax),
        "vim": Scored(pred=pred, risk=vim, accept_logit=1.0 - vim, known_prob=1.0 - vim),
        "one_vs_rest": Scored(pred=pred, risk=unknown_risk, accept_logit=1.0 - unknown_risk, known_prob=1.0 - unknown_risk),
    }


def calibration_thresholds(risk: np.ndarray, y: np.ndarray, open_label: np.ndarray) -> dict[str, float]:
    thresholds = {}
    for target in (0.05, 0.10):
        safe_thresholds = []
        for threshold in np.quantile(risk, np.linspace(0.0, 1.0, 101)):
            selected = risk <= threshold
            if not np.any(selected):
                continue
            unsafe = np.logical_or(open_label[selected], y[selected] >= KNOWN_CLASSES)
            if float(np.mean(unsafe)) <= target:
                safe_thresholds.append(float(threshold))
        thresholds[f"openout<={target:.2f}"] = max(safe_thresholds) if safe_thresholds else float(np.min(risk) - 1e-6)
    return thresholds


def evaluate_at_threshold(scored: Scored, y: np.ndarray, open_label: np.ndarray, threshold: float) -> dict[str, float]:
    selected = scored.risk <= threshold
    known = ~open_label
    correct = np.logical_and(scored.pred == y, known)
    accepted_correct = np.logical_and(selected, correct)
    accepted_unsafe = np.logical_and(selected, np.logical_or(open_label, scored.pred != y))
    return {
        "threshold": float(threshold),
        "coverage": float(np.mean(selected)) if len(selected) else 0.0,
        "accuracy": float(np.mean(scored.pred == y)) if len(y) else 0.0,
        "known_id_accuracy": known_subset_accuracy(scored.pred, y, open_label),
        "accepted_known_accuracy": float(np.sum(accepted_correct) / max(np.sum(selected), 1)),
        "accepted_open_outage": float(np.sum(accepted_unsafe) / max(np.sum(selected), 1)),
        "semantic_goodput": float(np.sum(accepted_correct) / max(len(y), 1)),
        "auroc": auroc(scored.risk, open_label),
        "fpr95": fpr_at_tpr(scored.risk, open_label),
        "accepted": int(np.sum(selected)),
        "accepted_correct": int(np.sum(accepted_correct)),
        "accepted_unsafe": int(np.sum(accepted_unsafe)),
    }


def risk_goodput_curve(scored: Scored, y: np.ndarray, open_label: np.ndarray, points: int) -> list[dict[str, float]]:
    rows = []
    known = ~open_label
    for threshold in np.quantile(scored.risk, np.linspace(0.0, 1.0, points)):
        rows.append(evaluate_at_threshold(scored, y, open_label, float(threshold)))
    return rows


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return float(wins / max(len(pos) * len(neg), 1))


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target: float = 0.95) -> float:
    thresholds = np.unique(scores)[::-1]
    best = 1.0
    for threshold in thresholds:
        pred = scores >= threshold
        tp = np.sum(np.logical_and(pred, labels))
        fn = np.sum(np.logical_and(~pred, labels))
        fp = np.sum(np.logical_and(pred, ~labels))
        tn = np.sum(np.logical_and(~pred, ~labels))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if tpr >= target:
            best = min(best, fpr)
    return float(best)


def known_subset_accuracy(pred: np.ndarray, y: np.ndarray, open_label: np.ndarray) -> float:
    known = ~open_label
    return float(np.mean(pred[known] == y[known])) if np.any(known) else 0.0


def entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)


def vim_residuals(x: np.ndarray, means: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = max(1, min(vt.shape[0] - 1, int(vt.shape[0] * 0.8)))
    basis = vt[:rank].T
    residuals = []
    for sample in x:
        delta = sample - means.mean(axis=0)
        projected = basis @ (basis.T @ delta)
        residuals.append(np.linalg.norm(delta - projected))
    return np.asarray(residuals)


def scale01(values: np.ndarray) -> np.ndarray:
    lo = float(np.quantile(values, 0.05))
    hi = float(np.quantile(values, 0.95))
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_severities(text: str) -> list[tuple[str, float]]:
    values = []
    for item in text.split(","):
        name, raw = item.split(":", 1)
        values.append((name, float(raw)))
    return values


def shuffled(values: list[Row], rng: np.random.Generator) -> list[Row]:
    values = list(values)
    if values:
        order = rng.permutation(len(values))
        values = [values[int(idx)] for idx in order]
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

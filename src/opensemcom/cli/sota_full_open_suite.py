"""SOTA full-open safety-goodput comparison suite on feature manifests."""

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
    paths: dict[str, Path]
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


@dataclass
class Scored:
    pred: np.ndarray
    risk: np.ndarray
    known_prob: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run requested full-open SOTA comparison suite.")
    parser.add_argument("--feature-manifest", action="append", required=True, help="name=manifest.csv")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--main-features", default="dino", help="Comma-separated feature names for OpenSemCom.")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-known-per-class", type=int, default=192)
    parser.add_argument("--train-open", type=int, default=1024)
    parser.add_argument("--cal-known-per-class", type=int, default=64)
    parser.add_argument("--cal-open", type=int, default=768)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--severity", default="mild:0.25,medium:0.50,hard:0.75,extreme:0.91")
    parser.add_argument("--curve-points", type=int, default=101)
    parser.add_argument("--methods", default="all", help="Comma-separated methods to score, or all.")
    parser.add_argument("--targets", default="0.05,0.10", help="Comma-separated accepted OpenOut targets.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = parse_manifest_specs(args.feature_manifest)
    rows, manifest_summary = load_feature_rows(specs)
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    main_features = tuple(name for name in args.main_features.split(",") if name in specs)
    if not main_features:
        raise ValueError(f"No requested main features are available: {args.main_features}")
    severities = parse_severities(args.severity)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    requested_methods = parse_methods(args.methods)
    targets = tuple(float(value.strip()) for value in args.targets.split(",") if value.strip())

    all_summary: list[dict] = []
    all_curves: list[dict] = []
    all_diag: list[dict] = []

    for seed in seeds:
        split = make_split(rows, seed, args)
        row_sets = {
            "train": split["train"],
            "calibration": split["calibration"],
        }
        features_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        feature_groups = build_feature_groups(specs, main_features)
        if requested_methods == {"opensemcom"}:
            feature_groups = {"main": main_features}
        for group_name, feature_names in feature_groups.items():
            features_cache[("train", group_name)] = build_arrays(row_sets["train"], feature_names)
            features_cache[("calibration", group_name)] = build_arrays(row_sets["calibration"], feature_names)

        receiver = FullOpenAwareReceiver(
            input_dim=features_cache[("train", "main")][0].shape[1],
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            seed=seed,
        )
        train_x, train_y, train_open, train_accept = features_cache[("train", "main")]
        receiver.fit(train_x, train_y, train_open, train_accept)

        heads = {}
        for group_name in feature_groups:
            x, y, open_label, _ = features_cache[("train", group_name)]
            heads[group_name] = fit_heads(x, y, open_label, feature_groups[group_name])

        for severity_name, open_fraction in severities:
            eval_rows = make_severity_eval(split["eval_known"], split["eval_open"], open_fraction, args.eval_size, seed)
            for group_name, feature_names in feature_groups.items():
                features_cache[("eval", group_name)] = build_arrays(eval_rows, feature_names)
            method_scores, method_meta = score_methods(receiver, heads, features_cache, requested_methods)
            cal_y = features_cache[("calibration", "main")][1]
            cal_open = features_cache[("calibration", "main")][2]
            eval_y = features_cache[("eval", "main")][1]
            eval_open = features_cache[("eval", "main")][2]
            for method, scored_pair in method_scores.items():
                cal_scored, eval_scored = scored_pair
                thresholds = calibration_thresholds(cal_scored.risk, cal_y, cal_open, targets)
                meta = method_meta[method]
                for target, threshold in thresholds.items():
                    all_summary.append(
                        {
                            "seed": seed,
                            "severity": severity_name,
                            "open_fraction": open_fraction,
                            "method": method,
                            **meta,
                            "target_openout": target,
                            **evaluate_at_threshold(eval_scored, eval_y, eval_open, threshold),
                        }
                    )
                for row in risk_goodput_curve(eval_scored, eval_y, eval_open, args.curve_points):
                    all_curves.append(
                        {
                            "seed": seed,
                            "severity": severity_name,
                            "open_fraction": open_fraction,
                            "method": method,
                            **meta,
                            **row,
                        }
                    )
                all_diag.append(
                    {
                        "seed": seed,
                        "severity": severity_name,
                        "method": method,
                        **meta,
                        "samples": len(eval_rows),
                        "auroc": auroc(eval_scored.risk, eval_open),
                        "fpr95": fpr_at_tpr(eval_scored.risk, eval_open),
                        "known_subset_accuracy": known_subset_accuracy(eval_scored.pred, eval_y, eval_open),
                    }
                )

    write_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), all_summary)
    write_csv(output_prefix.with_name(output_prefix.name + "_curves.csv"), all_curves)
    write_csv(output_prefix.with_name(output_prefix.name + "_diagnostics.csv"), all_diag)
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(all_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_prefix.with_name(output_prefix.name + "_manifest_summary.json").write_text(
        json.dumps(manifest_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_prefix": str(output_prefix),
                "rows": len(rows),
                "summary_rows": len(all_summary),
                "curve_rows": len(all_curves),
                "features": sorted(specs),
                "main_features": main_features,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_manifest_specs(values: list[str]) -> dict[str, Path]:
    specs = {}
    for value in values:
        name, path = value.split("=", 1)
        specs[name.strip()] = Path(path).expanduser().resolve()
    if "dino" not in specs:
        raise ValueError("A dino feature manifest is required.")
    return specs


def load_feature_rows(specs: dict[str, Path]) -> tuple[list[Row], dict]:
    raw_by_feature = {}
    for name, path in specs.items():
        rows = read_manifest(path)
        mapping = {}
        for row in rows:
            if row.get("dataset") == "ag_news":
                continue
            key = row_key(row)
            source = Path(row["source_path"]).expanduser().resolve()
            if source.exists():
                mapping[key] = row
        raw_by_feature[name] = mapping
    common_keys = set.intersection(*(set(values) for values in raw_by_feature.values()))
    dino_rows = raw_by_feature["dino"]
    rows = []
    for key in sorted(common_keys):
        base = dino_rows[key]
        rows.append(
            Row(
                key=key,
                paths={name: Path(raw_by_feature[name][key]["source_path"]).expanduser().resolve() for name in raw_by_feature},
                label=int(base["label"]),
                task=base["task"],
                domain=base["domain"],
                is_unknown=parse_bool(base["is_unknown"]),
                split=base.get("split") or "eval",
                regime=base.get("regime") or "",
            )
        )
    summary = {
        "features": sorted(specs),
        "rows_by_feature": {name: len(values) for name, values in raw_by_feature.items()},
        "common_rows": len(rows),
        "regimes": count_by(rows, lambda row: row.regime),
        "splits": count_by(rows, lambda row: row.split),
    }
    return rows, summary


def build_feature_groups(specs: dict[str, Path], main_features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups = {"main": main_features, "dino": ("dino",)}
    if "siglip2" in specs:
        groups["siglip2"] = ("siglip2",)
    elif "siglip" in specs:
        groups["siglip"] = ("siglip",)
    if "openclip" in specs:
        groups["openclip"] = ("openclip",)
    return groups


def score_methods(receiver, heads, features_cache, requested_methods: set[str] | None):
    methods = {}
    meta = {}

    cal_main = features_cache[("calibration", "main")][0]
    eval_main = features_cache[("eval", "main")][0]
    cal_receiver = receiver.score(cal_main)
    eval_receiver = receiver.score(eval_main)
    if include_method("opensemcom", requested_methods):
        cal_gate = score_head(heads["main"], cal_main, "one_vs_rest")
        eval_gate = score_head(heads["main"], eval_main, "one_vs_rest")
        methods["opensemcom"] = (
            Scored(cal_receiver.pred, cal_gate.risk, cal_receiver.known_prob),
            Scored(eval_receiver.pred, eval_gate.risk, eval_receiver.known_prob),
        )
        meta["opensemcom"] = {
            "backbone": "+".join(heads["main"].feature_names),
            "detector_control": "learned_selective_risk+semantic_harq_refine",
        }
    if include_method("opensemcom_risk_head", requested_methods):
        methods["opensemcom_risk_head"] = (cal_receiver, eval_receiver)
        meta["opensemcom_risk_head"] = {
            "backbone": "+".join(heads["main"].feature_names),
            "detector_control": "learned_risk_head_only",
        }
    if include_method("opensemcom_calibrated", requested_methods):
        cal_gate = score_head(heads["main"], cal_main, "one_vs_rest")
        eval_gate = score_head(heads["main"], eval_main, "one_vs_rest")
        methods["opensemcom_calibrated"] = (
            Scored(cal_receiver.pred, cal_gate.risk, cal_receiver.known_prob),
            Scored(eval_receiver.pred, eval_gate.risk, eval_receiver.known_prob),
        )
        meta["opensemcom_calibrated"] = {
            "backbone": "+".join(heads["main"].feature_names),
            "detector_control": "mixed_open_calibration",
        }
    if include_method("opensemcom_harq_refine", requested_methods):
        cal_gate = score_head(heads["main"], cal_main, "one_vs_rest")
        eval_gate = score_head(heads["main"], eval_main, "one_vs_rest")
        cal_risk = np.minimum(cal_gate.risk, np.clip(0.75 * cal_gate.risk + 0.25 * cal_receiver.risk, 0.0, 1.0))
        eval_risk = np.minimum(eval_gate.risk, np.clip(0.75 * eval_gate.risk + 0.25 * eval_receiver.risk, 0.0, 1.0))
        methods["opensemcom_harq_refine"] = (
            Scored(cal_receiver.pred, cal_risk, cal_receiver.known_prob),
            Scored(eval_receiver.pred, eval_risk, eval_receiver.known_prob),
        )
        meta["opensemcom_harq_refine"] = {
            "backbone": "+".join(heads["main"].feature_names),
            "detector_control": "semantic_harq_refine",
        }

    dino_methods = {
        "dino_msp": "msp",
        "dino_energy": "energy",
        "dino_mahalanobis": "mahalanobis",
        "dino_vim": "vim",
        "dino_react_energy": "react_energy",
        "dino_ash_energy": "ash_energy",
        "dino_one_vs_rest": "one_vs_rest",
        "deepjscc_style": "deepjscc",
    }
    for method, detector in dino_methods.items():
        if not include_method(method, requested_methods):
            continue
        methods[method] = (
            score_head(heads["dino"], features_cache[("calibration", "dino")][0], detector),
            score_head(heads["dino"], features_cache[("eval", "dino")][0], detector),
        )
        meta[method] = {"backbone": "DINOv3", "detector_control": detector}

    for group_name, label in (("siglip2", "SigLIP2"), ("siglip", "SigLIP-base"), ("openclip", "OpenCLIP-DFN5B")):
        if group_name not in heads:
            continue
        detector = select_best_detector(heads[group_name], features_cache[("calibration", group_name)])
        method = f"{group_name}_best_ood"
        if not include_method(method, requested_methods):
            continue
        methods[method] = (
            score_head(heads[group_name], features_cache[("calibration", group_name)][0], detector),
            score_head(heads[group_name], features_cache[("eval", group_name)][0], detector),
        )
        meta[method] = {"backbone": label, "detector_control": f"best_ood:{detector}"}
    return methods, meta


def parse_methods(text: str) -> set[str] | None:
    if text.strip().lower() == "all":
        return None
    return {value.strip() for value in text.split(",") if value.strip()}


def include_method(method: str, requested_methods: set[str] | None) -> bool:
    return requested_methods is None or method in requested_methods


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("raw_source_path") or row["source_path"],
        row.get("raw_artifact_index") or row.get("artifact_index") or "",
        row.get("regime") or "",
        row.get("task") or "",
        row.get("domain") or "",
        row.get("label") or "",
        str(parse_bool(row.get("is_unknown") or "")),
    )


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


def build_arrays(rows: list[Row], feature_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    open_labels = []
    accept_labels = []
    for row in rows:
        parts = [load_feature(row.paths[name]) for name in feature_names]
        parts.append(metadata_features(row, feature_names))
        features.append(np.concatenate(parts).astype(np.float32))
        labels.append(row.label if row.known_id else KNOWN_CLASSES)
        open_labels.append(row.open_exposure)
        accept_labels.append(row.known_id)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(open_labels, dtype=bool),
        np.asarray(accept_labels, dtype=bool),
    )


def metadata_features(row: Row, feature_names: tuple[str, ...]) -> np.ndarray:
    task = one_hot_hash(f"task:{row.task}", 16)
    domain = one_hot_hash(f"domain:{row.domain}", 32)
    feature_flags = np.asarray([float(name in feature_names) for name in ("dino", "siglip2", "siglip", "openclip")], dtype=np.float32)
    channel_state = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.concatenate([task, domain, feature_flags, channel_state]).astype(np.float32)


def one_hot_hash(value: str, buckets: int) -> np.ndarray:
    out = np.zeros(buckets, dtype=np.float32)
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    out[int.from_bytes(digest[:4], "big") % buckets] = 1.0
    return out


def load_feature(path: Path) -> np.ndarray:
    x = np.load(path).reshape(-1).astype(np.float32)
    norm = np.linalg.norm(x)
    return x / max(float(norm), 1e-6)


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
        weights = np.sum(counts) / np.maximum(counts * (KNOWN_CLASSES + 1), 1.0)
        class_loss = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=self.device))
        unsafe_loss = torch.nn.BCEWithLogitsLoss()
        accept_loss = torch.nn.BCEWithLogitsLoss()
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
                    + accept_loss(accept_logits, at[idx])
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
            unsafe = torch.sigmoid(self.unsafe_head(h)).reshape(-1)
            accept = torch.sigmoid(self.accept_head(h)).reshape(-1)
        probs_np = probs.detach().cpu().numpy()
        unknown_prob = probs_np[:, KNOWN_CLASSES]
        risk = np.clip(0.45 * unsafe.detach().cpu().numpy() + 0.25 * (1.0 - accept.detach().cpu().numpy()) + 0.30 * unknown_prob, 0.0, 1.0)
        return Scored(pred=np.argmax(probs_np[:, :KNOWN_CLASSES], axis=1), risk=risk, known_prob=1.0 - unknown_prob)


def ranking_loss(logits, unsafe, torch):
    safe_scores = logits[unsafe.reshape(-1) < 0.5]
    unsafe_scores = logits[unsafe.reshape(-1) >= 0.5]
    if safe_scores.numel() == 0 or unsafe_scores.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    safe = safe_scores[: min(safe_scores.numel(), 128)]
    uns = unsafe_scores[: min(unsafe_scores.numel(), 128)]
    return torch.relu(0.2 + safe.reshape(-1, 1) - uns.reshape(1, -1)).mean()


@dataclass
class Heads:
    feature_names: tuple[str, ...]
    scaler: object
    clf: object
    unknown: object
    means: dict[int, np.ndarray]
    precision: np.ndarray
    vim_center: np.ndarray
    vim_basis: np.ndarray
    react_clip: float
    pca: object
    pca_clf: object
    train_recon_mean: float
    train_recon_std: float


def fit_heads(x: np.ndarray, y: np.ndarray, open_label: np.ndarray, feature_names: tuple[str, ...]) -> Heads:
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    known = y < KNOWN_CLASSES
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(z[known], y[known])
    unknown = LogisticRegression(max_iter=2000, class_weight="balanced")
    unknown.fit(z, open_label.astype(np.int64))
    means = {label: z[np.logical_and(known, y == label)].mean(axis=0) for label in range(KNOWN_CLASSES)}
    cov = LedoitWolf().fit(z[known])
    center = z[known].mean(axis=0)
    _, _, vt = np.linalg.svd(z[known] - center, full_matrices=False)
    rank = max(1, min(vt.shape[0] - 1, int(vt.shape[0] * 0.8)))
    pca_components = max(2, min(64, z[known].shape[0] - 1, z.shape[1]))
    pca = PCA(n_components=pca_components, random_state=0)
    pca.fit(z[known])
    recon_known = pca.inverse_transform(pca.transform(z[known]))
    pca_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    pca_clf.fit(recon_known, y[known])
    errors = np.mean((z[known] - recon_known) ** 2, axis=1)
    return Heads(
        feature_names=feature_names,
        scaler=scaler,
        clf=clf,
        unknown=unknown,
        means=means,
        precision=cov.precision_,
        vim_center=center,
        vim_basis=vt[:rank].T,
        react_clip=float(np.quantile(z[known], 0.90)),
        pca=pca,
        pca_clf=pca_clf,
        train_recon_mean=float(errors.mean()),
        train_recon_std=max(float(errors.std()), 1e-6),
    )


def score_head(heads: Heads, x: np.ndarray, detector: str) -> Scored:
    z = heads.scaler.transform(x)
    if detector == "react_energy":
        return score_logits(heads.clf, react_transform(z, heads.react_clip), "energy")
    if detector == "ash_energy":
        return score_logits(heads.clf, ash_transform(z), "energy")
    if detector == "deepjscc":
        recon = heads.pca.inverse_transform(heads.pca.transform(z))
        logits = logits_from_clf(heads.pca_clf, recon)
        probs = softmax(logits)
        errors = np.mean((z - recon) ** 2, axis=1)
        risk = scale01((errors - heads.train_recon_mean) / heads.train_recon_std)
        return Scored(pred=np.argmax(probs, axis=1), risk=risk, known_prob=1.0 - risk)
    if detector in {"msp", "energy"}:
        return score_logits(heads.clf, z, detector)
    pred = heads.clf.predict(z)
    if detector == "one_vs_rest":
        probs = heads.unknown.predict_proba(z)
        classes = list(heads.unknown.classes_)
        risk = probs[:, classes.index(1)] if 1 in classes else np.zeros(z.shape[0])
        return Scored(pred=pred, risk=risk, known_prob=1.0 - risk)
    if detector == "mahalanobis":
        values = []
        for sample in z:
            dists = []
            for mean in heads.means.values():
                delta = sample - mean
                dists.append(float(delta @ heads.precision @ delta.T))
            values.append(math.sqrt(max(min(dists), 0.0)))
        risk = scale01(np.asarray(values))
        return Scored(pred=pred, risk=risk, known_prob=1.0 - risk)
    if detector == "vim":
        residuals = []
        for sample in z:
            delta = sample - heads.vim_center
            projected = heads.vim_basis @ (heads.vim_basis.T @ delta)
            residuals.append(np.linalg.norm(delta - projected))
        risk = scale01(np.asarray(residuals))
        return Scored(pred=pred, risk=risk, known_prob=1.0 - risk)
    raise ValueError(f"Unknown detector: {detector}")


def score_logits(clf, z: np.ndarray, detector: str) -> Scored:
    logits = logits_from_clf(clf, z)
    probs = softmax(logits)
    pred = np.argmax(probs, axis=1)
    if detector == "msp":
        risk = 1.0 - np.max(probs, axis=1)
    elif detector == "energy":
        risk = -logsumexp(logits, axis=1)
    else:
        raise ValueError(detector)
    return Scored(pred=pred, risk=risk, known_prob=1.0 - np.clip(risk, 0.0, 1.0))


def select_best_detector(heads: Heads, cal_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> str:
    x, y, open_label, _ = cal_arrays
    candidates = ("mahalanobis", "vim", "one_vs_rest")
    best = ("mahalanobis", -1.0, 1.0)
    for detector in candidates:
        scored = score_head(heads, x, detector)
        feasible = [row for row in risk_goodput_curve(scored, y, open_label, 101) if row["accepted_open_outage"] <= 0.05]
        if not feasible:
            continue
        row = max(feasible, key=lambda item: item["semantic_goodput"])
        candidate = (detector, row["semantic_goodput"], -row["accepted_open_outage"])
        if candidate[1:] > best[1:]:
            best = candidate
    return best[0]


def logits_from_clf(clf, z: np.ndarray) -> np.ndarray:
    logits = clf.decision_function(z)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    return logits.astype(np.float64)


def react_transform(z: np.ndarray, clip_value: float) -> np.ndarray:
    return np.minimum(z, clip_value)


def ash_transform(z: np.ndarray, keep_ratio: float = 0.20) -> np.ndarray:
    if z.shape[1] <= 1:
        return z
    k = max(1, int(z.shape[1] * keep_ratio))
    abs_z = np.abs(z)
    threshold = np.partition(abs_z, -k, axis=1)[:, -k][:, None]
    masked = np.where(abs_z >= threshold, z, 0.0)
    original = np.sum(abs_z, axis=1, keepdims=True)
    kept = np.sum(np.abs(masked), axis=1, keepdims=True)
    return masked * (original / np.maximum(kept, 1e-9))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_v = np.max(values, axis=axis, keepdims=True)
    return (max_v + np.log(np.sum(np.exp(values - max_v), axis=axis, keepdims=True))).reshape(-1)


def calibration_thresholds(risk: np.ndarray, y: np.ndarray, open_label: np.ndarray, targets: tuple[float, ...]) -> dict[str, float]:
    thresholds = {}
    for target in targets:
        safe = []
        for threshold in np.quantile(risk, np.linspace(0.0, 1.0, 101)):
            selected = risk <= threshold
            if not np.any(selected):
                continue
            unsafe = np.logical_or(open_label[selected], y[selected] >= KNOWN_CLASSES)
            if float(np.mean(unsafe)) <= target:
                safe.append(float(threshold))
        thresholds[f"openout<={target:.2f}"] = max(safe) if safe else float(np.min(risk) - 1e-6)
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
    return [evaluate_at_threshold(scored, y, open_label, float(threshold)) for threshold in np.quantile(scored.risk, np.linspace(0.0, 1.0, points))]


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


def count_by(rows: list[Row], fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(fn(row))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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

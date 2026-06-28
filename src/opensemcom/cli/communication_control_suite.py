
"""Focused communication-control experiments for OpenSemCom.

This suite tests whether progressive semantic refinement and resource-aware
accept/refine/reject control improve safety-constrained useful goodput. It uses
existing feature manifests and measured DeepSense 6G metadata.
"""

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
DEEPSENSE_BEAM_SECTORS = 8
TRAIN_TASKS = {"classification"}
TRAIN_DOMAINS = {"cifar10"}


@dataclass(frozen=True)
class Row:
    key: tuple[str, str]
    paths: dict[str, Path]
    raw_source_path: str
    dataset: str
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


@dataclass
class SplitData:
    train: list[Row]
    calibration: list[Row]
    eval: list[Row]


@dataclass
class ChannelSample:
    max_power: float
    mean_power: float
    pdop: float
    hdop: float
    num_sat: float
    fix_3d: float


@dataclass
class ChannelContext:
    by_camera: dict[str, ChannelSample]
    samples: list[ChannelSample]
    mins: np.ndarray
    spans: np.ndarray

    def vector_for(self, row: Row) -> np.ndarray:
        sample = self.by_camera.get(Path(row.raw_source_path).name)
        matched = 1.0 if sample is not None else 0.0
        if sample is None:
            digest = hashlib.sha256("|".join(row.key).encode("utf-8")).digest()
            sample = self.samples[int.from_bytes(digest[:4], "big") % len(self.samples)]
        raw = np.asarray([sample.max_power, sample.mean_power, sample.pdop, sample.hdop, sample.num_sat, sample.fix_3d], dtype=np.float32)
        return np.concatenate([np.clip((raw - self.mins) / self.spans, 0.0, 1.0), np.asarray([matched], dtype=np.float32)]).astype(np.float32)

    def summary(self) -> dict[str, int]:
        return {"samples": len(self.samples), "camera_indexed_samples": len(self.by_camera)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run focused OpenSemCom communication-control experiments.")
    p.add_argument("--feature-manifest", action="append", required=True, help="name=manifest.csv")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--deepsense-scenario-root", default="data/deepsense6g/Scenario1")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--targets", default="0.05")
    p.add_argument("--resource-budgets", default="0.60,0.80,1.00")
    p.add_argument("--eval-size", type=int, default=1024)
    p.add_argument("--train-known-per-class", type=int, default=192)
    p.add_argument("--train-open", type=int, default=1024)
    p.add_argument("--cal-known-per-class", type=int, default=64)
    p.add_argument("--cal-open", type=int, default=768)
    return p


def main() -> None:
    args = build_parser().parse_args()
    specs = parse_manifest_specs(args.feature_manifest)
    rows, manifest_summary = load_rows(specs)
    channel = load_channel_context(Path(args.deepsense_scenario_root))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    budgets = [float(x) for x in args.resource_budgets.split(",") if x.strip()]
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    policy_rows: list[dict] = []
    for seed in seeds:
        full_split = make_fullopen_split(rows, seed, args)
        summary_rows.extend(run_task("full-open", full_split, specs, channel, targets, budgets, seed, policy_rows))
        beam_split = make_deepsense_split(rows, seed)
        summary_rows.extend(run_task("deepsense-beam", beam_split, specs, channel, targets, budgets, seed, policy_rows))

    write_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), summary_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_policies.csv"), policy_rows)
    output_prefix.with_name(output_prefix.name + "_manifest_summary.json").write_text(
        json.dumps({**manifest_summary, "channel_context": channel.summary()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary_rows": len(summary_rows), "policy_rows": len(policy_rows), "channel_context": channel.summary()}, indent=2, sort_keys=True))


def run_task(task_name: str, split: SplitData, specs: dict[str, Path], channel: ChannelContext, targets: list[float], budgets: list[float], seed: int, policy_rows: list[dict]) -> list[dict]:
    train_y, train_open = labels_for(task_name, split.train)
    cal_y, cal_open = labels_for(task_name, split.calibration)
    eval_y, eval_open = labels_for(task_name, split.eval)
    n_classes = int(max(np.max(train_y), np.max(cal_y), np.max(eval_y))) + 1

    arrays = {
        "dino": (make_features(split.train, ("dino",), channel, False), make_features(split.calibration, ("dino",), channel, False), make_features(split.eval, ("dino",), channel, False)),
        "ensemble": (make_features(split.train, tuple(specs), channel, False), make_features(split.calibration, tuple(specs), channel, False), make_features(split.eval, tuple(specs), channel, False)),
        "ensemble_channel": (make_features(split.train, tuple(specs), channel, True), make_features(split.calibration, tuple(specs), channel, True), make_features(split.eval, tuple(specs), channel, True)),
    }

    dino = fit_model(arrays["dino"][0], train_y, train_open, n_classes, detector_open=(task_name == "full-open"))
    ens = fit_model(arrays["ensemble"][0], train_y, train_open, n_classes, detector_open=(task_name == "full-open"))
    ens_ch = fit_model(arrays["ensemble_channel"][0], train_y, train_open, n_classes, detector_open=(task_name == "full-open"))
    jscc = fit_deepjscc(arrays["dino"][0], train_y, train_open, n_classes, detector_open=(task_name == "full-open"))
    receiver_ch = fit_receiver(arrays["ensemble_channel"][0], train_y, train_open, n_classes, has_open_class=(task_name == "full-open"), seed=seed)
    receiver_ens = fit_receiver(arrays["ensemble"][0], train_y, train_open, n_classes, has_open_class=(task_name == "full-open"), seed=seed + 1000)

    dino_cal = score_model(dino, arrays["dino"][1])
    dino_eval = score_model(dino, arrays["dino"][2])
    ens_cal = score_model(ens, arrays["ensemble"][1])
    ens_eval = score_model(ens, arrays["ensemble"][2])
    ens_ch_cal = score_model(ens_ch, arrays["ensemble_channel"][1])
    ens_ch_eval = score_model(ens_ch, arrays["ensemble_channel"][2])
    jscc_cal = score_deepjscc(jscc, arrays["dino"][1])
    jscc_eval = score_deepjscc(jscc, arrays["dino"][2])
    recv_ch_cal = score_receiver(receiver_ch, arrays["ensemble_channel"][1])
    recv_ch_eval = score_receiver(receiver_ch, arrays["ensemble_channel"][2])
    recv_ens_cal = score_receiver(receiver_ens, arrays["ensemble"][1])
    recv_ens_eval = score_receiver(receiver_ens, arrays["ensemble"][2])

    methods = {
        "dino_detector": (dino_cal, dino_eval, 1.0),
        "ensemble_detector": (ens_cal, ens_eval, 1.2),
        "deepjscc_pca": (jscc_cal, jscc_eval, 0.7),
        "witt_context_style": (ens_ch_cal, ens_ch_eval, 1.4),
        "fixed_refine_all": (ens_ch_cal, ens_ch_eval, 1.6),
        "opensemcom_receiver_only": (recv_ch_cal, recv_ch_eval, 1.2),
        "opensemcom_no_channel": (recv_ens_cal, recv_ens_eval, 1.2),
    }
    dino_channel_cal = fuse_scores(dino_cal, ens_ch_cal, disagreement_penalty=0.04)
    dino_channel_eval = fuse_scores(dino_eval, ens_ch_eval, disagreement_penalty=0.04)
    ensemble_channel_cal = fuse_scores(ens_cal, ens_ch_cal, disagreement_penalty=0.04)
    ensemble_channel_eval = fuse_scores(ens_eval, ens_ch_eval, disagreement_penalty=0.04)
    receiver_channel_cal = fuse_scores(recv_ch_cal, ens_ch_cal, disagreement_penalty=0.03)
    receiver_channel_eval = fuse_scores(recv_ch_eval, ens_ch_eval, disagreement_penalty=0.03)
    receiver_dino_cal = fuse_scores(recv_ch_cal, dino_channel_cal, disagreement_penalty=0.03)
    receiver_dino_eval = fuse_scores(recv_ch_eval, dino_channel_eval, disagreement_penalty=0.03)
    progressive_candidates = {
        "dino_core": (dino_cal, dino_eval, ens_ch_cal, ens_ch_eval),
        "ensemble_core": (ens_cal, ens_eval, ens_ch_cal, ens_ch_eval),
        "dino_channel_fusion_core": (dino_channel_cal, dino_channel_eval, ens_ch_cal, ens_ch_eval),
        "ensemble_channel_fusion_core": (ensemble_channel_cal, ensemble_channel_eval, ens_ch_cal, ens_ch_eval),
        "trained_receiver_core": (recv_ch_cal, recv_ch_eval, ens_ch_cal, ens_ch_eval),
        "trained_receiver_channel_fusion_core": (receiver_channel_cal, receiver_channel_eval, ens_ch_cal, ens_ch_eval),
        "trained_receiver_dino_fusion_core": (receiver_dino_cal, receiver_dino_eval, ens_ch_cal, ens_ch_eval),
        "trained_receiver_no_channel_core": (recv_ens_cal, recv_ens_eval, ens_ch_cal, ens_ch_eval),
    }

    rows: list[dict] = []
    for target in targets:
        for budget in budgets:
            for name, (cal_score, eval_score, accept_cost) in methods.items():
                policy = select_single_policy(cal_score, cal_y, cal_open, target, budget, accept_cost)
                policy_rows.append({"task": task_name, "method": name, "seed": seed, "target_openout": target, "resource_budget": budget, **policy})
                rows.append({"task": task_name, "method": name, "seed": seed, "target_openout": target, "resource_budget": budget, **eval_single(eval_score, eval_y, eval_open, policy["threshold"], accept_cost)})
            policy = select_best_progressive_policy(progressive_candidates, cal_y, cal_open, target, budget)
            route = policy["route"]
            _, core_eval, _, refine_eval = progressive_candidates[route]
            policy_rows.append({"task": task_name, "method": "opensemcom_progressive", "seed": seed, "target_openout": target, "resource_budget": budget, **policy})
            rows.append({"task": task_name, "method": "opensemcom_progressive", "seed": seed, "target_openout": target, "resource_budget": budget, **eval_progressive(core_eval, refine_eval, eval_y, eval_open, policy)})
    return rows


def parse_manifest_specs(values: list[str]) -> dict[str, Path]:
    specs = {}
    for value in values:
        name, path = value.split("=", 1)
        specs[name.strip()] = Path(path).expanduser().resolve()
    if "dino" not in specs:
        raise ValueError("dino manifest required")
    return specs


def load_rows(specs: dict[str, Path]) -> tuple[list[Row], dict]:
    raw_by_feature = {}
    for name, path in specs.items():
        mapping = {}
        for row in read_manifest(path):
            if row.get("dataset") == "ag_news":
                continue
            source = Path(row["source_path"]).expanduser().resolve()
            if source.exists():
                mapping[row_key(row)] = row
        raw_by_feature[name] = mapping
    common = set.intersection(*(set(v) for v in raw_by_feature.values()))
    rows = []
    for key in sorted(common):
        base = raw_by_feature["dino"][key]
        rows.append(Row(
            key=key,
            paths={name: Path(raw_by_feature[name][key]["source_path"]).expanduser().resolve() for name in raw_by_feature},
            raw_source_path=base.get("raw_source_path") or base["source_path"],
            dataset=base.get("dataset") or "",
            label=int(base["label"]),
            task=base["task"],
            domain=base["domain"],
            is_unknown=parse_bool(base["is_unknown"]),
            split=base.get("split") or "eval",
            regime=base.get("regime") or "",
        ))
    summary = {
        "features": sorted(specs),
        "common_rows": len(rows),
        "datasets": count_by(rows, lambda r: r.dataset),
        "tasks": count_by(rows, lambda r: r.task),
        "regimes": count_by(rows, lambda r: r.regime),
    }
    return rows, summary


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


def make_fullopen_split(rows: list[Row], seed: int, args) -> SplitData:
    rng = np.random.default_rng(seed)
    known_all = [r for r in rows if r.known_id]
    open_all = [r for r in rows if r.regime == "full-open" and r.open_exposure]
    by_class = {label: [] for label in range(KNOWN_CLASSES)}
    for row in known_all:
        by_class[row.label].append(row)
    train, cal, eval_known = [], [], []
    for label, values in by_class.items():
        values = shuffled(values, rng)
        train += values[: args.train_known_per_class]
        cal += values[args.train_known_per_class : args.train_known_per_class + args.cal_known_per_class]
        eval_known += values[args.train_known_per_class + args.cal_known_per_class :]
    open_values = shuffled(open_all, rng)
    train += open_values[: args.train_open]
    cal += open_values[args.train_open : args.train_open + args.cal_open]
    eval_open = open_values[args.train_open + args.cal_open :]
    eval_rows = shuffled(eval_known, rng)[: args.eval_size // 2] + shuffled(eval_open, rng)[: args.eval_size // 2]
    return SplitData(shuffled(train, rng), shuffled(cal, rng), shuffled(eval_rows, rng))


def make_deepsense_split(rows: list[Row], seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    ds = [r for r in rows if r.dataset == "deepsense6g" and r.task == "beam-prediction"]
    if len(ds) < 100:
        raise ValueError("Not enough DeepSense rows for beam task")
    raw_labels = sorted({r.label for r in ds})
    label_rank = {label: idx for idx, label in enumerate(raw_labels)}
    remapped = []
    for r in ds:
        sector = min(DEEPSENSE_BEAM_SECTORS - 1, int(label_rank[r.label] * DEEPSENSE_BEAM_SECTORS / max(len(raw_labels), 1)))
        remapped.append(Row(r.key, r.paths, r.raw_source_path, r.dataset, sector, r.task, r.domain, False, r.split, r.regime))
    by_label: dict[int, list[Row]] = {}
    for row in remapped:
        by_label.setdefault(row.label, []).append(row)
    train, cal, ev = [], [], []
    for values in by_label.values():
        values = shuffled(values, rng)
        n = len(values)
        train += values[: max(1, int(0.50 * n))]
        cal += values[max(1, int(0.50 * n)) : max(2, int(0.75 * n))]
        ev += values[max(2, int(0.75 * n)) :]
    return SplitData(shuffled(train, rng), shuffled(cal, rng), shuffled(ev, rng))


def labels_for(task_name: str, rows: list[Row]) -> tuple[np.ndarray, np.ndarray]:
    if task_name == "full-open":
        y = np.asarray([r.label if r.known_id else KNOWN_CLASSES for r in rows], dtype=np.int64)
        open_label = np.asarray([r.open_exposure for r in rows], dtype=bool)
    else:
        y = np.asarray([r.label for r in rows], dtype=np.int64)
        open_label = np.zeros(len(rows), dtype=bool)
    return y, open_label


def make_features(rows: list[Row], feature_names: tuple[str, ...], channel: ChannelContext, include_channel: bool) -> np.ndarray:
    values = []
    for row in rows:
        parts = [load_feature(row.paths[name]) for name in feature_names]
        if include_channel:
            parts.append(channel.vector_for(row))
        values.append(np.concatenate(parts).astype(np.float32))
    return np.asarray(values, dtype=np.float32)


def load_feature(path: Path) -> np.ndarray:
    x = np.load(path).reshape(-1).astype(np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-6)


def load_channel_context(root: Path) -> ChannelContext:
    root = root.expanduser().resolve()
    rows = read_csv(root / "scenario1.csv")
    samples: list[ChannelSample] = []
    by_camera = {}
    for raw in rows:
        mmwave = (root / raw.get("unit1_pwr_60ghz", "").strip()).resolve()
        stats = read_mmwave_stats(mmwave) if mmwave.exists() else None
        if stats is None:
            continue
        sample = ChannelSample(
            max_power=stats["max_power"],
            mean_power=stats["mean_power"],
            pdop=parse_float(raw.get("unit2_PDOP"), 0.0),
            hdop=parse_float(raw.get("unit2_HDOP"), 0.0),
            num_sat=parse_float(raw.get("unit2_num_sat"), 0.0),
            fix_3d=1.0 if str(raw.get("unit2_fix_type", "")).strip().upper() == "3D" else 0.0,
        )
        samples.append(sample)
        by_camera[Path(raw.get("unit1_rgb", "").strip()).name] = sample
    if not samples:
        raise ValueError("No DeepSense channel samples found")
    matrix = np.asarray([[s.max_power, s.mean_power, s.pdop, s.hdop, s.num_sat, s.fix_3d] for s in samples], dtype=np.float32)
    mins = matrix.min(axis=0)
    spans = np.maximum(matrix.max(axis=0) - mins, 1e-6)
    return ChannelContext(by_camera=by_camera, samples=samples, mins=mins, spans=spans)


def read_mmwave_stats(path: Path) -> dict[str, float] | None:
    vals = []
    for tok in path.read_text(encoding="utf-8", errors="replace").replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {"max_power": float(arr.max()), "mean_power": float(arr.mean())}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fit_model(x: np.ndarray, y: np.ndarray, open_label: np.ndarray, n_classes: int, detector_open: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    known = y < n_classes
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(z[known], y[known])
    detector = None
    if detector_open and len(set(open_label.astype(int))) > 1:
        detector = LogisticRegression(max_iter=2000, class_weight="balanced")
        detector.fit(z, open_label.astype(np.int64))
    return {"scaler": scaler, "clf": clf, "detector": detector}


def score_model(model, x: np.ndarray) -> Scored:
    z = model["scaler"].transform(x)
    probs = model["clf"].predict_proba(z)
    pred = model["clf"].classes_[np.argmax(probs, axis=1)]
    msp = 1.0 - np.max(probs, axis=1)
    if model["detector"] is not None:
        det_probs = model["detector"].predict_proba(z)
        classes = list(model["detector"].classes_)
        open_prob = det_probs[:, classes.index(1)] if 1 in classes else np.zeros(z.shape[0])
        risk = np.clip(0.65 * open_prob + 0.35 * msp, 0.0, 1.0)
    else:
        risk = msp
    return Scored(pred=pred.astype(np.int64), risk=risk.astype(np.float64))


def fuse_scores(left: Scored, right: Scored, disagreement_penalty: float) -> Scored:
    choose_left = left.risk <= right.risk
    pred = np.where(choose_left, left.pred, right.pred)
    risk = np.minimum(left.risk, right.risk)
    risk = risk + disagreement_penalty * (left.pred != right.pred)
    return Scored(pred=pred.astype(np.int64), risk=np.clip(risk, 0.0, 1.0).astype(np.float64))


class TrainedReceiver:
    def __init__(self, input_dim: int, n_classes: int, has_open_class: bool, seed: int, hidden_dim: int = 256, epochs: int = 45, lr: float = 1e-3):
        import torch

        self.torch = torch
        self.n_classes = int(n_classes)
        self.has_open_class = bool(has_open_class)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.12),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.08),
        ).to(self.device)
        self.class_head = torch.nn.Linear(hidden_dim, self.n_classes).to(self.device)
        self.unsafe_head = torch.nn.Linear(hidden_dim, 1).to(self.device)
        self.accept_head = torch.nn.Linear(hidden_dim, 1).to(self.device)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray, y: np.ndarray, open_label: np.ndarray) -> None:
        torch = self.torch
        x_np = np.asarray(x, dtype=np.float32)
        self.mean = x_np.mean(axis=0, keepdims=True)
        self.std = np.maximum(x_np.std(axis=0, keepdims=True), 1e-6)
        x_np = (x_np - self.mean) / self.std
        open_class = self.n_classes - 1 if self.has_open_class else -1
        unsafe = np.logical_or(open_label, y == open_class).astype(np.float32)
        accept = (unsafe < 0.5).astype(np.float32)
        xt = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        ut = torch.as_tensor(unsafe.reshape(-1, 1), dtype=torch.float32, device=self.device)
        at = torch.as_tensor(accept.reshape(-1, 1), dtype=torch.float32, device=self.device)
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float32)
        class_weights = np.sum(counts) / np.maximum(counts * self.n_classes, 1.0)
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
        generator.manual_seed(2031)
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
                    + 1.35 * unsafe_loss(unsafe_logits, ut[idx])
                    + 0.90 * accept_loss(accept_logits, at[idx])
                    + ranking_loss(unsafe_logits, ut[idx], torch)
                    - 0.04 * torch.mean(torch.sigmoid(accept_logits) * at[idx])
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.model.eval()

    def score(self, x: np.ndarray) -> Scored:
        torch = self.torch
        x_np = (np.asarray(x, dtype=np.float32) - self.mean) / self.std
        with torch.inference_mode():
            xt = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
            h = self.model(xt)
            probs = torch.softmax(self.class_head(h), dim=-1)
            unsafe_prob = torch.sigmoid(self.unsafe_head(h)).reshape(-1)
            accept_prob = torch.sigmoid(self.accept_head(h)).reshape(-1)
        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        if self.has_open_class:
            pred = np.argmax(probs_np[:, : self.n_classes - 1], axis=1)
            unknown_prob = probs_np[:, self.n_classes - 1]
        else:
            pred = np.argmax(probs_np, axis=1)
            unknown_prob = np.zeros(probs_np.shape[0], dtype=np.float64)
        entropy = -np.sum(probs_np * np.log(np.maximum(probs_np, 1e-12)), axis=1) / math.log(max(self.n_classes, 2))
        unsafe_np = unsafe_prob.detach().cpu().numpy().astype(np.float64)
        accept_np = accept_prob.detach().cpu().numpy().astype(np.float64)
        if self.has_open_class:
            risk = 0.40 * unsafe_np + 0.25 * (1.0 - accept_np) + 0.25 * unknown_prob + 0.10 * entropy
        else:
            risk = 0.45 * unsafe_np + 0.35 * entropy + 0.20 * (1.0 - accept_np)
        return Scored(pred=pred.astype(np.int64), risk=np.clip(risk, 0.0, 1.0).astype(np.float64))


def fit_receiver(x: np.ndarray, y: np.ndarray, open_label: np.ndarray, n_classes: int, has_open_class: bool, seed: int) -> TrainedReceiver:
    receiver = TrainedReceiver(input_dim=x.shape[1], n_classes=n_classes, has_open_class=has_open_class, seed=seed)
    receiver.fit(x, y, open_label)
    return receiver


def score_receiver(receiver: TrainedReceiver, x: np.ndarray) -> Scored:
    return receiver.score(x)


def ranking_loss(logits, unsafe, torch):
    safe_scores = logits[unsafe.reshape(-1) < 0.5]
    unsafe_scores = logits[unsafe.reshape(-1) >= 0.5]
    if safe_scores.numel() == 0 or unsafe_scores.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    safe = safe_scores[: min(safe_scores.numel(), 128)]
    uns = unsafe_scores[: min(unsafe_scores.numel(), 128)]
    return torch.relu(0.2 + safe.reshape(-1, 1) - uns.reshape(1, -1)).mean()


def fit_deepjscc(x: np.ndarray, y: np.ndarray, open_label: np.ndarray, n_classes: int, detector_open: bool):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    n_comp = max(2, min(32, z.shape[0] - 1, z.shape[1]))
    pca = PCA(n_components=n_comp, random_state=0)
    low = pca.fit_transform(z)
    rec = pca.inverse_transform(low)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(low, y)
    err = np.mean((z - rec) ** 2, axis=1)
    return {"scaler": scaler, "pca": pca, "clf": clf, "err_mean": float(err.mean()), "err_std": max(float(err.std()), 1e-6)}


def score_deepjscc(model, x: np.ndarray) -> Scored:
    z = model["scaler"].transform(x)
    low = model["pca"].transform(z)
    rec = model["pca"].inverse_transform(low)
    probs = model["clf"].predict_proba(low)
    pred = model["clf"].classes_[np.argmax(probs, axis=1)]
    msp = 1.0 - np.max(probs, axis=1)
    err = np.mean((z - rec) ** 2, axis=1)
    recon_risk = scale01((err - model["err_mean"]) / model["err_std"])
    return Scored(pred=pred.astype(np.int64), risk=np.clip(0.5 * msp + 0.5 * recon_risk, 0.0, 1.0))


def select_single_policy(cal: Scored, y: np.ndarray, open_label: np.ndarray, target: float, budget: float, accept_cost: float) -> dict:
    best = None
    for threshold in candidate_thresholds(cal.risk):
        metrics = eval_single(cal, y, open_label, threshold, accept_cost, include_detection=False)
        upper = wilson_upper(metrics["accepted_unsafe"], metrics["accepted"])
        if upper <= target and metrics["resource_per_sample"] <= budget:
            score = (metrics["semantic_goodput"], metrics["goodput_per_resource"], -upper)
            if best is None or score > best[0]:
                best = (score, threshold, metrics, upper)
    if best is None:
        threshold = float(np.min(cal.risk) - 1e-6)
        metrics = eval_single(cal, y, open_label, threshold, accept_cost, include_detection=False)
        upper = wilson_upper(metrics["accepted_unsafe"], metrics["accepted"])
        return {"threshold": threshold, "cal_goodput": metrics["semantic_goodput"], "cal_openout": metrics["accepted_open_outage"], "cal_openout_upper": upper}
    return {"threshold": float(best[1]), "cal_goodput": best[2]["semantic_goodput"], "cal_openout": best[2]["accepted_open_outage"], "cal_openout_upper": best[3]}


def eval_single(scored: Scored, y: np.ndarray, open_label: np.ndarray, threshold: float, accept_cost: float, include_detection: bool = True) -> dict:
    selected = scored.risk <= threshold
    rejected = ~selected
    unsafe = np.logical_or(open_label, scored.pred != y)
    correct = ~unsafe
    accepted_correct = np.logical_and(selected, correct)
    accepted_unsafe = np.logical_and(selected, unsafe)
    resource = float(accept_cost * np.sum(selected) + 0.1 * np.sum(rejected))
    return metrics_dict(selected, np.zeros_like(selected, dtype=bool), rejected, accepted_correct, accepted_unsafe, y, scored.pred, scored.risk, unsafe, resource, include_detection)


def select_best_progressive_policy(candidates: dict[str, tuple[Scored, Scored, Scored, Scored]], y: np.ndarray, open_label: np.ndarray, target: float, budget: float) -> dict:
    best = None
    for route, (core_cal, _core_eval, refine_cal, _refine_eval) in candidates.items():
        policy = select_progressive_policy(core_cal, refine_cal, y, open_label, target, budget)
        score = (policy["cal_goodput"], policy.get("cal_goodput_per_resource", 0.0), -policy.get("cal_openout_upper", 0.0))
        if best is None or score > best[0]:
            best = (score, route, policy)
    if best is None:
        raise ValueError("No progressive candidate policies were evaluated")
    return {"route": best[1], **best[2]}


def select_progressive_policy(core: Scored, refine: Scored, y: np.ndarray, open_label: np.ndarray, target: float, budget: float) -> dict:
    thresholds = candidate_thresholds(np.concatenate([core.risk, refine.risk]))
    best = None
    for q1 in thresholds:
        for q2 in thresholds:
            if q2 < q1:
                continue
            for qr in thresholds:
                metrics = eval_progressive(core, refine, y, open_label, {"q1": q1, "q2": q2, "qr": qr}, include_detection=False)
                upper = wilson_upper(metrics["accepted_unsafe"], metrics["accepted"])
                if upper <= target and metrics["resource_per_sample"] <= budget:
                    score = (metrics["semantic_goodput"], metrics["goodput_per_resource"], -upper)
                    if best is None or score > best[0]:
                        best = (score, q1, q2, qr, metrics, upper)
    if best is None:
        return {"q1": float(np.min(core.risk) - 1e-6), "q2": float(np.min(core.risk) - 1e-6), "qr": float(np.min(refine.risk) - 1e-6), "cal_goodput": 0.0, "cal_openout": 0.0, "cal_openout_upper": 0.0, "cal_goodput_per_resource": 0.0, "cal_resource_per_sample": 0.0, "cal_refine_rate": 0.0}
    return {"q1": float(best[1]), "q2": float(best[2]), "qr": float(best[3]), "cal_goodput": best[4]["semantic_goodput"], "cal_openout": best[4]["accepted_open_outage"], "cal_openout_upper": best[5], "cal_goodput_per_resource": best[4]["goodput_per_resource"], "cal_resource_per_sample": best[4]["resource_per_sample"], "cal_refine_rate": best[4]["refine_rate"]}


def eval_progressive(core: Scored, refine: Scored, y: np.ndarray, open_label: np.ndarray, policy: dict, include_detection: bool = True) -> dict:
    q1, q2, qr = policy["q1"], policy["q2"], policy["qr"]
    core_accept = core.risk <= q1
    refine_mask = np.logical_and(core.risk > q1, core.risk <= q2)
    refine_accept = np.logical_and(refine_mask, refine.risk <= qr)
    selected = np.logical_or(core_accept, refine_accept)
    rejected = ~selected
    resource_rejected = ~np.logical_or(core_accept, refine_mask)
    final_pred = np.where(refine_accept, refine.pred, core.pred)
    unsafe = np.logical_or(open_label, final_pred != y)
    correct = ~unsafe
    accepted_correct = np.logical_and(selected, correct)
    accepted_unsafe = np.logical_and(selected, unsafe)
    final_risk = np.where(refine_mask, refine.risk, core.risk)
    resource = float(np.sum(core_accept) + 1.6 * np.sum(refine_mask) + 0.1 * np.sum(resource_rejected))
    return metrics_dict(selected, refine_mask, rejected, accepted_correct, accepted_unsafe, y, final_pred, final_risk, unsafe, resource, include_detection)


def metrics_dict(selected, refined, rejected, accepted_correct, accepted_unsafe, y, pred, risk, unsafe, resource: float, include_detection: bool) -> dict:
    n = len(y)
    accepted = int(np.sum(selected))
    metrics = {
        "semantic_goodput": float(np.sum(accepted_correct) / max(n, 1)),
        "coverage": float(np.mean(selected)) if n else 0.0,
        "accepted_known_accuracy": float(np.sum(accepted_correct) / max(accepted, 1)),
        "accepted_open_outage": float(np.sum(accepted_unsafe) / max(accepted, 1)),
        "accepted": accepted,
        "accepted_correct": int(np.sum(accepted_correct)),
        "accepted_unsafe": int(np.sum(accepted_unsafe)),
        "refine_rate": float(np.mean(refined)) if n else 0.0,
        "reject_rate": float(np.mean(rejected)) if n else 0.0,
        "resource_units": resource,
        "resource_per_sample": float(resource / max(n, 1)),
        "goodput_per_resource": float(np.sum(accepted_correct) / max(resource, 1e-9)),
        "accuracy": float(np.mean(pred == y)) if n else 0.0,
    }
    if include_detection:
        metrics["auroc"] = auroc(np.asarray(risk, dtype=np.float64), np.asarray(unsafe, dtype=bool))
        metrics["fpr95"] = fpr_at_tpr(np.asarray(risk, dtype=np.float64), np.asarray(unsafe, dtype=bool))
    return metrics


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p_score in pos:
        wins += float(np.sum(p_score > neg)) + 0.5 * float(np.sum(p_score == neg))
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


def wilson_upper(errors: int, total: int, z: float = 1.64) -> float:
    if total <= 0:
        return 0.0
    phat = errors / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return float(min(1.0, (centre + margin) / denom))


def candidate_thresholds(risk: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(risk, np.linspace(0.0, 1.0, 25)))


def scale01(values: np.ndarray) -> np.ndarray:
    lo = float(np.quantile(values, 0.05))
    hi = float(np.quantile(values, 0.95))
    return np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def shuffled(values: list[Row], rng: np.random.Generator) -> list[Row]:
    values = list(values)
    if values:
        order = rng.permutation(len(values))
        values = [values[int(i)] for i in order]
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
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

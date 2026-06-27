
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

    methods = {
        "dino_detector": (score_model(dino, arrays["dino"][1]), score_model(dino, arrays["dino"][2]), 1.0),
        "ensemble_detector": (score_model(ens, arrays["ensemble"][1]), score_model(ens, arrays["ensemble"][2]), 1.2),
        "deepjscc_pca": (score_deepjscc(jscc, arrays["dino"][1]), score_deepjscc(jscc, arrays["dino"][2]), 0.7),
        "witt_context_style": (score_model(ens_ch, arrays["ensemble_channel"][1]), score_model(ens_ch, arrays["ensemble_channel"][2]), 1.4),
        "fixed_refine_all": (score_model(ens_ch, arrays["ensemble_channel"][1]), score_model(ens_ch, arrays["ensemble_channel"][2]), 1.6),
    }
    core_cal = score_model(dino, arrays["dino"][1])
    core_eval = score_model(dino, arrays["dino"][2])
    refine_cal = score_model(ens_ch, arrays["ensemble_channel"][1])
    refine_eval = score_model(ens_ch, arrays["ensemble_channel"][2])

    rows: list[dict] = []
    for target in targets:
        for budget in budgets:
            for name, (cal_score, eval_score, accept_cost) in methods.items():
                policy = select_single_policy(cal_score, cal_y, cal_open, target, budget, accept_cost)
                policy_rows.append({"task": task_name, "method": name, "seed": seed, "target_openout": target, "resource_budget": budget, **policy})
                rows.append({"task": task_name, "method": name, "seed": seed, "target_openout": target, "resource_budget": budget, **eval_single(eval_score, eval_y, eval_open, policy["threshold"], accept_cost)})
            policy = select_progressive_policy(core_cal, refine_cal, cal_y, cal_open, target, budget)
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
        metrics = eval_single(cal, y, open_label, threshold, accept_cost)
        upper = wilson_upper(metrics["accepted_unsafe"], metrics["accepted"])
        if upper <= target and metrics["resource_per_sample"] <= budget:
            score = (metrics["semantic_goodput"], metrics["goodput_per_resource"], -upper)
            if best is None or score > best[0]:
                best = (score, threshold, metrics, upper)
    if best is None:
        threshold = float(np.min(cal.risk) - 1e-6)
        metrics = eval_single(cal, y, open_label, threshold, accept_cost)
        upper = wilson_upper(metrics["accepted_unsafe"], metrics["accepted"])
        return {"threshold": threshold, "cal_goodput": metrics["semantic_goodput"], "cal_openout": metrics["accepted_open_outage"], "cal_openout_upper": upper}
    return {"threshold": float(best[1]), "cal_goodput": best[2]["semantic_goodput"], "cal_openout": best[2]["accepted_open_outage"], "cal_openout_upper": best[3]}


def eval_single(scored: Scored, y: np.ndarray, open_label: np.ndarray, threshold: float, accept_cost: float) -> dict:
    selected = scored.risk <= threshold
    rejected = ~selected
    unsafe = np.logical_or(open_label, scored.pred != y)
    correct = ~unsafe
    accepted_correct = np.logical_and(selected, correct)
    accepted_unsafe = np.logical_and(selected, unsafe)
    resource = float(accept_cost * np.sum(selected) + 0.1 * np.sum(rejected))
    return metrics_dict(selected, np.zeros_like(selected, dtype=bool), rejected, accepted_correct, accepted_unsafe, y, scored.pred, resource)


def select_progressive_policy(core: Scored, refine: Scored, y: np.ndarray, open_label: np.ndarray, target: float, budget: float) -> dict:
    thresholds = candidate_thresholds(np.concatenate([core.risk, refine.risk]))
    best = None
    for q1 in thresholds:
        for q2 in thresholds:
            if q2 < q1:
                continue
            for qr in thresholds:
                metrics = eval_progressive(core, refine, y, open_label, {"q1": q1, "q2": q2, "qr": qr})
                upper = wilson_upper(metrics["accepted_unsafe"], metrics["accepted"])
                if upper <= target and metrics["resource_per_sample"] <= budget:
                    score = (metrics["semantic_goodput"], metrics["goodput_per_resource"], -upper)
                    if best is None or score > best[0]:
                        best = (score, q1, q2, qr, metrics, upper)
    if best is None:
        return {"q1": float(np.min(core.risk) - 1e-6), "q2": float(np.min(core.risk) - 1e-6), "qr": float(np.min(refine.risk) - 1e-6), "cal_goodput": 0.0, "cal_openout": 0.0, "cal_openout_upper": 0.0}
    return {"q1": float(best[1]), "q2": float(best[2]), "qr": float(best[3]), "cal_goodput": best[4]["semantic_goodput"], "cal_openout": best[4]["accepted_open_outage"], "cal_openout_upper": best[5]}


def eval_progressive(core: Scored, refine: Scored, y: np.ndarray, open_label: np.ndarray, policy: dict) -> dict:
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
    resource = float(np.sum(core_accept) + 1.6 * np.sum(refine_mask) + 0.1 * np.sum(resource_rejected))
    return metrics_dict(selected, refine_mask, rejected, accepted_correct, accepted_unsafe, y, final_pred, resource)


def metrics_dict(selected, refined, rejected, accepted_correct, accepted_unsafe, y, pred, resource: float) -> dict:
    n = len(y)
    accepted = int(np.sum(selected))
    return {
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


def wilson_upper(errors: int, total: int, z: float = 1.64) -> float:
    if total <= 0:
        return 0.0
    phat = errors / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return float(min(1.0, (centre + margin) / denom))


def candidate_thresholds(risk: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(risk, np.linspace(0.0, 1.0, 41)))


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

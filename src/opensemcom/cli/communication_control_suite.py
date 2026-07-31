
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
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opensemcom.certification import (
    clopper_pearson_upper,
    minimum_zero_error_accepts,
)

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
    policy_selection: list[Row]
    certificate: list[Row]
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
    p.add_argument("--train-known-per-class", type=int, default=64)
    p.add_argument("--train-open", type=int, default=512)
    p.add_argument("--cal-known-per-class", type=int, default=32)
    p.add_argument("--cal-open", type=int, default=2000)
    p.add_argument("--certificate-fraction", type=float, default=0.70)
    p.add_argument("--certification-alpha", type=float, default=0.05)
    p.add_argument(
        "--certificate-family-size",
        type=int,
        default=1,
        help=(
            "Bonferroni family size for simultaneous reliability claims; "
            "each policy uses certification-alpha / family-size"
        ),
    )
    p.add_argument(
        "--primary-method",
        default="opensemcom_progressive",
        help="Predeclared primary policy; recorded in every output artifact",
    )
    p.add_argument("--selection-safety-factor", type=float, default=0.50)
    p.add_argument("--minimum-certified-accepts", type=int, default=0)
    p.add_argument("--full-open-severity", default="mild:0.25,medium:0.50,hard:0.75,extreme:0.91")
    p.add_argument("--checkpoint-dir", help="Save trained task/seed model bundles under this directory")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.certificate_fraction < 1.0:
        raise ValueError("--certificate-fraction must lie strictly between zero and one")
    if not 0.0 < args.certification_alpha < 1.0:
        raise ValueError("--certification-alpha must lie strictly between zero and one")
    if args.certificate_family_size < 1:
        raise ValueError("--certificate-family-size must be at least one")
    if not 0.0 < args.selection_safety_factor <= 1.0:
        raise ValueError("--selection-safety-factor must lie in (0, 1]")
    specs = parse_manifest_specs(args.feature_manifest)
    rows, manifest_summary = load_rows(specs)
    channel = load_channel_context(Path(args.deepsense_scenario_root))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    budgets = [float(x) for x in args.resource_budgets.split(",") if x.strip()]
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    severities = parse_severities(args.full_open_severity)
    provenance = build_provenance(args, specs, manifest_summary, channel, seeds, targets, budgets, severities)

    summary_rows: list[dict] = []
    policy_rows: list[dict] = []
    split_audits: list[dict] = []
    for seed in seeds:
        for severity_name, open_fraction in severities:
            full_split = make_fullopen_split(rows, seed, args, open_fraction)
            task_name = f"full-open-{severity_name}"
            split_audits.append(split_audit(task_name, seed, full_split))
            summary_rows.extend(run_task(task_name, full_split, specs, channel, targets, budgets, seed, args, policy_rows, checkpoint_dir, provenance))
        sector_split = make_deepsense_split(rows, seed, sectors=True)
        split_audits.append(split_audit("deepsense-sector", seed, sector_split))
        summary_rows.extend(run_task("deepsense-sector", sector_split, specs, channel, targets, budgets, seed, args, policy_rows, checkpoint_dir, provenance))
        exact_split = make_deepsense_split(rows, seed, sectors=False)
        split_audits.append(split_audit("deepsense-exact", seed, exact_split))
        summary_rows.extend(run_task("deepsense-exact", exact_split, specs, channel, targets, budgets, seed, args, policy_rows, checkpoint_dir, provenance))

    write_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), summary_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_policies.csv"), policy_rows)
    output_prefix.with_name(output_prefix.name + "_manifest_summary.json").write_text(
        json.dumps(
            {
                **manifest_summary,
                "channel_context": channel.summary(),
                "certificate_protocol": provenance["configuration"],
                "split_audits": split_audits,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if checkpoint_dir is not None:
        write_checkpoint_index(checkpoint_dir, provenance)
    print(json.dumps({"summary_rows": len(summary_rows), "policy_rows": len(policy_rows), "channel_context": channel.summary(), "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None}, indent=2, sort_keys=True))


def run_task(
    task_name: str,
    split: SplitData,
    specs: dict[str, Path],
    channel: ChannelContext,
    targets: list[float],
    budgets: list[float],
    seed: int,
    args,
    policy_rows: list[dict],
    checkpoint_dir: Path | None = None,
    provenance: dict | None = None,
) -> list[dict]:
    validate_split_disjoint(split)
    for name, values in (
        ("train", split.train),
        ("policy_selection", split.policy_selection),
        ("certificate", split.certificate),
        ("eval", split.eval),
    ):
        if not values:
            raise ValueError(f"{task_name} has an empty required {name} cohort")
    train_y, train_open = labels_for(task_name, split.train)
    selection_y, selection_open = labels_for(task_name, split.policy_selection)
    certificate_y, certificate_open = labels_for(task_name, split.certificate)
    eval_y, eval_open = labels_for(task_name, split.eval)
    label_arrays = [
        values
        for values in (train_y, selection_y, certificate_y, eval_y)
        if len(values)
    ]
    n_classes = int(max(float(np.max(values)) for values in label_arrays)) + 1

    arrays = {
        "dino": (
            make_features(split.train, ("dino",), channel, False),
            make_features(split.policy_selection, ("dino",), channel, False),
            make_features(split.certificate, ("dino",), channel, False),
            make_features(split.eval, ("dino",), channel, False),
        ),
        "ensemble": (
            make_features(split.train, tuple(specs), channel, False),
            make_features(split.policy_selection, tuple(specs), channel, False),
            make_features(split.certificate, tuple(specs), channel, False),
            make_features(split.eval, tuple(specs), channel, False),
        ),
        "ensemble_channel": (
            make_features(split.train, tuple(specs), channel, True),
            make_features(split.policy_selection, tuple(specs), channel, True),
            make_features(split.certificate, tuple(specs), channel, True),
            make_features(split.eval, tuple(specs), channel, True),
        ),
    }

    dino = fit_model(arrays["dino"][0], train_y, train_open, n_classes, detector_open=task_name.startswith("full-open"))
    ens = fit_model(arrays["ensemble"][0], train_y, train_open, n_classes, detector_open=task_name.startswith("full-open"))
    ens_ch = fit_model(arrays["ensemble_channel"][0], train_y, train_open, n_classes, detector_open=task_name.startswith("full-open"))
    jscc = fit_deepjscc(arrays["dino"][0], train_y, train_open, n_classes, detector_open=task_name.startswith("full-open"))
    receiver_ch = fit_receiver(arrays["ensemble_channel"][0], train_y, train_open, n_classes, has_open_class=task_name.startswith("full-open"), seed=seed)
    receiver_ens = fit_receiver(arrays["ensemble"][0], train_y, train_open, n_classes, has_open_class=task_name.startswith("full-open"), seed=seed + 1000)

    if checkpoint_dir is not None:
        save_task_checkpoint_bundle(
            checkpoint_dir=checkpoint_dir,
            task_name=task_name,
            seed=seed,
            split=split,
            n_classes=n_classes,
            models={"dino": dino, "ensemble": ens, "ensemble_channel": ens_ch, "deepjscc": jscc},
            receiver_channel=receiver_ch,
            receiver_no_channel=receiver_ens,
            provenance=provenance or {},
        )

    dino_selection = score_model(dino, arrays["dino"][1])
    dino_certificate = score_model(dino, arrays["dino"][2])
    dino_eval = score_model(dino, arrays["dino"][3])
    ens_selection = score_model(ens, arrays["ensemble"][1])
    ens_certificate = score_model(ens, arrays["ensemble"][2])
    ens_eval = score_model(ens, arrays["ensemble"][3])
    ens_ch_selection = score_model(ens_ch, arrays["ensemble_channel"][1])
    ens_ch_certificate = score_model(ens_ch, arrays["ensemble_channel"][2])
    ens_ch_eval = score_model(ens_ch, arrays["ensemble_channel"][3])
    jscc_selection = score_deepjscc(jscc, arrays["dino"][1])
    jscc_certificate = score_deepjscc(jscc, arrays["dino"][2])
    jscc_eval = score_deepjscc(jscc, arrays["dino"][3])
    recv_ch_selection = score_receiver(receiver_ch, arrays["ensemble_channel"][1])
    recv_ch_certificate = score_receiver(receiver_ch, arrays["ensemble_channel"][2])
    recv_ch_eval = score_receiver(receiver_ch, arrays["ensemble_channel"][3])
    recv_ens_selection = score_receiver(receiver_ens, arrays["ensemble"][1])
    recv_ens_certificate = score_receiver(receiver_ens, arrays["ensemble"][2])
    recv_ens_eval = score_receiver(receiver_ens, arrays["ensemble"][3])

    methods = {
        "dino_detector": (dino_selection, dino_certificate, dino_eval, 1.0),
        "ensemble_detector": (ens_selection, ens_certificate, ens_eval, 1.2),
        "deepjscc_pca": (jscc_selection, jscc_certificate, jscc_eval, 0.7),
        "witt_context_style": (ens_ch_selection, ens_ch_certificate, ens_ch_eval, 1.4),
        "fixed_refine_all": (ens_ch_selection, ens_ch_certificate, ens_ch_eval, 1.6),
        "opensemcom_receiver_only": (recv_ch_selection, recv_ch_certificate, recv_ch_eval, 1.2),
        "opensemcom_no_channel": (recv_ens_selection, recv_ens_certificate, recv_ens_eval, 1.2),
    }
    declared_methods = {*methods, "opensemcom_progressive"}
    if args.primary_method not in declared_methods:
        raise ValueError(
            f"Unknown --primary-method {args.primary_method!r}; "
            f"choose one of {sorted(declared_methods)}"
        )
    dino_channel_selection = fuse_scores(dino_selection, ens_ch_selection, disagreement_penalty=0.04)
    dino_channel_certificate = fuse_scores(dino_certificate, ens_ch_certificate, disagreement_penalty=0.04)
    dino_channel_eval = fuse_scores(dino_eval, ens_ch_eval, disagreement_penalty=0.04)
    ensemble_channel_selection = fuse_scores(ens_selection, ens_ch_selection, disagreement_penalty=0.04)
    ensemble_channel_certificate = fuse_scores(ens_certificate, ens_ch_certificate, disagreement_penalty=0.04)
    ensemble_channel_eval = fuse_scores(ens_eval, ens_ch_eval, disagreement_penalty=0.04)
    receiver_channel_selection = fuse_scores(recv_ch_selection, ens_ch_selection, disagreement_penalty=0.03)
    receiver_channel_certificate = fuse_scores(recv_ch_certificate, ens_ch_certificate, disagreement_penalty=0.03)
    receiver_channel_eval = fuse_scores(recv_ch_eval, ens_ch_eval, disagreement_penalty=0.03)
    receiver_dino_selection = fuse_scores(recv_ch_selection, dino_channel_selection, disagreement_penalty=0.03)
    receiver_dino_certificate = fuse_scores(recv_ch_certificate, dino_channel_certificate, disagreement_penalty=0.03)
    receiver_dino_eval = fuse_scores(recv_ch_eval, dino_channel_eval, disagreement_penalty=0.03)
    progressive_candidates = {
        "dino_core": (dino_selection, dino_certificate, dino_eval, ens_ch_selection, ens_ch_certificate, ens_ch_eval),
        "ensemble_core": (ens_selection, ens_certificate, ens_eval, ens_ch_selection, ens_ch_certificate, ens_ch_eval),
        "dino_channel_fusion_core": (dino_channel_selection, dino_channel_certificate, dino_channel_eval, ens_ch_selection, ens_ch_certificate, ens_ch_eval),
        "ensemble_channel_fusion_core": (ensemble_channel_selection, ensemble_channel_certificate, ensemble_channel_eval, ens_ch_selection, ens_ch_certificate, ens_ch_eval),
        "trained_receiver_core": (recv_ch_selection, recv_ch_certificate, recv_ch_eval, ens_ch_selection, ens_ch_certificate, ens_ch_eval),
        "trained_receiver_channel_fusion_core": (receiver_channel_selection, receiver_channel_certificate, receiver_channel_eval, ens_ch_selection, ens_ch_certificate, ens_ch_eval),
        "trained_receiver_dino_fusion_core": (receiver_dino_selection, receiver_dino_certificate, receiver_dino_eval, ens_ch_selection, ens_ch_certificate, ens_ch_eval),
        "trained_receiver_no_channel_core": (recv_ens_selection, recv_ens_certificate, recv_ens_eval, ens_ch_selection, ens_ch_certificate, ens_ch_eval),
    }

    policy_alpha = args.certification_alpha / args.certificate_family_size
    rows: list[dict] = []
    for target in targets:
        for budget in budgets:
            for name, (selection_score, certificate_score, eval_score, accept_cost) in methods.items():
                policy = select_single_policy(
                    selection_score,
                    selection_y,
                    selection_open,
                    target,
                    budget,
                    accept_cost,
                    safety_factor=args.selection_safety_factor,
                )
                policy = certify_single_policy(
                    policy,
                    certificate_score,
                    certificate_y,
                    certificate_open,
                    target,
                    accept_cost,
                    alpha=policy_alpha,
                    minimum_accepts=args.minimum_certified_accepts,
                )
                policy["certificate_family_alpha"] = args.certification_alpha
                policy["certificate_family_size"] = args.certificate_family_size
                policy_rows.append({"task": task_name, "method": name, "is_primary_policy": name == args.primary_method, "seed": seed, "target_openout": target, "resource_budget": budget, **policy})
                metrics = eval_single(eval_score, eval_y, eval_open, policy["threshold"], accept_cost)
                rows.append({"task": task_name, "method": name, "is_primary_policy": name == args.primary_method, "seed": seed, "target_openout": target, "resource_budget": budget, **certificate_columns(policy), **metrics})
            policy = select_best_progressive_policy(
                progressive_candidates,
                selection_y,
                selection_open,
                target,
                budget,
                safety_factor=args.selection_safety_factor,
            )
            route = policy["route"]
            (
                _core_selection,
                core_certificate,
                core_eval,
                _refine_selection,
                refine_certificate,
                refine_eval,
            ) = progressive_candidates[route]
            policy = certify_progressive_policy(
                policy,
                core_certificate,
                refine_certificate,
                certificate_y,
                certificate_open,
                target,
                alpha=policy_alpha,
                minimum_accepts=args.minimum_certified_accepts,
            )
            policy["certificate_family_alpha"] = args.certification_alpha
            policy["certificate_family_size"] = args.certificate_family_size
            policy_rows.append({"task": task_name, "method": "opensemcom_progressive", "is_primary_policy": args.primary_method == "opensemcom_progressive", "seed": seed, "target_openout": target, "resource_budget": budget, **policy})
            rows.append({"task": task_name, "method": "opensemcom_progressive", "is_primary_policy": args.primary_method == "opensemcom_progressive", "seed": seed, "target_openout": target, "resource_budget": budget, **certificate_columns(policy), **eval_progressive(core_eval, refine_eval, eval_y, eval_open, policy)})
    return rows


def parse_severities(value: str) -> list[tuple[str, float]]:
    severities: list[tuple[str, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, raw_fraction = item.split(":", 1)
        fraction = float(raw_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Invalid severity open fraction: {item}")
        severities.append((name.strip(), fraction))
    if not severities:
        raise ValueError("At least one full-open severity is required")
    return severities


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


def make_fullopen_split(rows: list[Row], seed: int, args, open_fraction: float = 0.50) -> SplitData:
    rng = np.random.default_rng(seed)
    known_all = unique_source_rows(
        sorted(
            (row for row in rows if row.known_id),
            key=lambda row: (row.regime != "closed-id", row.split != "calibration"),
        )
    )
    open_all = unique_source_rows(
        row
        for row in rows
        if row.regime == "full-open" and row.open_exposure
    )
    by_class = {label: [] for label in range(KNOWN_CLASSES)}
    for row in known_all:
        by_class[row.label].append(row)
    train, known_cal_pool, eval_known = [], [], []
    for label, values in by_class.items():
        values = shuffled(values, rng)
        train += values[: args.train_known_per_class]
        known_cal_pool += values[
            args.train_known_per_class :
            args.train_known_per_class + args.cal_known_per_class
        ]
        eval_known += values[args.train_known_per_class + args.cal_known_per_class :]
    open_values = shuffled(open_all, rng)
    train += open_values[: args.train_open]
    open_cal_pool = open_values[args.train_open : args.train_open + args.cal_open]
    eval_open = open_values[args.train_open + args.cal_open :]

    calibration_total = maximum_mixture_size(
        len(known_cal_pool),
        len(open_cal_pool),
        open_fraction,
    )
    calibration_open_n = min(
        len(open_cal_pool),
        int(round(calibration_total * open_fraction)),
    )
    calibration_known_n = min(
        len(known_cal_pool),
        calibration_total - calibration_open_n,
    )
    selected_known = shuffled(known_cal_pool, rng)[:calibration_known_n]
    selected_open = shuffled(open_cal_pool, rng)[:calibration_open_n]
    validate_mixture_fraction(
        "calibration",
        len(selected_known),
        len(selected_open),
        open_fraction,
    )
    certificate_fraction = float(np.clip(args.certificate_fraction, 0.05, 0.95))
    known_certificate_n = int(round(len(selected_known) * certificate_fraction))
    open_certificate_n = int(round(len(selected_open) * certificate_fraction))
    certificate = (
        selected_known[:known_certificate_n]
        + selected_open[:open_certificate_n]
    )
    policy_selection = (
        selected_known[known_certificate_n:]
        + selected_open[open_certificate_n:]
    )

    evaluation_total = min(
        int(args.eval_size),
        maximum_mixture_size(
            len(eval_known),
            len(eval_open),
            open_fraction,
        ),
    )
    open_n = min(
        len(eval_open),
        int(round(evaluation_total * float(open_fraction))),
    )
    known_n = min(len(eval_known), evaluation_total - open_n)
    validate_mixture_fraction("evaluation", known_n, open_n, open_fraction)
    eval_rows = shuffled(eval_known, rng)[:known_n] + shuffled(eval_open, rng)[:open_n]
    return SplitData(
        shuffled(train, rng),
        shuffled(policy_selection, rng),
        shuffled(certificate, rng),
        shuffled(eval_rows, rng),
    )


def make_deepsense_split(rows: list[Row], seed: int, sectors: bool = True) -> SplitData:
    rng = np.random.default_rng(seed)
    ds = unique_source_rows(
        r
        for r in rows
        if r.dataset == "deepsense6g" and r.task == "beam-prediction"
    )
    if len(ds) < 100:
        raise ValueError("Not enough DeepSense rows for beam task")
    raw_labels = sorted({r.label for r in ds})
    label_rank = {label: idx for idx, label in enumerate(raw_labels)}
    remapped = []
    for r in ds:
        if sectors:
            label = min(DEEPSENSE_BEAM_SECTORS - 1, int(label_rank[r.label] * DEEPSENSE_BEAM_SECTORS / max(len(raw_labels), 1)))
        else:
            label = label_rank[r.label]
        remapped.append(Row(r.key, r.paths, r.raw_source_path, r.dataset, label, r.task, r.domain, False, r.split, r.regime))
    by_label: dict[int, list[Row]] = {}
    for row in remapped:
        by_label.setdefault(row.label, []).append(row)
    train, policy_selection, certificate, ev = [], [], [], []
    for values in by_label.values():
        values = shuffled(values, rng)
        n = len(values)
        train_end = max(1, int(0.50 * n))
        policy_end = max(train_end + 1, int(0.65 * n))
        certificate_end = max(policy_end + 1, int(0.85 * n))
        train += values[:train_end]
        policy_selection += values[train_end:policy_end]
        certificate += values[policy_end:certificate_end]
        ev += values[certificate_end:]
    return SplitData(
        shuffled(train, rng),
        shuffled(policy_selection, rng),
        shuffled(certificate, rng),
        shuffled(ev, rng),
    )


def maximum_mixture_size(
    known_available: int,
    open_available: int,
    open_fraction: float,
) -> int:
    """Largest cohort with the requested known/open mixture."""

    fraction = float(np.clip(open_fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return int(known_available)
    if fraction >= 1.0:
        return int(open_available)
    return max(
        0,
        int(
            min(
                known_available / (1.0 - fraction),
                open_available / fraction,
            )
        ),
    )


def unique_source_rows(rows) -> list[Row]:
    """Keep one deterministic row per raw source identity."""

    unique: dict[tuple[str, ...], Row] = {}
    for row in rows:
        unique.setdefault(row.key[:2], row)
    return list(unique.values())


def validate_mixture_fraction(
    cohort: str,
    known_count: int,
    open_count: int,
    requested_open_fraction: float,
) -> None:
    total = known_count + open_count
    if total <= 0:
        raise ValueError(f"{cohort} cohort is empty")
    actual = open_count / total
    tolerance = max(0.01, 1.0 / total)
    if abs(actual - requested_open_fraction) > tolerance:
        raise ValueError(
            f"{cohort} open fraction {actual:.4f} does not match requested "
            f"{requested_open_fraction:.4f}; add source rows instead of "
            "silently changing severity"
        )


def validate_split_disjoint(split: SplitData) -> None:
    """Fail on source leakage across fit, selection, certificate, and eval."""

    groups = {
        "train": {row.key[:2] for row in split.train},
        "policy_selection": {row.key[:2] for row in split.policy_selection},
        "certificate": {row.key[:2] for row in split.certificate},
        "eval": {row.key[:2] for row in split.eval},
    }
    rows_by_name = {
        "train": split.train,
        "policy_selection": split.policy_selection,
        "certificate": split.certificate,
        "eval": split.eval,
    }
    for name, identities in groups.items():
        if len(identities) != len(rows_by_name[name]):
            raise ValueError(f"Duplicate source artifact inside {name} split.")
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                example = next(iter(overlap))
                raise ValueError(
                    f"Source leakage between {left} and {right}: {example}"
                )


def split_audit(task: str, seed: int, split: SplitData) -> dict:
    validate_split_disjoint(split)
    groups = {
        "train": split.train,
        "policy_selection": split.policy_selection,
        "certificate": split.certificate,
        "eval": split.eval,
    }
    return {
        "task": task,
        "seed": seed,
        "rows": {name: len(values) for name, values in groups.items()},
        "row_key_sha256": {
            name: row_key_sha256(values)
            for name, values in groups.items()
        },
        "open_fraction": {
            name: (
                float(np.mean([row.open_exposure for row in values]))
                if values
                else 0.0
            )
            for name, values in groups.items()
        },
    }


def labels_for(task_name: str, rows: list[Row]) -> tuple[np.ndarray, np.ndarray]:
    if task_name.startswith("full-open"):
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
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        has_open_class: bool,
        seed: int,
        hidden_dim: int = 256,
        epochs: int = 45,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        import torch

        self.torch = torch
        self.input_dim = int(input_dim)
        self.n_classes = int(n_classes)
        self.has_open_class = bool(has_open_class)
        self.seed = int(seed)
        self.hidden_dim = int(hidden_dim)
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected_device)
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

    def save_checkpoint(self, path: Path) -> None:
        torch = self.torch
        if self.mean is None or self.std is None:
            raise ValueError("Cannot save an unfitted receiver")
        payload = {
            "format_version": 1,
            "model_type": "opensemcom_trained_receiver",
            "input_dim": self.input_dim,
            "n_classes": self.n_classes,
            "has_open_class": self.has_open_class,
            "seed": self.seed,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "mean": torch.as_tensor(self.mean, dtype=torch.float32),
            "std": torch.as_tensor(self.std, dtype=torch.float32),
            "encoder_state": cpu_state_dict(self.model),
            "class_head_state": cpu_state_dict(self.class_head),
            "unsafe_head_state": cpu_state_dict(self.unsafe_head),
            "accept_head_state": cpu_state_dict(self.accept_head),
        }
        atomic_torch_save(payload, path, torch)

    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cpu") -> "TrainedReceiver":
        import torch

        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("format_version") != 1 or payload.get("model_type") != "opensemcom_trained_receiver":
            raise ValueError(f"Unsupported receiver checkpoint: {path}")
        receiver = cls(
            input_dim=int(payload["input_dim"]),
            n_classes=int(payload["n_classes"]),
            has_open_class=bool(payload["has_open_class"]),
            seed=int(payload["seed"]),
            hidden_dim=int(payload["hidden_dim"]),
            epochs=int(payload["epochs"]),
            lr=float(payload["lr"]),
            device=device,
        )
        receiver.model.load_state_dict(payload["encoder_state"])
        receiver.class_head.load_state_dict(payload["class_head_state"])
        receiver.unsafe_head.load_state_dict(payload["unsafe_head_state"])
        receiver.accept_head.load_state_dict(payload["accept_head_state"])
        receiver.mean = payload["mean"].cpu().numpy()
        receiver.std = payload["std"].cpu().numpy()
        receiver.model.eval()
        receiver.class_head.eval()
        receiver.unsafe_head.eval()
        receiver.accept_head.eval()
        return receiver

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


def select_single_policy(
    cal: Scored,
    y: np.ndarray,
    open_label: np.ndarray,
    target: float,
    budget: float,
    accept_cost: float,
    safety_factor: float = 0.5,
) -> dict:
    best = None
    selection_target = float(np.clip(target * safety_factor, 0.0, 1.0))
    for threshold in candidate_thresholds(cal.risk):
        metrics = eval_single(cal, y, open_label, threshold, accept_cost, include_detection=False)
        empirical_outage = metrics["accepted_open_outage"]
        if empirical_outage <= selection_target and metrics["resource_per_sample"] <= budget:
            score = (
                metrics["semantic_goodput"],
                metrics["goodput_per_resource"],
                -empirical_outage,
            )
            if best is None or score > best[0]:
                best = (score, threshold, metrics)
    if best is None:
        threshold = reject_threshold(cal.risk)
        metrics = eval_single(cal, y, open_label, threshold, accept_cost, include_detection=False)
        return {
            "threshold": threshold,
            "cal_goodput": metrics["semantic_goodput"],
            "cal_openout": metrics["accepted_open_outage"],
            "cal_openout_upper": 1.0,
        }
    return {
        "threshold": float(best[1]),
        "cal_goodput": best[2]["semantic_goodput"],
        "cal_openout": best[2]["accepted_open_outage"],
        "cal_openout_upper": 1.0,
    }


def certify_single_policy(
    policy: dict,
    certificate_score: Scored,
    y: np.ndarray,
    open_label: np.ndarray,
    target: float,
    accept_cost: float,
    alpha: float = 0.05,
    minimum_accepts: int = 0,
) -> dict:
    """Certify one threshold selected without inspecting this split."""

    selected_threshold = float(policy["threshold"])
    metrics = eval_single(
        certificate_score,
        y,
        open_label,
        selected_threshold,
        accept_cost,
        include_detection=False,
    )
    certificate = certificate_fields(
        metrics,
        len(y),
        target,
        alpha,
        minimum_accepts,
    )
    deployed_threshold = (
        selected_threshold
        if certificate["certificate_valid"]
        else float("-inf")
    )
    return {
        **policy,
        "selected_threshold": selected_threshold,
        "threshold": deployed_threshold,
        **certificate,
    }


def eval_single(scored: Scored, y: np.ndarray, open_label: np.ndarray, threshold: float, accept_cost: float, include_detection: bool = True) -> dict:
    selected = scored.risk <= threshold
    rejected = ~selected
    unsafe = np.logical_or(open_label, scored.pred != y)
    correct = ~unsafe
    accepted_correct = np.logical_and(selected, correct)
    accepted_unsafe = np.logical_and(selected, unsafe)
    resource = float(accept_cost * np.sum(selected) + 0.1 * np.sum(rejected))
    latency = float(np.sum(selected) + 0.2 * np.sum(rejected))
    return metrics_dict(selected, np.zeros_like(selected, dtype=bool), rejected, accepted_correct, accepted_unsafe, y, scored.pred, scored.risk, unsafe, resource, latency, include_detection)


def select_best_progressive_policy(
    candidates: dict[str, tuple[Scored, Scored, Scored, Scored, Scored, Scored]],
    y: np.ndarray,
    open_label: np.ndarray,
    target: float,
    budget: float,
    safety_factor: float = 0.5,
) -> dict:
    best = None
    for route, (
        core_selection,
        _core_certificate,
        _core_eval,
        refine_selection,
        _refine_certificate,
        _refine_eval,
    ) in candidates.items():
        policy = select_progressive_policy(
            core_selection,
            refine_selection,
            y,
            open_label,
            target,
            budget,
            safety_factor=safety_factor,
        )
        score = (policy["cal_goodput"], policy.get("cal_goodput_per_resource", 0.0), -policy.get("cal_openout_upper", 0.0))
        if best is None or score > best[0]:
            best = (score, route, policy)
    if best is None:
        raise ValueError("No progressive candidate policies were evaluated")
    return {"route": best[1], **best[2]}


def select_progressive_policy(
    core: Scored,
    refine: Scored,
    y: np.ndarray,
    open_label: np.ndarray,
    target: float,
    budget: float,
    safety_factor: float = 0.5,
) -> dict:
    core_thresholds = candidate_thresholds(core.risk)
    refine_thresholds = candidate_thresholds(refine.risk)
    best = None
    selection_target = float(np.clip(target * safety_factor, 0.0, 1.0))
    for q1 in core_thresholds:
        for q2 in core_thresholds:
            if q2 < q1:
                continue
            for qr in refine_thresholds:
                metrics = eval_progressive(core, refine, y, open_label, {"q1": q1, "q2": q2, "qr": qr}, include_detection=False)
                empirical_outage = metrics["accepted_open_outage"]
                if empirical_outage <= selection_target and metrics["resource_per_sample"] <= budget:
                    score = (
                        metrics["semantic_goodput"],
                        metrics["goodput_per_resource"],
                        -empirical_outage,
                    )
                    if best is None or score > best[0]:
                        best = (score, q1, q2, qr, metrics)
    if best is None:
        return {
            "q1": reject_threshold(core.risk),
            "q2": reject_threshold(core.risk),
            "qr": reject_threshold(refine.risk),
            "cal_goodput": 0.0,
            "cal_openout": 0.0,
            "cal_openout_upper": 1.0,
            "cal_goodput_per_resource": 0.0,
            "cal_resource_per_sample": 0.0,
            "cal_refine_rate": 0.0,
        }
    return {
        "q1": float(best[1]),
        "q2": float(best[2]),
        "qr": float(best[3]),
        "cal_goodput": best[4]["semantic_goodput"],
        "cal_openout": best[4]["accepted_open_outage"],
        "cal_openout_upper": 1.0,
        "cal_goodput_per_resource": best[4]["goodput_per_resource"],
        "cal_resource_per_sample": best[4]["resource_per_sample"],
        "cal_refine_rate": best[4]["refine_rate"],
    }


def certify_progressive_policy(
    policy: dict,
    core_certificate: Scored,
    refine_certificate: Scored,
    y: np.ndarray,
    open_label: np.ndarray,
    target: float,
    alpha: float = 0.05,
    minimum_accepts: int = 0,
) -> dict:
    """Certify the selected route and complete progressive decision policy."""

    selected = {
        "q1": float(policy["q1"]),
        "q2": float(policy["q2"]),
        "qr": float(policy["qr"]),
    }
    metrics = eval_progressive(
        core_certificate,
        refine_certificate,
        y,
        open_label,
        selected,
        include_detection=False,
    )
    certificate = certificate_fields(
        metrics,
        len(y),
        target,
        alpha,
        minimum_accepts,
    )
    if certificate["certificate_valid"]:
        deployed = selected
    else:
        deployed = {
            "q1": float("-inf"),
            "q2": float("-inf"),
            "qr": float("-inf"),
        }
    return {
        **policy,
        "selected_q1": selected["q1"],
        "selected_q2": selected["q2"],
        "selected_qr": selected["qr"],
        **deployed,
        **certificate,
    }


def certificate_fields(
    metrics: dict,
    certificate_samples: int,
    target: float,
    alpha: float,
    minimum_accepts: int,
) -> dict:
    required = max(
        int(minimum_accepts),
        minimum_zero_error_accepts(target, alpha),
    )
    accepted = int(metrics["accepted"])
    unsafe = int(metrics["accepted_unsafe"])
    upper = (
        clopper_pearson_upper(unsafe, accepted, alpha)
        if accepted > 0
        else 1.0
    )
    valid = accepted >= required and upper <= target
    if accepted < required:
        reason = f"accepted {accepted} certificate examples; {required} required"
    elif upper > target:
        reason = "exact one-sided accepted-outage bound exceeds target"
    else:
        reason = ""
    return {
        "certificate_valid": bool(valid),
        "certificate_method": "clopper-pearson-composed-policy",
        "certificate_alpha": float(alpha),
        "certificate_confidence": float(1.0 - alpha),
        "certificate_target_outage": float(target),
        "certificate_upper_bound": float(upper),
        "certificate_samples": int(certificate_samples),
        "certificate_accepted": accepted,
        "certificate_unsafe": unsafe,
        "certificate_minimum_accepts": required,
        "certificate_reason": reason,
    }


def certificate_columns(policy: dict) -> dict:
    return {
        key: value
        for key, value in policy.items()
        if key.startswith("certificate_")
    }


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
    latency = float(np.sum(core_accept) + 2.0 * np.sum(refine_mask) + 0.2 * np.sum(resource_rejected))
    return metrics_dict(selected, refine_mask, rejected, accepted_correct, accepted_unsafe, y, final_pred, final_risk, unsafe, resource, latency, include_detection)


def metrics_dict(selected, refined, rejected, accepted_correct, accepted_unsafe, y, pred, risk, unsafe, resource: float, latency: float, include_detection: bool) -> dict:
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
        "latency_units": latency,
        "latency_per_sample": float(latency / max(n, 1)),
        "goodput_per_latency": float(np.sum(accepted_correct) / max(latency, 1e-9)),
        "retransmission_rate": float(np.mean(refined)) if n else 0.0,
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


def candidate_thresholds(risk: np.ndarray) -> np.ndarray:
    if risk.size == 0:
        return np.asarray([-1.0], dtype=np.float64)
    return np.unique(np.quantile(risk, np.linspace(0.0, 1.0, 11)))


def reject_threshold(risk: np.ndarray) -> float:
    if risk.size == 0:
        return -1.0
    return float(np.min(risk) - 1e-6)


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


def build_provenance(args, specs: dict[str, Path], manifest_summary: dict, channel: ChannelContext, seeds: list[int], targets: list[float], budgets: list[float], severities: list[tuple[str, float]]) -> dict:
    import sklearn
    import torch

    return {
        "format_version": 2,
        "suite": "communication_control_suite",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "feature_manifests": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in sorted(specs.items())
        },
        "manifest_summary": manifest_summary,
        "channel_context": channel.summary(),
        "configuration": {
            "seeds": seeds,
            "targets": targets,
            "resource_budgets": budgets,
            "eval_size": args.eval_size,
            "train_known_per_class": args.train_known_per_class,
            "train_open": args.train_open,
            "cal_known_per_class": args.cal_known_per_class,
            "cal_open": args.cal_open,
            "certificate_fraction": args.certificate_fraction,
            "certification_family_alpha": args.certification_alpha,
            "certificate_family_size": args.certificate_family_size,
            "certification_policy_alpha": (
                args.certification_alpha / args.certificate_family_size
            ),
            "primary_method": args.primary_method,
            "selection_safety_factor": args.selection_safety_factor,
            "minimum_certified_accepts": args.minimum_certified_accepts,
            "full_open_severity": dict(severities),
            "deepsense_scenario_root": str(Path(args.deepsense_scenario_root).expanduser().resolve()),
        },
    }


def save_task_checkpoint_bundle(
    checkpoint_dir: Path,
    task_name: str,
    seed: int,
    split: SplitData,
    n_classes: int,
    models: dict,
    receiver_channel: TrainedReceiver,
    receiver_no_channel: TrainedReceiver,
    provenance: dict,
) -> None:
    import joblib

    bundle_dir = checkpoint_dir / task_name / f"seed_{seed}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    classical_path = bundle_dir / "classical_models.joblib"
    receiver_channel_path = bundle_dir / "receiver_channel.pt"
    receiver_no_channel_path = bundle_dir / "receiver_no_channel.pt"
    atomic_joblib_dump(models, classical_path, joblib)
    receiver_channel.save_checkpoint(receiver_channel_path)
    receiver_no_channel.save_checkpoint(receiver_no_channel_path)

    model_files = [classical_path, receiver_channel_path, receiver_no_channel_path]
    metadata = {
        "format_version": 2,
        "task": task_name,
        "seed": seed,
        "n_classes": n_classes,
        "split_rows": {
            "train": len(split.train),
            "policy_selection": len(split.policy_selection),
            "certificate": len(split.certificate),
            "eval": len(split.eval),
        },
        "split_key_sha256": {
            "train": row_key_sha256(split.train),
            "policy_selection": row_key_sha256(split.policy_selection),
            "certificate": row_key_sha256(split.certificate),
            "eval": row_key_sha256(split.eval),
        },
        "feature_manifests": provenance.get("feature_manifests", {}),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in model_files
        },
    }
    atomic_write_json(bundle_dir / "bundle.json", metadata)


def write_checkpoint_index(checkpoint_dir: Path, provenance: dict) -> None:
    files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file() and path.name != "checkpoint_index.json")
    bundles = sorted(str(path.parent.relative_to(checkpoint_dir)) for path in checkpoint_dir.rglob("bundle.json"))
    payload = {
        **provenance,
        "bundles": bundles,
        "bundle_count": len(bundles),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": {
            str(path.relative_to(checkpoint_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        },
    }
    atomic_write_json(checkpoint_dir / "checkpoint_index.json", payload)


def load_classical_checkpoint(path: Path) -> dict:
    import joblib

    return joblib.load(path)


def cpu_state_dict(module) -> dict:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def atomic_torch_save(payload: dict, path: Path, torch_module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch_module.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_joblib_dump(payload, path: Path, joblib_module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        joblib_module.dump(payload, tmp, compress=3)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key_sha256(rows: list[Row]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\0".join(row.key).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


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

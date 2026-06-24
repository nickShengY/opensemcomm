"""Real dataset manifest builders and validators."""

from __future__ import annotations

import csv
import json
import pickle
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

SCRATCH_PREFIXES = ("/scratch/", "/home/nickyun/links/scratch/")
REQUIRED_COLUMNS = ("source_path", "label", "task", "domain", "is_unknown", "split", "regime")
OPTIONAL_COLUMNS = ("dataset", "label_name", "artifact_index", "artifact_key")
MANIFEST_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass(frozen=True)
class ManifestBuildResult:
    rows: list[dict[str, str]]
    availability: dict[str, dict[str, str | int | bool]]


def build_real_manifest(
    output: str | Path,
    roots: Iterable[str | Path],
    max_per_source: int = 512,
    max_calibration_per_class: int = 64,
    max_eval_per_class: int = 64,
) -> ManifestBuildResult:
    """Build a manifest from discovered real datasets under scratch."""

    output = Path(output).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in roots]
    rows: list[dict[str, str]] = []
    availability: dict[str, dict[str, str | int | bool]] = {}

    cifar10 = _first_existing(
        roots,
        ("cifar-10-batches-py", "cifar10/cifar-10-batches-py", "cifar/cifar-10-batches-py"),
    )
    if cifar10:
        cifar_rows = build_cifar_rows(
            cifar10,
            dataset="cifar10",
            label_key="labels",
            max_calibration_per_class=max_calibration_per_class,
            max_eval_per_class=max_eval_per_class,
        )
        rows.extend(cifar_rows)
        availability["cifar10"] = {"available": True, "root": str(cifar10), "rows": len(cifar_rows)}
    else:
        availability["cifar10"] = {"available": False, "reason": "not found under provided scratch roots"}

    cifar100 = _first_existing(
        roots,
        ("cifar-100-python", "cifar100/cifar-100-python", "cifar/cifar-100-python"),
    )
    if cifar100:
        cifar_rows = build_cifar_rows(
            cifar100,
            dataset="cifar100",
            label_key="fine_labels",
            max_calibration_per_class=0,
            max_eval_per_class=max(1, max_eval_per_class // 2),
            source_regime="source-open",
        )
        rows.extend(cifar_rows)
        availability["cifar100"] = {"available": True, "root": str(cifar100), "rows": len(cifar_rows)}
    else:
        availability["cifar100"] = {"available": False, "reason": "not found under provided scratch roots"}

    coco = _first_existing(roots, ("coco",))
    if coco:
        coco_rows = build_coco_rows(coco, max_rows=max_per_source)
        rows.extend(coco_rows)
        availability["coco"] = {"available": bool(coco_rows), "root": str(coco), "rows": len(coco_rows)}
    else:
        availability["coco"] = {"available": False, "reason": "not found under provided scratch roots"}

    cityscapes = _first_existing(roots, ("cityscapes",))
    if cityscapes:
        city_rows = build_cityscapes_rows(cityscapes, max_rows=max_per_source)
        rows.extend(city_rows)
        availability["cityscapes"] = {"available": bool(city_rows), "root": str(cityscapes), "rows": len(city_rows)}
    else:
        availability["cityscapes"] = {"available": False, "reason": "not found under provided scratch roots"}

    nuscenes = _first_existing(roots, ("nuscenes",))
    if nuscenes:
        nuscenes_rows = build_nuscenes_rows(nuscenes, max_rows=max_per_source)
        rows.extend(nuscenes_rows)
        availability["nuscenes"] = {"available": bool(nuscenes_rows), "root": str(nuscenes), "rows": len(nuscenes_rows)}
    else:
        availability["nuscenes"] = {"available": False, "reason": "not found under provided scratch roots"}

    deepsense = _first_existing(roots, ("deepsense6g",))
    if deepsense:
        deepsense_rows = build_deepsense_rows(deepsense, max_rows=max_per_source)
        rows.extend(deepsense_rows)
        availability["deepsense6g"] = {
            "available": bool(deepsense_rows),
            "root": str(deepsense),
            "rows": len(deepsense_rows),
        }
    else:
        availability["deepsense6g"] = {"available": False, "reason": "not found under provided scratch roots"}

    ag_news = _first_existing(roots, ("ag_news", "text/ag_news"))
    if ag_news:
        text_rows = build_ag_news_rows(ag_news, max_rows=max_per_source)
        rows.extend(text_rows)
        availability["text"] = {"available": bool(text_rows), "root": str(ag_news), "rows": len(text_rows)}
    else:
        availability["text"] = {"available": False, "reason": "not found under provided scratch roots"}

    bdd = _first_existing(roots, ("bdd100k", "bdd100k_hf"))
    if bdd:
        bdd_rows = build_bdd100k_rows(bdd, max_rows=max_per_source)
        rows.extend(bdd_rows)
        bdd_images = len(list((bdd / "images").glob("*.jpg"))) if (bdd / "images").exists() else 0
        availability["bdd100k"] = {
            "available": bool(bdd_rows),
            "root": str(bdd),
            "rows": len(bdd_rows),
            "downloaded_images": bdd_images,
            "reason": (
                "official label/image archives not found or no matching labeled images"
                if not bdd_rows
                else "official BDD100K labels and images staged from downloaded archives"
            ),
        }
    else:
        availability["bdd100k"] = {"available": False, "reason": "not found under provided scratch roots"}

    if not rows:
        raise ValueError("No real dataset rows could be built from the provided scratch roots.")

    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(output, rows)
    report_path = output.with_suffix(".availability.json")
    report_path.write_text(json.dumps(availability, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ManifestBuildResult(rows=rows, availability=availability)


def build_cifar_rows(
    root: str | Path,
    dataset: str,
    label_key: str,
    max_calibration_per_class: int,
    max_eval_per_class: int,
    source_regime: str = "",
) -> list[dict[str, str]]:
    root = Path(root).expanduser().resolve()
    label_names = _cifar_label_names(root)
    rows: list[dict[str, str]] = []

    train_files = sorted(root.glob("data_batch_*")) or [root / "train"]
    if max_calibration_per_class > 0:
        counters: dict[int, int] = defaultdict(int)
        for batch in train_files:
            if not batch.exists():
                continue
            payload = _load_pickle_dict(batch)
            labels = payload[label_key]
            for idx, raw_label in enumerate(labels):
                label = int(raw_label)
                if label >= 6 or counters[label] >= max_calibration_per_class:
                    continue
                counters[label] += 1
                rows.append(
                    _row(
                        batch,
                        label,
                        "classification",
                        dataset,
                        False,
                        "calibration",
                        "closed-id",
                        dataset,
                        _label_name(label_names, label),
                        idx,
                    )
                )
            if all(counters[label] >= max_calibration_per_class for label in range(6)):
                break

    eval_file = root / "test_batch" if (root / "test_batch").exists() else root / "test"
    if eval_file.exists():
        payload = _load_pickle_dict(eval_file)
        labels = payload[label_key]
        counters: dict[tuple[str, int], int] = defaultdict(int)
        for idx, raw_label in enumerate(labels):
            original_label = int(raw_label)
            known = original_label < 6
            label = original_label if known else 6 + (original_label % 2)
            regimes = _cifar_regimes(dataset, known, source_regime)
            for regime in regimes:
                key = (regime, original_label)
                if counters[key] >= max_eval_per_class:
                    continue
                counters[key] += 1
                rows.append(
                    _row(
                        eval_file,
                        label,
                        "classification",
                        dataset,
                        not known,
                        "eval",
                        regime,
                        dataset,
                        _label_name(label_names, original_label),
                        idx,
                    )
                )
    return rows


def build_coco_rows(root: str | Path, max_rows: int) -> list[dict[str, str]]:
    root = Path(root).expanduser().resolve()
    annotation_path = root / "annotations" / "instances_val2017.json"
    image_root = root / "val2017"
    if not annotation_path.exists() or not image_root.exists():
        return []
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): cat["name"] for cat in payload.get("categories", [])}
    image_by_id = {int(img["id"]): img["file_name"] for img in payload.get("images", [])}
    seen_images: set[int] = set()
    rows: list[dict[str, str]] = []
    for ann in payload.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id in seen_images:
            continue
        seen_images.add(image_id)
        category_id = int(ann["category_id"])
        file_name = image_by_id.get(image_id)
        if not file_name:
            continue
        path = image_root / file_name
        if not path.exists():
            continue
        category_rank = sorted(categories).index(category_id) if category_id in categories else category_id
        known = category_rank < 6
        label = category_rank if known else 6 + (category_rank % 2)
        rows.append(
            _row(
                path,
                label,
                "detection",
                "coco",
                not known,
                "eval",
                "task-open" if known else "full-open",
                "coco",
                categories.get(category_id, str(category_id)),
            )
        )
        if len(rows) >= max_rows:
            break
    return rows


def build_cityscapes_rows(root: str | Path, max_rows: int) -> list[dict[str, str]]:
    root = Path(root).expanduser().resolve()
    image_root = root / "leftImg8bit"
    if not image_root.exists():
        return []
    rows: list[dict[str, str]] = []
    city_to_label: dict[str, int] = {}
    for split in ("val", "test", "train"):
        for path in sorted((image_root / split).glob("*/*_leftImg8bit.png")):
            city = path.parent.name
            if city not in city_to_label:
                city_to_label[city] = len(city_to_label) % 6
            rows.append(
                _row(
                    path,
                    city_to_label[city],
                    "segmentation",
                    f"cityscapes-{city}",
                    False,
                    "eval",
                    "source-open" if split != "test" else "task-open",
                    "cityscapes",
                    city,
                )
            )
            if len(rows) >= max_rows:
                return rows
    return rows


def build_ag_news_rows(root: str | Path, max_rows: int) -> list[dict[str, str]]:
    root = Path(root).expanduser().resolve()
    metadata_path = root / "metadata.jsonl"
    if not metadata_path.exists():
        return []
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    rows: list[dict[str, str]] = []
    per_split_limit = max(1, max_rows // 2)
    counters: dict[str, int] = defaultdict(int)
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            split = "calibration" if record["split"] == "train" else "eval"
            if counters[split] >= per_split_limit:
                continue
            path = (root / record["path"]).resolve()
            if not path.exists():
                continue
            label = int(record["label"])
            counters[split] += 1
            rows.append(
                _row(
                    path,
                    label,
                    "text-classification",
                    "ag-news",
                    False,
                    split,
                    "task-open" if split == "eval" else "closed-id",
                    "ag_news",
                    str(record.get("label_text") or label_names.get(label, label)),
                )
            )
    return rows


def build_nuscenes_rows(root: str | Path, max_rows: int) -> list[dict[str, str]]:
    root = Path(root).expanduser().resolve()
    version_root = root / "v1.0-mini"
    if not version_root.exists():
        return []
    sample_data_path = version_root / "sample_data.json"
    sample_path = version_root / "sample.json"
    anns_path = version_root / "sample_annotation.json"
    calibrated_sensor_path = version_root / "calibrated_sensor.json"
    sensor_path = version_root / "sensor.json"
    instance_path = version_root / "instance.json"
    category_path = version_root / "category.json"
    required = (
        sample_data_path,
        sample_path,
        anns_path,
        calibrated_sensor_path,
        sensor_path,
        instance_path,
        category_path,
    )
    if not all(path.exists() for path in required):
        return []
    sample_data = json.loads(sample_data_path.read_text(encoding="utf-8"))
    samples = {item["token"]: item for item in json.loads(sample_path.read_text(encoding="utf-8"))}
    sensors = {item["token"]: item for item in json.loads(sensor_path.read_text(encoding="utf-8"))}
    calibrated = {item["token"]: item for item in json.loads(calibrated_sensor_path.read_text(encoding="utf-8"))}
    categories = {item["token"]: item["name"] for item in json.loads(category_path.read_text(encoding="utf-8"))}
    instances = {
        item["token"]: categories.get(item["category_token"], "unknown")
        for item in json.loads(instance_path.read_text(encoding="utf-8"))
    }
    anns_by_sample: dict[str, list[str]] = defaultdict(list)
    for ann in json.loads(anns_path.read_text(encoding="utf-8")):
        anns_by_sample[ann["sample_token"]].append(instances.get(ann["instance_token"], "unknown"))
    category_names = sorted({cat for values in anns_by_sample.values() for cat in values if cat != "unknown"})
    category_to_label = {category: idx for idx, category in enumerate(category_names[:6])}
    rows: list[dict[str, str]] = []
    for item in sample_data:
        if not item.get("is_key_frame"):
            continue
        calibrated_sensor = calibrated.get(item.get("calibrated_sensor_token", ""), {})
        sensor = sensors.get(calibrated_sensor.get("sensor_token", ""), {})
        channel = sensor.get("channel", "")
        if not channel.startswith("CAM"):
            continue
        path = root / item["filename"]
        if not path.exists():
            continue
        sample = samples.get(item["sample_token"], {})
        labels = anns_by_sample.get(sample.get("token", item["sample_token"]), [])
        if not labels:
            continue
        label_name = Counter(labels).most_common(1)[0][0]
        if label_name not in category_to_label:
            continue
        rows.append(
            _row(
                path,
                category_to_label[label_name],
                "driving-detection",
                f"nuscenes-mini-{channel.lower()}",
                False,
                "eval",
                "source-open",
                "nuscenes",
                label_name,
            )
        )
        if len(rows) >= max_rows:
            break
    return rows


def build_bdd100k_rows(root: str | Path, max_rows: int) -> list[dict[str, str]]:
    root = Path(root).expanduser().resolve()
    label_zip_path = root / "bdd100k_labels.zip"
    image_zip_path = root / "bdd100k_images_100k.zip"
    if not label_zip_path.exists() or not image_zip_path.exists():
        return []

    categories = (
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
        "train",
    )
    category_to_rank = {category: idx for idx, category in enumerate(categories)}
    staged_root = root / "staged"
    rows: list[dict[str, str]] = []

    with zipfile.ZipFile(label_zip_path) as label_zip, zipfile.ZipFile(image_zip_path) as image_zip:
        image_names = {name for name in image_zip.namelist() if name.lower().endswith(".jpg")}
        label_names = sorted(
            name
            for name in label_zip.namelist()
            if name.startswith(("100k/val/", "100k/train/")) and name.lower().endswith(".json")
        )
        for label_member in label_names:
            payload = json.loads(label_zip.read(label_member))
            split = label_member.split("/")[1]
            stem = str(payload.get("name") or Path(label_member).stem)
            image_member = f"100k/{split}/{stem}.jpg"
            if image_member not in image_names:
                continue

            objects = []
            for frame in payload.get("frames") or []:
                objects.extend(frame.get("objects") or [])
            category_counts = Counter(
                str(obj.get("category", "")).strip()
                for obj in objects
                if str(obj.get("category", "")).strip()
            )
            if not category_counts:
                continue
            label_name = category_counts.most_common(1)[0][0]
            rank = category_to_rank.get(label_name, len(category_to_rank))
            known = rank < 6
            label = rank if known else 6 + (rank % 2)

            staged_path = staged_root / image_member
            if not staged_path.exists():
                image_zip.extract(image_member, staged_root)
            if not staged_path.exists():
                continue

            attrs = payload.get("attributes") or {}
            domain_bits = [
                str(attrs.get("scene") or "unknown-scene"),
                str(attrs.get("timeofday") or "unknown-time"),
                str(attrs.get("weather") or "unknown-weather"),
            ]
            rows.append(
                _row(
                    staged_path.resolve(),
                    label,
                    "driving-detection",
                    "bdd100k-" + "-".join(_slug(part) for part in domain_bits),
                    not known,
                    "eval",
                    "task-open" if known else "full-open",
                    "bdd100k",
                    label_name,
                )
            )
            if len(rows) >= max_rows:
                break
    return rows


def build_deepsense_rows(root: str | Path, max_rows: int) -> list[dict[str, str]]:
    root = Path(root).expanduser().resolve()
    rows: list[dict[str, str]] = []
    for csv_path in sorted(root.rglob("scenario*.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            label_col = _first_column(reader.fieldnames, ("unit1_beam_index", "beam_index", "beam", "unit1_beam"))
            artifact_col = _first_column(
                reader.fieldnames,
                ("unit1_rgb", "unit1_camera", "unit1_pwr_60ghz", "unit1_lidar", "unit1_radar"),
            )
            if not label_col or not artifact_col:
                continue
            for record in reader:
                raw_label = str(record.get(label_col, "")).strip().strip("[]'")
                if not raw_label:
                    continue
                try:
                    label = int(float(raw_label.split()[0].split(",")[0]))
                except ValueError:
                    continue
                source = _resolve_deepsense_artifact(csv_path.parent, record.get(artifact_col, ""))
                if not source or not source.exists():
                    continue
                rows.append(
                    _row(
                        source,
                        label,
                        "beam-prediction",
                        csv_path.parent.name.lower(),
                        False,
                        "eval",
                        "task-open",
                        "deepsense6g",
                        f"beam-{label}",
                    )
                )
                if len(rows) >= max_rows:
                    return rows
    return rows


def validate_manifest(path: str | Path, require_scratch: bool = True) -> dict[str, int | list[str]]:
    path = Path(path).expanduser().resolve()
    missing: list[str] = []
    off_scratch: list[str] = []
    rows = 0
    indexed_rows = 0
    splits: dict[str, int] = defaultdict(int)
    regimes: dict[str, int] = defaultdict(int)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Manifest missing required columns: {', '.join(sorted(missing_columns))}")
        for row in reader:
            rows += 1
            source = Path(row["source_path"]).expanduser()
            if not source.is_absolute():
                source = path.parent / source
            source = source.resolve()
            if not source.exists():
                missing.append(str(source))
            if require_scratch and not _is_scratch_path(source):
                off_scratch.append(str(source))
            if row.get("artifact_index", "").strip():
                indexed_rows += 1
                _validate_index(source, int(row["artifact_index"]), row.get("artifact_key") or "data")
            splits[row.get("split") or "eval"] += 1
            regimes[row.get("regime") or ""] += 1

    if missing:
        raise FileNotFoundError(f"Manifest contains {len(missing)} missing real artifacts. First: {missing[0]}")
    if off_scratch:
        raise ValueError(f"Manifest contains {len(off_scratch)} paths outside scratch. First: {off_scratch[0]}")
    return {
        "rows": rows,
        "indexed_rows": indexed_rows,
        "splits": dict(sorted(splits.items())),
        "regimes": dict(sorted(regimes.items())),
    }


def write_manifest(path: str | Path, rows: list[dict[str, str]]) -> None:
    path = Path(path).expanduser().resolve()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def default_roots(project_root: str | Path) -> list[Path]:
    project_root = Path(project_root).expanduser().resolve()
    return [
        project_root / "data",
        Path("/scratch/nickyun/New-Jepa/data"),
        Path("/scratch/nickyun/semcom_cas_wcg_w2s/data"),
        Path("/scratch/nickyun/Dream-Mamba/data"),
    ]


def _row(
    source_path: Path,
    label: int,
    task: str,
    domain: str,
    is_unknown: bool,
    split: str,
    regime: str,
    dataset: str,
    label_name: str,
    artifact_index: int | None = None,
) -> dict[str, str]:
    return {
        "source_path": str(source_path),
        "label": str(label),
        "task": task,
        "domain": domain,
        "is_unknown": "true" if is_unknown else "false",
        "split": split,
        "regime": regime,
        "dataset": dataset,
        "label_name": label_name,
        "artifact_index": "" if artifact_index is None else str(artifact_index),
        "artifact_key": "data" if artifact_index is not None else "",
    }


def _cifar_regimes(dataset: str, known: bool, source_regime: str) -> tuple[str, ...]:
    if source_regime and known:
        return (source_regime, "full-open")
    if known:
        return ("closed-id", "channel-open", "resource-open", "full-open")
    return ("class-open", "full-open")


@lru_cache(maxsize=32)
def _load_pickle_dict(path: Path) -> dict:
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="dtype\\(\\): align should be passed")
            payload = pickle.load(handle, encoding="latin1")
    return {str(key): value for key, value in payload.items()}


def _validate_index(path: Path, index: int, key: str) -> None:
    payload = _load_pickle_dict(path)
    if key not in payload:
        raise ValueError(f"Indexed artifact key '{key}' not found in {path}")
    if index < 0 or index >= len(payload[key]):
        raise IndexError(f"Artifact index {index} out of range for {path}")


def _cifar_label_names(root: Path) -> list[str]:
    for name in ("batches.meta", "meta"):
        path = root / name
        if path.exists():
            payload = _load_pickle_dict(path)
            values = payload.get("label_names") or payload.get("fine_label_names")
            if values:
                return [str(value) for value in values]
    return []


def _label_name(label_names: list[str], label: int) -> str:
    if 0 <= label < len(label_names):
        return label_names[label]
    return str(label)


def _first_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _resolve_deepsense_artifact(base: Path, raw_value: str | None) -> Path | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip().strip("[]'\"")
    if not value:
        return None
    value = value.split()[0].split(",")[0].strip("'\"")
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    candidates = [base / path, base.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _first_existing(roots: Iterable[Path], candidates: Iterable[str]) -> Path | None:
    for root in roots:
        for candidate in candidates:
            path = root / candidate
            if path.exists():
                return path
        if root.name in {Path(candidate).name for candidate in candidates} and root.exists():
            return root
    return None


def _is_scratch_path(path: Path) -> bool:
    text = str(path)
    return any(text.startswith(prefix) for prefix in SCRATCH_PREFIXES)


def _slug(value: str) -> str:
    return value.strip().lower().replace("/", "-").replace(" ", "-")

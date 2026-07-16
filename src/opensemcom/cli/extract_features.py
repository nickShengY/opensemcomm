"""Extract pretrained features from manifest artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from opensemcom.manifest import MANIFEST_COLUMNS, validate_manifest


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_SUFFIXES = {".txt", ".md"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract pretrained features for a dataset manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--feature-slug")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_manifest(args.manifest)
    rows = _read_rows(args.manifest)
    if args.max_rows:
        rows = rows[: args.max_rows]

    print(f"initializing model={args.model_id} device={args.device}", flush=True)
    extractor = PretrainedExtractor(args.model_id, args.device)
    print(f"initialized model={args.model_id} resolved_device={extractor.device}", flush=True)
    feature_slug = args.feature_slug or _slug(args.model_id)
    feature_root = Path(args.feature_root).expanduser().resolve() / feature_slug
    feature_root.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    pending: list[tuple[dict[str, str], Any, Path]] = []
    processed = 0
    for row in rows:
        item = _load_manifest_item(row)
        if item is None:
            skipped.append({"source_path": row["source_path"], "reason": "unsupported artifact"})
            continue
        modality, payload = item
        if not extractor.supports(modality):
            skipped.append({"source_path": row["source_path"], "reason": f"model does not support {modality}"})
            continue
        out_path = feature_root / f"{_feature_key(row, args.model_id)}.npy"
        if out_path.exists():
            output_rows.append(_feature_row(row, out_path, args.model_id))
            continue
        pending.append((row, (modality, payload), out_path))
        if len(pending) >= args.batch_size:
            output_rows.extend(extractor.extract_batch(pending, args.model_id))
            processed += len(pending)
            print(f"processed={processed} written={len(output_rows)} skipped={len(skipped)}", flush=True)
            pending.clear()

    if pending:
        output_rows.extend(extractor.extract_batch(pending, args.model_id))
        processed += len(pending)
        print(f"processed={processed} written={len(output_rows)} skipped={len(skipped)}", flush=True)

    output_path = Path(args.output_manifest).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(dict.fromkeys(MANIFEST_COLUMNS + ("raw_source_path", "raw_artifact_index", "feature_model")))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "model_id": args.model_id,
        "feature_slug": feature_slug,
        "output_manifest": str(output_path),
        "feature_root": str(feature_root),
        "rows": len(output_rows),
        "skipped": len(skipped),
        "skipped_reasons": _count_reasons(skipped),
    }
    (output_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


class PretrainedExtractor:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        cache_dir = os.environ.get("TRANSFORMERS_CACHE")
        print("loading processor from local cache", flush=True)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=True,
                cache_dir=cache_dir,
            )
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(
                model_id,
                local_files_only=True,
                cache_dir=cache_dir,
            )
        print("loading model from local cache", flush=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            local_files_only=True,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()

    def supports(self, modality: str) -> bool:
        if modality == "image":
            return hasattr(self.processor, "__call__")
        return (
            modality == "text"
            and hasattr(self.model, "get_text_features")
            and (hasattr(self.processor, "tokenizer") or hasattr(self.processor, "current_processor"))
        )

    def extract_batch(self, pending: list[tuple[dict[str, str], tuple[str, Any], Path]], model_id: str) -> list[dict[str, str]]:
        out_rows: list[dict[str, str]] = []
        by_modality: dict[str, list[tuple[dict[str, str], Any, Path]]] = {"image": [], "text": []}
        for row, (modality, payload), out_path in pending:
            by_modality[modality].append((row, payload, out_path))
        for modality, items in by_modality.items():
            if not items:
                continue
            features = self._extract_modality(modality, [payload for _, payload, _ in items])
            for feature, (row, _payload, out_path) in zip(features, items):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, feature.astype(np.float32))
                out_rows.append(_feature_row(row, out_path, model_id))
        return out_rows

    def _extract_modality(self, modality: str, payloads: list[Any]) -> np.ndarray:
        with self.torch.inference_mode():
            if modality == "image":
                try:
                    inputs = self.processor(images=payloads, return_tensors="pt", padding=True)
                except TypeError:
                    inputs = self.processor(images=payloads, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                if hasattr(self.model, "get_image_features"):
                    values = self.model.get_image_features(**inputs)
                else:
                    output = self.model(**inputs)
                    values = _pooled_tensor(output)
            elif modality == "text":
                inputs = self.processor(text=payloads, return_tensors="pt", padding=True, truncation=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                values = self.model.get_text_features(**inputs)
            else:
                raise ValueError(f"unsupported modality: {modality}")
            values = _pooled_tensor(values)
            values = values.detach().float().cpu().numpy()
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-9)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_manifest_item(row: dict[str, str]) -> tuple[str, Any] | None:
    source = Path(row["source_path"]).expanduser().resolve()
    artifact_index = row.get("artifact_index", "").strip()
    if artifact_index:
        return _load_indexed_image(source, int(artifact_index), row.get("artifact_key") or "data")
    suffix = source.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        from PIL import Image

        with Image.open(source) as image:
            return "image", image.convert("RGB")
    if suffix in TEXT_SUFFIXES:
        return "text", source.read_text(encoding="utf-8", errors="replace")
    return None


def _load_indexed_image(path: Path, index: int, key: str) -> tuple[str, Any] | None:
    from PIL import Image

    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    values = payload.get(key)
    if values is None or index < 0 or index >= len(values):
        return None
    array = np.asarray(values[index])
    if array.ndim == 1 and array.size == 3072:
        array = array.reshape(3, 32, 32).transpose(1, 2, 0)
    elif array.ndim == 1:
        return None
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    array = np.asarray(np.clip(array, 0, 255), dtype=np.uint8)
    return "image", Image.fromarray(array).convert("RGB")


def _feature_row(row: dict[str, str], out_path: Path, model_id: str) -> dict[str, str]:
    result = dict(row)
    result["raw_source_path"] = row["source_path"]
    result["raw_artifact_index"] = row.get("artifact_index", "")
    result["source_path"] = str(out_path.resolve())
    result["artifact_index"] = ""
    result["artifact_key"] = ""
    result["feature_model"] = model_id
    return result


def _feature_key(row: dict[str, str], model_id: str) -> str:
    raw = "|".join(
        [
            model_id,
            row["source_path"],
            row.get("artifact_index", ""),
            row.get("artifact_key", ""),
            row.get("label", ""),
            row.get("regime", ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _slug(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def _count_reasons(skipped: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped:
        reason = item["reason"]
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _pooled_tensor(values):
    if hasattr(values, "detach"):
        return values
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        candidate = getattr(values, attr, None)
        if candidate is not None:
            return candidate
    hidden = getattr(values, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    if isinstance(values, (tuple, list)) and values:
        return _pooled_tensor(values[0])
    raise TypeError(f"Could not extract tensor from model output type {type(values).__name__}")

if __name__ == "__main__":
    main()

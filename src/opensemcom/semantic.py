"""Semantic parser, layered encoder, and prototype decoder."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from opensemcom.config import ModelConfig
from opensemcom.types import Array, SemanticLayers, SemanticSample


def softmax(logits: Array) -> Array:
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / max(float(np.sum(exp)), 1e-12)


@dataclass
class PrototypeBook:
    centroids: Array
    class_ids: Array

    def nearest(self, z: Array) -> tuple[int, float]:
        distances = np.linalg.norm(self.centroids - z.reshape(1, -1), axis=1)
        idx = int(np.argmin(distances))
        return int(self.class_ids[idx]), float(distances[idx])


class WorldAwareSemanticParser:
    """Maps raw source vectors to task-aware semantic layers."""

    def __init__(self, config: ModelConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        if config.projection == "identity":
            self.projection = np.eye(config.latent_dim, config.input_dim, dtype=np.float64)
        else:
            self.projection = rng.normal(0.0, 1.0 / np.sqrt(config.input_dim), size=(config.latent_dim, config.input_dim))
        self.task_offsets = {
            task: rng.normal(0.0, 0.05, size=config.latent_dim)
            for task in config.tasks
        }

    def parse(self, sample: SemanticSample) -> SemanticLayers:
        z = self.projection @ sample.x
        z = np.tanh(z + self.task_offsets.get(sample.task, 0.0))
        core_len = max(2, self.config.latent_dim // 2)
        ref_len = max(2, self.config.latent_dim // 4)
        core = z[:core_len]
        refinement = z[core_len : core_len + ref_len]
        evidence = z[-ref_len:]
        fallback = sample.x.astype(np.float64)
        uncertainty_proxy = np.abs(z - np.mean(z))
        voi = uncertainty_proxy / max(float(np.max(uncertainty_proxy)), 1e-9)
        return SemanticLayers(core=core, refinement=refinement, evidence=evidence, fallback=fallback, voi=voi)


class LayeredSemanticEncoder:
    """Converts semantic layers into channel symbols."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def encode(self, layers: SemanticLayers, selected_layers: tuple[str, ...]) -> Array:
        payload = layers.select(selected_layers)
        if payload.size == 0:
            return payload
        norm = np.linalg.norm(payload)
        return payload / max(float(norm), 1e-9)


class PrototypeSemanticDecoder:
    """Simple prototype receiver head that can be replaced by a neural decoder."""

    def __init__(self, config: ModelConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.prototype_book = self._make_prototypes()
        self.adapter_bias = np.zeros(config.latent_dim, dtype=np.float64)
        self.classifier = None
        self.class_ids = np.arange(config.num_known_classes)

    def decode(self, received: Array) -> tuple[int, Array, Array]:
        z = self._expand_to_latent(received) + self.adapter_bias
        if self.classifier is not None:
            raw = self.classifier.predict_proba(z.reshape(1, -1))[0]
            probabilities = np.zeros(self.config.num_known_classes, dtype=np.float64)
            for idx, class_id in enumerate(self.class_ids):
                if 0 <= int(class_id) < probabilities.size:
                    probabilities[int(class_id)] = raw[idx]
            probabilities = probabilities / max(float(np.sum(probabilities)), 1e-12)
        else:
            logits = -np.linalg.norm(self.prototype_book.centroids - z.reshape(1, -1), axis=1)
            probabilities = softmax(logits)
        y_hat = int(np.argmax(probabilities))
        return y_hat, probabilities, z

    def risk(self, samples: list[tuple[Array, int]]) -> float:
        if not samples:
            return 1.0
        errors = 0
        for received, y in samples:
            y_hat, _, _ = self.decode(received)
            errors += int(y_hat != y)
        return errors / len(samples)

    def candidate_with_bias(self, bias_delta: Array) -> "PrototypeSemanticDecoder":
        clone = PrototypeSemanticDecoder(self.config, self.rng)
        clone.prototype_book = self.prototype_book
        clone.adapter_bias = self.adapter_bias + bias_delta
        clone.classifier = self.classifier
        clone.class_ids = self.class_ids
        return clone

    def apply_bias(self, bias_delta: Array) -> None:
        self.adapter_bias = self.adapter_bias + bias_delta

    def set_prototypes(self, centroids: Array) -> None:
        known = centroids[: self.config.num_known_classes]
        normalized = known / np.maximum(np.linalg.norm(known, axis=1, keepdims=True), 1e-9)
        self.prototype_book = PrototypeBook(
            centroids=normalized.astype(np.float64),
            class_ids=np.arange(self.config.num_known_classes),
        )

    def fit_prototypes(self, latents: list[tuple[Array, int] | tuple[Array, int, bool]]) -> None:
        grouped: dict[int, list[Array]] = defaultdict(list)
        for item in latents:
            latent, label, is_open = _unpack_labeled_latent(item)
            if is_open:
                continue
            if 0 <= label < self.config.num_known_classes:
                grouped[label].append(latent)
        if not grouped:
            raise ValueError("Cannot fit receiver prototypes without labeled calibration samples.")
        centroids = self.prototype_book.centroids.copy()
        for label, values in grouped.items():
            centroids[label] = np.mean(values, axis=0)
        centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        self.prototype_book = PrototypeBook(
            centroids=centroids.astype(np.float64),
            class_ids=np.arange(self.config.num_known_classes),
        )
        if self.config.classifier == "logistic":
            self._fit_logistic_head(latents)
        elif self.config.classifier == "torch_mlp":
            self._fit_torch_mlp_head(latents)

    def _fit_logistic_head(self, latents: list[tuple[Array, int] | tuple[Array, int, bool]]) -> None:
        x_values = []
        y_values = []
        for item in latents:
            latent, label, is_open = _unpack_labeled_latent(item)
            if is_open:
                continue
            if 0 <= label < self.config.num_known_classes:
                x_values.append(latent)
                y_values.append(label)
        if len(set(y_values)) < 2:
            return
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError("model.classifier=logistic requires scikit-learn in the scratch environment.") from exc
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=int(self.config.num_known_classes),
            ),
        )
        model.fit(np.asarray(x_values, dtype=np.float64), np.asarray(y_values, dtype=np.int64))
        self.classifier = model
        self.class_ids = np.asarray(model.classes_, dtype=np.int64)

    def _fit_torch_mlp_head(self, latents: list[tuple[Array, int] | tuple[Array, int, bool]]) -> None:
        x_values = []
        y_values = []
        for item in latents:
            latent, label, is_open = _unpack_labeled_latent(item)
            if is_open:
                continue
            if 0 <= label < self.config.num_known_classes:
                x_values.append(latent)
                y_values.append(label)
        if len(set(y_values)) < 2:
            return
        classifier = TorchMLPClassifier(
            input_dim=self.config.latent_dim,
            num_classes=self.config.num_known_classes,
            hidden_dim=self.config.torch_hidden_dim,
            epochs=self.config.torch_epochs,
            lr=self.config.torch_lr,
            device=self.config.torch_device,
            seed=int(self.config.num_known_classes * 100 + self.config.latent_dim),
        )
        classifier.fit(np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.int64))
        self.classifier = classifier
        self.class_ids = np.arange(self.config.num_known_classes, dtype=np.int64)

    def _make_prototypes(self) -> PrototypeBook:
        centroids = self.rng.normal(0.0, 1.0, size=(self.config.num_known_classes, self.config.latent_dim))
        centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        return PrototypeBook(centroids=centroids, class_ids=np.arange(self.config.num_known_classes))

    def _expand_to_latent(self, received: Array) -> Array:
        z = np.zeros(self.config.latent_dim, dtype=np.float64)
        n = min(received.size, z.size)
        z[:n] = received[:n]
        return z


def _unpack_labeled_latent(item: tuple[Array, int] | tuple[Array, int, bool]) -> tuple[Array, int, bool]:
    if len(item) == 2:
        latent, label = item
        return latent, int(label), False
    latent, label, is_open = item
    return latent, int(label), bool(is_open)


class TorchMLPClassifier:
    """Small torch classifier with a scikit-learn-like predict_proba method."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        epochs: int,
        lr: float,
        device: str,
        seed: int,
    ):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("model.classifier=torch_mlp requires torch in the scratch environment.") from exc
        self.torch = torch
        if device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device
        self.device = torch.device(selected_device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.epochs = max(1, int(epochs))
        self.lr = float(lr)
        self.num_classes = int(num_classes)
        hidden_dim = max(16, int(hidden_dim))
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, num_classes),
        ).to(self.device)
        self.mean = None
        self.std = None

    def fit(self, x_values: Array, y_values: np.ndarray) -> None:
        torch = self.torch
        x_np = np.asarray(x_values, dtype=np.float32)
        y_np = np.asarray(y_values, dtype=np.int64)
        self.mean = x_np.mean(axis=0, keepdims=True)
        self.std = np.maximum(x_np.std(axis=0, keepdims=True), 1e-6)
        x_np = (x_np - self.mean) / self.std
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
        counts = np.bincount(y_np, minlength=self.num_classes).astype(np.float32)
        weights = np.sum(counts) / np.maximum(counts * self.num_classes, 1.0)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(991)
        batch_size = min(512, max(32, x.shape[0]))
        self.model.train()
        for _ in range(self.epochs):
            order = torch.randperm(x.shape[0], generator=generator, device=self.device)
            for start in range(0, x.shape[0], batch_size):
                batch = order[start : start + batch_size]
                logits = self.model(x[batch])
                loss = loss_fn(logits, y[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.model.eval()

    def predict_proba(self, x_values: Array) -> Array:
        if self.mean is None or self.std is None:
            raise RuntimeError("TorchMLPClassifier must be fitted before predict_proba.")
        torch = self.torch
        x_np = np.asarray(x_values, dtype=np.float32)
        x_np = (x_np - self.mean) / self.std
        with torch.inference_mode():
            x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
            probabilities = torch.softmax(self.model(x), dim=-1)
        return probabilities.detach().cpu().numpy().astype(np.float64)

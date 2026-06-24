import csv
import pickle

import numpy as np

from opensemcom.benchmark import BenchmarkRegime, OpenSemComBench
from opensemcom.config import OpenSemComConfig
from opensemcom.manifest import validate_manifest


def test_indexed_real_artifact_manifest_loads_distinct_rows(tmp_path):
    artifact = tmp_path / "batch"
    payload = {"data": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8), "labels": [0, 1]}
    with artifact.open("wb") as handle:
        pickle.dump(payload, handle)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "label",
                "task",
                "domain",
                "is_unknown",
                "split",
                "regime",
                "artifact_index",
                "artifact_key",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_path": str(artifact),
                "label": "0",
                "task": "classification",
                "domain": "cifar10",
                "is_unknown": "false",
                "split": "calibration",
                "regime": "closed-id",
                "artifact_index": "0",
                "artifact_key": "data",
            }
        )
        writer.writerow(
            {
                "source_path": str(artifact),
                "label": "1",
                "task": "classification",
                "domain": "cifar10",
                "is_unknown": "false",
                "split": "eval",
                "regime": "closed-id",
                "artifact_index": "1",
                "artifact_key": "data",
            }
        )

    summary = validate_manifest(manifest, require_scratch=False)
    assert summary["rows"] == 2
    bench = OpenSemComBench(OpenSemComConfig(), BenchmarkRegime.CLOSED_ID, manifest)
    sample = bench.samples(1)[0]
    assert sample.y == 1
    assert sample.x.shape == (OpenSemComConfig().model.input_dim,)


def test_validate_manifest_rejects_missing_artifact(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "source_path,label,task,domain,is_unknown,split,regime",
                f"{tmp_path / 'missing.npy'},0,classification,cifar10,false,eval,closed-id",
            ]
        ),
        encoding="utf-8",
    )
    try:
        validate_manifest(manifest, require_scratch=False)
    except FileNotFoundError as exc:
        assert "missing real artifacts" in str(exc)
    else:
        raise AssertionError("missing artifact was not rejected")

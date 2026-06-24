from pathlib import Path

import pytest

from opensemcom.benchmark import BenchmarkRegime
from opensemcom.simulation import run_experiment


def test_experiment_requires_real_dataset_manifest_by_default():
    with pytest.raises(ValueError, match="real dataset manifest"):
        run_experiment(regime=BenchmarkRegime.FULL_OPEN, samples=4, seed=1)


def test_source_and_docs_do_not_reference_generated_data_scaffold():
    forbidden = ("syn" + "thetic").lower()
    scanned_roots = [
        Path("README.md"),
        Path("src"),
        Path("examples"),
    ]
    offenders = []
    for root in scanned_roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.is_file() and path.suffix in {".md", ".py"}:
                text = path.read_text(encoding="utf-8").lower()
                if forbidden in text:
                    offenders.append(str(path))
    assert offenders == []

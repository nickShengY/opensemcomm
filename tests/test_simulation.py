from opensemcom.benchmark import BenchmarkRegime
from opensemcom.simulation import run_experiment


def write_manifest(tmp_path):
    plan = (tmp_path.cwd() / "OpenSemCom_Research_Plan.md").resolve()
    readme = (tmp_path.cwd() / "README.md").resolve()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "source_path,label,task,domain,is_unknown,split,regime",
                f"{plan},0,classification,paper,false,calibration,closed-id",
                f"{readme},1,classification,docs,false,calibration,closed-id",
                f"{plan},0,classification,paper,false,eval,closed-id",
                f"{readme},1,classification,docs,false,eval,closed-id",
                f"{plan},6,hazard,paper,true,eval,full-open",
                f"{readme},1,retrieval,docs,false,eval,full-open",
                f"{plan},0,classification,paper,false,eval,full-open",
                f"{readme},6,hazard,docs,true,eval,full-open",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def test_full_open_experiment_runs_and_reports_main_metrics(tmp_path):
    manifest = write_manifest(tmp_path)
    result = run_experiment(
        regime=BenchmarkRegime.FULL_OPEN,
        samples=4,
        calibration_samples=2,
        users=2,
        seed=3,
        dataset_manifest=str(manifest),
    )
    assert "open_semantic_risk" in result.metrics
    assert "open_semantic_outage" in result.metrics
    assert "semantic_goodput" in result.metrics
    assert sum(result.decisions.values()) == 4


def test_closed_id_experiment_has_traces(tmp_path):
    manifest = write_manifest(tmp_path)
    result = run_experiment(
        regime=BenchmarkRegime.CLOSED_ID,
        samples=2,
        calibration_samples=2,
        seed=4,
        dataset_manifest=str(manifest),
    )
    assert len(result.traces) == 2
    assert all("decision" in trace for trace in result.traces)


def test_manifest_with_utf8_bom_runs_from_windows_tools(tmp_path):
    manifest = write_manifest(tmp_path)
    manifest.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8-sig")
    result = run_experiment(
        regime=BenchmarkRegime.CLOSED_ID,
        samples=2,
        calibration_samples=2,
        seed=4,
        dataset_manifest=str(manifest),
    )
    assert len(result.traces) == 2

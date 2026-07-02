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
    assert all("harq_refinement_rounds" in trace for trace in result.traces)
    assert all("harq_transmissions" in trace for trace in result.traces)
    assert all("harq_hit_max_refinements" in trace for trace in result.traces)


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

def test_experiment_reports_resource_usage_metrics(tmp_path):
    manifest = write_manifest(tmp_path)
    result = run_experiment(
        regime=BenchmarkRegime.FULL_OPEN,
        samples=4,
        calibration_samples=2,
        users=2,
        seed=3,
        dataset_manifest=str(manifest),
    )
    for key in (
        "total_bandwidth",
        "avg_bandwidth",
        "bandwidth_per_accepted",
        "bandwidth_per_correct_accepted",
        "goodput_per_bandwidth",
        "total_resource_cost",
        "avg_resource_cost",
        "total_latency",
        "avg_latency",
        "total_repetitions",
        "avg_repetitions",
        "total_harq_refinement_rounds",
        "avg_harq_refinement_rounds",
        "harq_refined_sample_rate",
        "total_harq_transmissions",
        "avg_harq_transmissions",
        "total_harq_full_payload_rounds",
        "avg_harq_full_payload_rounds",
        "harq_full_payload_sample_rate",
        "harq_hit_max_refinements_rate",
    ):
        assert key in result.metrics
    assert result.metrics["total_bandwidth"] >= 0.0
    assert result.metrics["total_resource_cost"] >= 0.0
    assert result.metrics["total_repetitions"] >= 4.0
    assert result.metrics["total_harq_transmissions"] >= 4.0
    assert 0.0 <= result.metrics["harq_refined_sample_rate"] <= 1.0
    assert 0.0 <= result.metrics["harq_hit_max_refinements_rate"] <= 1.0

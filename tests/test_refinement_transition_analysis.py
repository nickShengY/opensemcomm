import csv
import json

from opensemcom.cli.analyze_refinement_transitions import write_refinement_diagnostics


def test_transition_analyzer_writes_raw_and_terminal_summaries(tmp_path):
    traces = [
        {
            "run_identifier": "closed-id",
            "configuration_identifier": "config.yaml",
            "seed": 7,
            "evaluation_condition": "closed-id",
            "stage_aware_thresholds": True,
            "sample_key": "sample-a",
            "refinement_transitions": [
                {
                    "refinement_round": 1,
                    "current_stage_index": 0,
                    "next_stage_index": 1,
                    "current_stage": "core",
                    "next_stage": "core+refinement",
                    "transition_type": "semantic_expansion_0_to_1",
                    "current_risk_score": 0.7,
                    "next_risk_score": 0.4,
                    "current_q_accept": 0.3,
                    "next_q_accept": 0.2,
                    "current_q_refine": 0.8,
                    "next_q_refine": 0.6,
                    "raw_score_change": 0.3,
                    "current_acceptance_margin": 0.4,
                    "next_acceptance_margin": 0.2,
                    "margin_change": 0.2,
                    "current_action": "refine",
                    "next_action": "accept",
                    "final_terminal_action": "accept",
                    "final_semantic_stage": "core+refinement",
                    "terminal_outcome": "correct-supported accepted",
                }
            ],
        }
    ]
    trace_path = tmp_path / "traces.json"
    trace_path.write_text(json.dumps(traces), encoding="utf-8")

    paths = write_refinement_diagnostics([trace_path], tmp_path / "analysis")

    with paths["raw_csv"].open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    with paths["summary_csv"].open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with paths["summary_by_terminal_csv"].open(newline="", encoding="utf-8") as handle:
        terminal_rows = list(csv.DictReader(handle))
    assert raw_rows[0]["sample_key"] == "sample-a"
    assert raw_rows[0]["margin_change"] == "0.2"
    assert summary_rows[0]["raw_positive_change_rate"] == "1.0"
    assert terminal_rows[0]["terminal_group"] == "accepted"
    assert "Stage-aware thresholds recorded in traces: yes" in paths["report"].read_text(encoding="utf-8")

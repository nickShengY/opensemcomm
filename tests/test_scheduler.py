"""Scheduler selection tests."""

from opensemcom.cli.run_config import _config_from_dict
from opensemcom.config import ResourceWeights
from opensemcom.scheduler import RiskAwareScheduler
from opensemcom.types import ResourceBudget


def test_default_scheduler_prefers_full_payload_for_default_codec():
    scheduler = RiskAwareScheduler(ResourceBudget(), ResourceWeights())

    action = scheduler.schedule(base_risk=0.5)

    assert action.layers == ("core", "refinement", "evidence")


def test_configured_resource_penalty_prefers_core_payload():
    scheduler = RiskAwareScheduler(
        ResourceBudget(),
        ResourceWeights(scheduler_resource_penalty=0.60),
    )

    action = scheduler.schedule(base_risk=0.5)

    assert action.layers == ("core",)

def test_yaml_resource_penalty_is_loaded_into_scheduler_config():
    config = _config_from_dict({"resource_weights": {"scheduler_resource_penalty": 0.60}})

    action = RiskAwareScheduler(config.resource_budget, config.resource_weights).schedule(base_risk=0.5)

    assert action.layers == ("core",)
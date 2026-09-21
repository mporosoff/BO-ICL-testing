"""Main-view random controls preserve independent effort and graph compatibility."""

import json
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from boicl.campaign import CampaignService
from boicl.campaign_controls import (
    random_control_state,
    record_random_control,
    start_random_control,
)
from boicl.toolkit_bridge import project


MASS = {"quantification_method": "gsas_ii_mass_fraction", "normalization": "all phases"}


@pytest.fixture
def campaign(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Graph and random-control regressions must not call an engine")

    service = CampaignService(tmp_path, runner=forbidden)
    cid = service.create_generic(
        [
            {
                "candidate_id": f"c{i}",
                "procedure": f"Recipe {i}",
                "x": i,
                "value": 10 * (i + 1) if i < 2 else None,
            }
            for i in range(5)
        ],
        [{"column": "x"}],
        bounds=[0, 100],
        overrides={"auto_suggest": False, "new_measurement_budget": 1},
    )
    legacy = SimpleNamespace(to_json=lambda: {}, list_campaigns=lambda: [])
    return service, cid, legacy


def revise(service, cid, policy):
    service.revise_measurement_definition(
        cid, MASS, policy, "Documented scientific cohort decision for this test"
    )


def test_excluded_random_result_still_fills_budget_and_retains_main_graph_markers(
    campaign,
):
    service, cid, legacy = campaign
    walk = start_random_control(service, cid)
    control = walk["campaign_id"]
    record_random_control(
        service,
        cid,
        {"suggestion_id": walk["current_candidate"]["suggestion_id"], "value": 50},
    )
    for arm in (cid, control):
        revise(service, arm, "exclude")
    before = {arm: service.export(arm) for arm in (cid, control)}

    walk = random_control_state(service, cid)
    assert walk["status"] == "complete"
    assert walk["completed_count"] == walk["target_count"] == 1
    assert walk["observations"] == []
    payload = project(service, cid, legacy)
    projected = payload["live_random_walk"]
    assert projected["completed_count"] == 1
    assert projected["initialization_count"] == 2
    assert projected["plot_observations"] == []
    excluded = projected["excluded_measurement_points"]
    assert [row["axis_label"] for row in excluded] == ["i1", "i2", "1"]
    assert excluded[-1]["index"] == 3
    assert all("value" not in row for row in excluded)
    assert payload["plot_x_axis"]["labels"][-1]["index"] == 3
    assert payload["comparison_diagnostics"] == []
    assert {arm: service.export(arm) for arm in before} == before


def test_parent_cohort_revision_hides_incompatible_overlay_without_losing_control(
    campaign,
):
    service, cid, legacy = campaign
    walk = start_random_control(service, cid)
    control = walk["campaign_id"]
    revise(service, cid, "exclude")
    before = {arm: service.export(arm) for arm in (cid, control)}

    payload = project(service, cid, legacy)
    projected = payload["live_random_walk"]
    assert payload["shared_control_id"] == control
    assert projected["current_candidate"] == walk["current_candidate"]
    assert len(projected["initialization_observations"]) == 2
    assert projected["plot_observations"] == []
    assert payload["live_random_walk_trace"] == []
    assert not projected.get("excluded_measurement_points")
    diagnostic = payload["comparison_diagnostics"][0]
    assert diagnostic["campaign_id"] == control
    assert diagnostic["kind"] == "random_control"
    assert not diagnostic["compatible"]
    assert "effective_initialization" in diagnostic["mismatches"]
    assert {arm: service.export(arm) for arm in before} == before

    # Restoring matching scientific inclusion re-enables the same independent arm.
    for arm in (cid, control):
        revise(service, arm, "retain_with_justification")
    payload = project(service, cid, legacy)
    projected = payload["live_random_walk"]
    assert projected["campaign_id"] == control
    assert projected["observations"] == []  # control form counts new results only
    assert [row["axis_label"] for row in projected["plot_observations"]] == ["i1", "i2"]
    assert len(payload["live_random_walk_trace"]) == 2
    assert payload["comparison_diagnostics"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
@pytest.mark.parametrize(
    "included,completed,status,expected_progress,complete",
    [
        (1, 2, "complete", "2/2", True),
        (0, 2, "complete", "2/2", True),
        (1, None, "idle", "1/2", False),
        (2, None, "complete", "2/2", True),
    ],
)
def test_actual_random_control_renderer_uses_physical_count_with_legacy_fallback(
    included, completed, status, expected_progress, complete
):
    from boicl.local_app import INDEX_HTML

    source = re.search(
        r"function renderLiveRandomWalk\(\)\s*\{.*?(?=\n\s*function renderProgress)",
        INDEX_HTML,
        re.S,
    )
    assert source is not None, "Execute the production random-control renderer"
    walk = {
        "observations": [{"value": 10}] * included,
        "target_count": 2,
        "status": status,
    }
    if completed is not None:
        walk["completed_count"] = completed
    script = r"""
const fs=require('fs'),vm=require('vm');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const nodes={randomWalkTarget:{value:2},randomWalkProgress:{value:''},
 randomWalkCandidate:{textContent:''},addRandomWalkResult:{disabled:false}};
const context={state:input.state,busy:false,$:id=>{
 if(!Object.hasOwn(nodes,id))throw Error('Unexpected DOM access: '+id);
 return nodes[id];}};
vm.createContext(context);vm.runInContext(input.source,context);
context.renderLiveRandomWalk();process.stdout.write(JSON.stringify(nodes));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script],
        input=json.dumps(
            {
                "source": source.group(),
                "state": {"live_random_walk": walk, "observations": []},
            }
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = json.loads(result.stdout)
    assert nodes["randomWalkProgress"]["value"] == expected_progress
    assert (
        nodes["randomWalkCandidate"]["textContent"] == "Random walk complete."
    ) is complete
    assert nodes["addRandomWalkResult"]["disabled"] is True

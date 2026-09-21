"""Generic text engines preserve the named objective and bounded demo units."""
import pytest

from boicl.campaign import CampaignService
from boicl.moc_demo import demo_runner
from boicl.campaign_config import resolve_config
from boicl.toolkit_bridge import config_changes, legacy_config


@pytest.mark.parametrize("engine", ["llm", "gpr_embeddings"])
@pytest.mark.parametrize(
    "direction,bounds,values",
    [("maximize", [-10, 10], [-4, 0]), ("minimize", [0, 50], [30, 20])],
)
def test_text_only_generic_demo_uses_original_objective(
    tmp_path, engine, direction, bounds, values
):
    service = CampaignService(tmp_path, runner=demo_runner)
    records = [
        dict(
            candidate_id=f"C{i}",
            procedure=f"Set controllable synthesis knob to {i}",
            cost=values[i] if i < 2 else None,
        )
        for i in range(4)
    ]
    cid = service.create_generic(
        records,
        [],
        objective="cost",
        direction=direction,
        bounds=bounds,
        preset="generic_llm" if engine == "llm" else "generic_embedding_gp",
        synthetic_demo=True,
        overrides={"auto_suggest": False},
    )
    service.start_suggestion(cid, background=False)
    row = service.get(cid)["suggestions"][-1]
    assert row["status"] == "suggested", row.get("selection_reason")
    assert bounds[0] <= row["prediction"]["mean"] <= bounds[1]
    assert row["candidate_id"] in {"C2", "C3"}
    if engine == "llm":
        result = row["engine_result"]
        expected_direction = 1 if direction == "maximize" else -1
        assert result["target"]["direction"] == expected_direction
        for call in result["request_log"]:
            messages = call["request"]["messages"]
            assert "cost" in messages[0]["content"]
            assert "MoC" not in str(messages)
        assert (
            service.replay(cid, row["suggestion_id"])["selected_candidate_id"]
            == row["candidate_id"]
        )


@pytest.mark.parametrize(
    "direction,first,expected", [("maximize", -2, 0.7), ("minimize", 2, -0.7)]
)
def test_unbounded_zero_baseline_scale_roundtrips_and_controls_target(
    tmp_path, direction, first, expected
):
    assert resolve_config("generic_llm")["llm"]["reference_scale"] == 1
    assert resolve_config("moc_llm")["llm"]["reference_scale"] == 100
    service = CampaignService(tmp_path, runner=demo_runner)
    records = [
        dict(candidate_id=f"C{i}", procedure=f"Set knob to {i}", score=value)
        for i, value in enumerate([first, 0, None])
    ]
    cid = service.create_generic(
        records,
        [],
        objective="score",
        direction=direction,
        bounds=None,
        preset="generic_llm",
        synthetic_demo=True,
        overrides={"auto_suggest": False},
    )
    config = service.get(cid)["config"]
    assert legacy_config(config)["inverse_target_reference_scale"] == 1
    changes = config_changes(
        {"inverse_target_reference_scale": 3.5, "inverse_target_jitter": 0}, config
    )
    service.update_config(cid, changes)
    restored = CampaignService(tmp_path, runner=demo_runner)
    assert (
        legacy_config(restored.get(cid)["config"])["inverse_target_reference_scale"]
        == 3.5
    )
    with pytest.raises(ValueError, match="reference_scale"):
        restored.update_config(
            cid,
            config_changes(
                {"inverse_target_reference_scale": 0}, restored.get(cid)["config"]
            ),
        )
    restored.start_suggestion(cid, background=False)
    row = restored.get(cid)["suggestions"][-1]
    assert row["status"] == "suggested"
    target = row["engine_result"]["target"]
    assert target["best"] == 0
    assert target["reference_scale"] == 3.5
    assert target["resolved_target"] == pytest.approx(expected)


def test_bounded_generic_zero_scale_uses_objective_width():
    config = resolve_config(
        "generic_llm", {"bounds": [-5, 10], "llm": {"reference_scale": 123}}
    )
    assert legacy_config(config)["inverse_target_reference_scale"] == 15
    changes = config_changes({"inverse_target_reference_scale": 999}, config)
    assert (
        resolve_config(
            "generic_llm", {**config, "llm": {**config["llm"], **changes["llm"]}}
        )["llm"]["reference_scale"]
        == 15
    )

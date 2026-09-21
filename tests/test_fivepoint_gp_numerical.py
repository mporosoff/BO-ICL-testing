"""Exercise the study preset with its real full-pool production GP settings."""

import socket

import pytest

from boicl.campaign import CampaignService
from boicl.campaign_config import resolve_config


def test_five_point_first_recommendation_is_full_pool_transformed_ei(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("The synthesis-parameter GP must not access a provider or cache")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("boicl.embedding_cache.create_embedding_cache", forbidden)
    service = CampaignService(tmp_path)
    cid = service.create("moc_five_gp")
    data = service.get(cid)
    config = data["config"]
    source = resolve_config("moc_gp")["structured_gp"]
    assert config["structured_gp"] == {
        **source,
        "ei_after_unique_measured_designs": 3,
    }
    assert config["new_measurement_budget"] == 5
    assert config["auto_suggest"] is False
    assert len(data["candidates"]) == 7776
    assert [o["moc_wt_pct"] for o in service.active(data)] == [72.1, 83.8, 23.4]

    job = service.start_suggestion(cid, background=False)
    assert job["status"] == "suggested", job
    saved = service.get(cid)
    suggestion = saved["suggestions"][-1]
    result = suggestion["engine_result"]
    assert result["stage"] == "expected_improvement"
    assert result["acquisition_units"] == "standardized padded-logit units"
    assert result["unique_design_count"] == 3
    assert len(result["scores"]) == 7773
    assert result["diagnostics"]["burn_in"] == 1000
    assert result["diagnostics"]["retained_draws"] == 4000
    assert config["structured_gp"]["ei_xi_standardized_logit"] == 0.01
    assert result["score"] > 0
    assert 0 <= suggestion["prediction"]["lower95"]
    assert suggestion["prediction"]["lower95"] <= suggestion["prediction"]["mean"]
    assert (
        suggestion["prediction"]["mean"] <= suggestion["prediction"]["upper95"] <= 100
    )
    assert len(service.active(saved)) == 3
    assert service.completed(saved) == 0
    assert service.summary(cid)["counts"]["pending"] == 0

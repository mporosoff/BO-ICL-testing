"""Offline request previews and isolated inverse proposals share live payloads."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from boicl.campaign_config import resolve_config
from boicl.llm_engine import (
    LLMEngine,
    build_chat_request,
    preview_request,
    score_responses,
)
from boicl.aqfxns import upper_confidence_bound
from boicl.llm_model import make_dd


def records():
    candidates = [
        dict(candidate_id=key, procedure=f"recipe {key}", hidden_label=987654)
        for key in "ABCDE"
    ]
    observations = [
        dict(
            candidate_id=key,
            observation_id=f"obs{key}",
            procedure=f"recipe {key}",
            value=value,
        )
        for key, value in [("A", 20), ("B", 40)]
    ]
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.8, 0.6], [-1, 0]])
    return candidates, observations, vectors


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(deepcopy(request))
        raw = next(self.responses)
        return dict(
            choices=[dict(message=dict(content=value)) for value in raw],
            model=request["model"],
        )


@pytest.mark.parametrize("selector_mode", ["nearest", "all"])
def test_current_preview_exactly_matches_executed_requests_and_preserves_inputs(
    selector_mode,
):
    candidates, observations, vectors = records()
    client = Client([["inverse recipe"], ["50"] * 5])
    engine = LLMEngine(
        dict(
            selector_mode=selector_mode,
            selector_k=1,
            shortlist_size=1,
            inverse_system_message="custom inverse\n",
            forward_system_message="",
        ),
        client,
    )
    before = deepcopy(engine.rng.bit_generator.state)
    inv = preview_request(
        engine.config,
        candidates,
        observations,
        role="inverse",
        candidate_vectors=vectors,
        rng_state=before,
    )
    forward = preview_request(
        engine.config,
        candidates,
        observations,
        candidate_id="C",
        candidate_vectors=vectors,
        rng_state=before,
    )
    assert inv["status"] == forward["status"] == "exact"
    assert engine.rng.bit_generator.state == before
    assert not client.requests
    result = engine.suggest(
        candidates,
        observations,
        candidate_vectors=vectors,
        query_embedder=lambda _: vectors[2],
    )
    assert result["selected_candidate_id"] == "C"
    assert inv["request"] == client.requests[0]
    assert forward["request"] == client.requests[1]
    assert inv["example_ids"] == result["inverse_example_ids"]
    assert forward["example_ids"] == result["predictions"]["C"]["example_ids"]
    assert forward["request"]["messages"][0]["content"] == ""
    assert "987654" not in str(inv) + str(forward)
    assert inv["target"] == result["target"]


def test_unresolved_preview_shows_effective_payload_parts_without_any_provider(
    monkeypatch,
):
    candidates, observations, _ = records()

    def forbidden(*args, **kwargs):
        pytest.fail("Preview may not call or construct a provider")

    monkeypatch.setattr(LLMEngine, "_embed", forbidden)
    monkeypatch.setattr(LLMEngine, "_chat", forbidden)
    preview = preview_request(
        dict(forward_system_message="my exact system", manual_inverse_target=0),
        candidates,
        observations,
        candidate_id="C",
    )
    assert preview["status"] == "unresolved" and preview["request"] is None
    assert "cached procedure embeddings" in preview["reason"]
    assert preview["system_message"] == "my exact system"
    assert "recipe C" in preview["query_message"]
    assert preview["request_parameters"] == dict(
        model="gpt-4o", n=5, temperature=0.7, max_tokens=256
    )
    assert preview["example_selection"]["resolved"] is False
    assert preview["target"]["resolved_target"] == 0
    exact = preview_request(
        dict(selector_mode="all"), candidates, observations, candidate_id="C"
    )
    assert exact["status"] == "exact"


def test_partial_cache_and_distinct_selector_model_are_not_confused():
    candidates, observations, vectors = records()
    partial = {
        row["candidate_id"]: vector for row, vector in zip(candidates[:3], vectors[:3])
    }
    assert (
        preview_request(
            {}, candidates, observations, candidate_id="C", candidate_vectors=partial
        )["status"]
        == "exact"
    )
    config = dict(selector_embedding_model="different-selector-model")
    missing = preview_request(
        config, candidates, observations, candidate_id="C", candidate_vectors=partial
    )
    assert missing["status"] == "unresolved"
    assert (
        preview_request(
            config,
            candidates,
            observations,
            candidate_id="C",
            selector_candidate_vectors=partial,
        )["status"]
        == "exact"
    )


def test_preview_without_selected_candidate_retains_settings_and_honest_dependency():
    candidates, observations, _ = records()
    preview = preview_request(
        dict(forward_system_message="effective system"), candidates, observations
    )
    assert preview["status"] == "unresolved" and preview["request"] is None
    assert "shortlist depends" in preview["reason"]
    assert preview["system_message"] == "effective system"
    assert preview["request_parameters"]["n"] == 5
    assert preview["query_message"] is None
    # All-example previews do not validate irrelevant or unavailable cache data.
    exact = preview_request(
        dict(selector_mode="all"),
        candidates,
        observations,
        candidate_id="C",
        candidate_vectors=[],
    )
    assert exact["status"] == "exact"


def test_recorded_preview_uses_original_request_despite_new_settings_and_failed_responses():
    candidates, observations, vectors = records()
    client = Client([["query"], ["invalid"] * 5])
    result = LLMEngine(dict(shortlist_size=1), client).suggest(
        candidates,
        observations,
        candidate_vectors=vectors,
        query_embedder=lambda _: vectors[2],
    )
    assert result["status"] == "failed"
    changed = dict(
        forward_model="gpt-4o-mini", forward_system_message="NEW", selector_mode="all"
    )
    preview = preview_request(
        changed, role="forward", candidate_id="C", recorded_result=result
    )
    assert preview["source"] == "recorded" and preview["status"] == "exact"
    assert preview["request"] == client.requests[1]
    assert preview["example_ids"] == result["predictions"]["C"]["example_ids"]
    assert (
        preview_request({}, role="forward", candidate_id="D", recorded_result=result)[
            "status"
        ]
        == "unresolved"
    )


@pytest.mark.parametrize("direction,manual", [("maximize", 0), ("minimize", -2)])
def test_standalone_count_and_manual_raw_target_do_not_change_bo_count_or_inputs(
    direction, manual
):
    candidates, observations, vectors = records()
    observations[0]["value"], observations[1]["value"] = -5, -3
    original = deepcopy((candidates, observations))
    config = resolve_config(
        "generic_llm",
        dict(
            objective="response",
            units="units",
            direction=direction,
            bounds=[-10, 10],
            llm=dict(
                manual_inverse_target=manual,
                inverse_proposal_count=3,
                inverse_multiplier=10,
                inverse_jitter=2,
            ),
        ),
    )
    client = Client(
        [["proposal one", "proposal two", "proposal three"], ["BO query"], ["-1"] * 5]
    )
    engine = LLMEngine(config, client)
    before = deepcopy(engine.rng.bit_generator.state)
    preview = preview_request(
        config,
        candidates,
        observations,
        role="inverse",
        count=3,
        candidate_vectors=vectors,
        rng_state=before,
    )
    standalone = engine.propose_inverse(
        candidates, observations, count=3, candidate_vectors=vectors
    )
    assert standalone["status"] == "proposed"
    assert standalone["procedures"] == [
        "proposal one",
        "proposal two",
        "proposal three",
    ]
    assert standalone["requested_count"] == standalone["returned_count"] == 3
    assert standalone["target"]["resolved_target"] == manual
    assert standalone["target"]["multiplier_draw"] is None
    assert standalone["rng_state_after"] == before
    assert standalone["request_log"][0]["request"] == preview["request"]
    assert (
        f"Target response (units): {manual:g}"
        in client.requests[0]["messages"][1]["content"]
    )
    assert engine.config["inverse_n"] == 1
    assert (candidates, observations) == original
    # A separate ordinary BO step still requests exactly one inverse completion.
    engine.config["shortlist_size"] = 1
    result = engine.suggest(
        candidates,
        observations,
        candidate_vectors=vectors,
        query_embedder=lambda _: vectors[2],
    )
    assert result["status"] == "suggested"
    assert [r["n"] for r in client.requests] == [3, 1, 5]


@pytest.mark.parametrize("target", [-1, 101, float("nan")])
def test_invalid_explicit_target_never_calls_provider(target):
    candidates, observations, vectors = records()
    client = Client([])
    result = LLMEngine(client=client).propose_inverse(
        candidates, observations, target=target, candidate_vectors=vectors
    )
    assert result["status"] == "failed" and not client.requests


@pytest.mark.parametrize("count", [0, 21, True, 1.5])
def test_invalid_standalone_count_never_calls_provider(count):
    candidates, observations, _ = records()
    client = Client([])
    result = LLMEngine(client=client).propose_inverse(
        candidates, observations, count=count
    )
    assert result["status"] == "failed" and not client.requests


def test_standalone_cancellation_and_incomplete_response_are_explicit():
    candidates, observations, _ = records()
    client = Client([["only one"]])
    engine = LLMEngine(dict(selector_mode="all"), client)
    cancelled = engine.propose_inverse(candidates, observations, cancel=lambda: True)
    assert cancelled["status"] == "cancelled" and not client.requests
    incomplete = engine.propose_inverse(candidates, observations, count=2)
    assert incomplete["status"] == "failed" and incomplete["raw_responses"] == [
        "only one"
    ]
    assert incomplete["returned_count"] == 1


@pytest.mark.parametrize("maximize", [True, False])
def test_shared_engine_ucb_still_matches_direction_correct_public_score(maximize):
    config = resolve_config(
        "generic_llm",
        dict(
            direction="maximize" if maximize else "minimize",
            llm=dict(acquisition="upper_confidence_bound"),
        ),
    )["llm"]
    values = [-8, -4, -7, -7, -9]
    scored = score_responses("candidate", [str(v) for v in values], -6, config)
    assert scored["mean"] == pytest.approx(np.mean(values))
    assert scored["acquisition"] == pytest.approx(
        upper_confidence_bound(make_dd(values, [0.2] * 5), -6, 0.5, maximize)
    )

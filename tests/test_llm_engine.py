from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from boicl.aqfxns import expected_improvement, probability_of_improvement
from boicl.campaign_config import resolve_config
from boicl.llm_engine import (
    LLMEngine,
    PROMPT_PATH,
    crystal_embedding_input,
    render_messages,
    replay_step,
    resolve_inverse_target,
    retrieve_candidates,
    score_responses,
)
from boicl.llm_model import DiscreteDist, make_dd, scale_distribution


@pytest.mark.parametrize(
    "best,maximize,target",
    [(40, True, 48), (40, False, 32), (-40, True, -32), (-40, False, -48)],
)
def test_d1_raw_direction(best, maximize, target):
    assert resolve_inverse_target(best, maximize=maximize, jitter=0)[
        "resolved_target"
    ] == pytest.approx(target)


def test_d1_bounds_zero_and_manual_validation():
    target = resolve_inverse_target(96.5, jitter=0, bounds=(0, 100))
    assert target["raw_target"] == pytest.approx(115.8)
    assert target["resolved_target"] == 100
    assert target["bounds_applied"]
    assert resolve_inverse_target(100, bounds=(0, 100))["saturated"]
    assert resolve_inverse_target(0, jitter=0, bounds=(0, 100), reference_scale=100)[
        "resolved_target"
    ] == pytest.approx(20)
    with pytest.raises(ValueError, match="reference scale"):
        resolve_inverse_target(0)
    with pytest.raises(ValueError, match="Manual"):
        resolve_inverse_target(40, manual=101, bounds=(0, 100))
    with pytest.raises(ValueError, match="conflicts"):
        resolve_inverse_target(40, floor=110, bounds=(0, 100))
    assert resolve_inverse_target(40, multiplier=0.5, jitter=0)["resolved_target"] == 40


def test_d2_empirical_identity_agreement_and_duplicate_mass():
    assert np.std([86.7, 91.5, 0, 96.5, 91.8]) == pytest.approx(36.7809189662)
    agreement = make_dd([96.5] * 5, [0.2] * 5)
    assert isinstance(agreement, DiscreteDist)
    assert agreement.std() == 0
    assert expected_improvement(agreement, 96.5) == 0
    dist = make_dd([80, 90, 100, 90, 90], [0.2] * 5)
    assert scale_distribution(dist, 1) is dist
    assert dist.mean() == pytest.approx(90)
    assert dist.std() == pytest.approx(np.sqrt(40))
    assert expected_improvement(dist, 96.5) == pytest.approx(0.7)
    assert probability_of_improvement(dist, 96.5) == pytest.approx(0.2)
    assert scale_distribution(dist, 0).std() == 0
    expanded = scale_distribution(dist, 3, (0, 100))
    assert max(expanded.values) == 100
    with pytest.raises(ValueError):
        scale_distribution(dist, -1)


def test_d2_invalid_partial_samples_keep_reasons_and_zero():
    config = resolve_config()["llm"]
    scored = score_responses(
        "a", ["0", "100", "150", "heat at 600 °C", "1e999"], 96.5, config
    )
    assert scored["accepted_values"] == [0, 100]
    assert scored["accepted_samples"] == 2
    assert scored["partial_samples"] and scored["status"] == "scored"
    assert len(scored["rejected_responses"]) == 3
    assert (
        score_responses("a", ["0", "bad"], 96.5, config)["status"]
        == "insufficient_samples"
    )


def fixture_rows():
    candidates = [
        dict(candidate_id=key, procedure=f"procedure {key}", hidden_label=99)
        for key in "ABCDE"
    ]
    observations = [
        dict(
            candidate_id=key,
            observation_id=f"obs{key}",
            procedure=f"procedure {key}",
            value=value,
        )
        for key, value in [("A", 72.1), ("B", 83.8)]
    ]
    vectors = np.array(
        [[0, 1], [0, 1], [1, 0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32
    )
    return candidates, observations, vectors


class MockClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        raw = next(self.responses)
        return dict(
            model="gpt-4o-test-snapshot",
            choices=[dict(message=dict(content=text)) for text in raw],
            usage=dict(total_tokens=20),
        )


def test_d3_exact_prompt_provenance_and_no_candidate_label():
    manifest = json.loads(
        (PROMPT_PATH / "manifest.json").read_text(encoding="utf-8-sig")
    )
    for filename, digest in manifest["sha256"].items():
        assert (
            hashlib.sha256((PROMPT_PATH / filename).read_bytes()).hexdigest() == digest
        )
    _, observations, _ = fixture_rows()
    messages = render_messages(
        "forward", observations, "candidate recipe", system_message="custom  text\n"
    )
    assert messages[0]["content"] == "custom  text\n"
    assert "Measured cubic MoC weight fraction (%): 83.8" in messages[1]["content"]
    candidate_part = messages[1]["content"].split("Candidate synthesis procedure:")[1]
    assert "83.8" not in candidate_part and "hidden_label" not in candidate_part
    assert crystal_embedding_input("body °C\n ") == "experimental procedure: body °C\n "


def test_d4_nearest100_precedes_mmr_and_stable_ties():
    ids = [f"c{i:03}" for i in range(102)]
    vectors = [[1, 0.01 * i] for i in range(102)]
    records = retrieve_candidates(
        ids, vectors, [1, 0], fetch_k=100, shortlist_size=16, mmr_lambda=0
    )
    assert len(records) == 16
    assert all(row["candidate_id"] not in {"c100", "c101"} for row in records)
    tied = retrieve_candidates(["z", "a", "b"], [[1, 0]] * 3, [1, 0], shortlist_size=2)
    assert [row["candidate_id"] for row in tied] == ["a", "b"]


def test_d5_d6_mocked_step_ownership_exclusions_requests_and_replay():
    candidates, observations, vectors = fixture_rows()
    client = MockClient(
        [["unlabeled procedure"], ["bad"] * 5, ["80", "90", "100", "90", "90"]]
    )
    engine = LLMEngine(
        dict(shortlist_size=3, forward_system_message="edited prompt"), client
    )
    result = engine.suggest(
        candidates,
        observations,
        excluded_ids=["E"],
        candidate_vectors=vectors,
        query_embedder=lambda _: [1, 0],
    )
    assert result["status"] == "suggested"
    assert result["selected_candidate_id"] == "D"
    assert set(result["predictions"]) == {"C", "D"}
    assert result["predictions"]["C"]["status"] == "insufficient_samples"
    assert result["predictions"]["D"]["mean"] == 90
    assert [request["n"] for request in client.requests] == [1, 5, 5]
    assert all(
        "top_p" not in request
        and "best_of" not in request
        and "logprobs" not in request
        for request in client.requests
    )
    assert result["prompt_provenance"]["forward"]["origin"] == "custom"
    assert (
        result["request_log"][1]["request"]["messages"][0]["content"] == "edited prompt"
    )
    before = len(client.requests)
    replay = replay_step(result)
    assert replay["replay_verified"] and replay["replayed_selected_candidate_id"] == "D"
    assert len(client.requests) == before
    changed = deepcopy(result)
    changed["predictions"]["C"]["raw_responses"] = ["100"] * 5
    assert not replay_step(changed)["replay_verified"]


def test_failed_scores_never_fabricate_random_result_and_cancel_no_requests():
    candidates, observations, vectors = fixture_rows()
    client = MockClient([["query"], ["bad"] * 5])
    result = LLMEngine(dict(shortlist_size=1), client).suggest(
        candidates,
        observations,
        candidate_vectors=vectors,
        query_embedder=lambda _: [1, 0],
    )
    assert result["status"] == "failed" and result["selected_candidate_id"] is None
    result = LLMEngine(client=client).suggest(
        candidates, observations, candidate_vectors=vectors, cancel=lambda: True
    )
    assert result["status"] == "cancelled" and not result["request_log"]


def test_rng_restore_and_initial_design_no_provider():
    candidates, observations, _ = fixture_rows()
    first = LLMEngine().suggest(candidates, observations[:1])
    assert first["status"] == "initial_design" and not first["request_log"]
    state = first["rng_state_after"]
    left = LLMEngine(rng_state=state).suggest(candidates, observations[:1])
    right = LLMEngine(rng_state=state).suggest(candidates, observations[:1])
    assert left["selected_candidate_id"] == right["selected_candidate_id"]

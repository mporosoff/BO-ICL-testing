from boicl import AskTellFewShotTopk, Pool
from boicl.local_app import LocalBOState, _best_trace, _coerce_float


def test_import_dataset_uses_first_column_and_optional_values(tmp_path):
    state = LocalBOState(tmp_path)
    raw = b"procedure,value,uncertainty\nproc a,1.2,0.1\nproc b,,\nproc c,2.5,\n"

    payload = state.import_dataset("dataset.csv", raw)

    assert payload["candidate_count"] == 3
    assert [candidate["procedure"] for candidate in payload["candidates"]] == [
        "proc a",
        "proc b",
        "proc c",
    ]
    assert [obs["value"] for obs in payload["observations"]] == [1.2, 2.5]
    assert payload["observations"][0]["uncertainty"] == 0.1
    assert payload["objective_names"] == ["value"]


def test_import_dataset_can_select_between_multiple_objectives(tmp_path):
    state = LocalBOState(tmp_path)
    raw = b"procedure,yield,selectivity,yield_uncertainty\nproc a,1.2,8.0,0.1\nproc b,2.5,7.0,0.2\n"

    state.import_dataset("dataset.csv", raw)
    payload = state.update_config({"objective_name": "selectivity"})

    assert payload["objective_names"] == ["yield", "selectivity"]
    assert [obs["value"] for obs in payload["observations"]] == [8.0, 7.0]
    assert payload["observations"][0]["objectives"]["yield"] == 1.2


def test_replicates_keep_candidate_available_until_limit(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure\nproc a\n")
    state.update_config({"replicates_per_candidate": 2})

    assert state.to_json()["available_count"] == 1
    state.add_observation({"candidate_id": "cand-0", "value": 1.0})
    assert state.to_json()["available_count"] == 1
    state.add_observation({"candidate_id": "cand-0", "value": 1.1})
    assert state.to_json()["available_count"] == 0


def test_best_trace_respects_direction():
    observations = [
        {"value": 5.0},
        {"value": 3.0},
        {"value": 7.0},
    ]

    assert [point["best"] for point in _best_trace(observations, "maximize")] == [
        5.0,
        5.0,
        7.0,
    ]
    assert [point["best"] for point in _best_trace(observations, "minimize")] == [
        5.0,
        3.0,
        3.0,
    ]


def test_float_coercion_rejects_empty_and_nonfinite():
    assert _coerce_float("1.5") == 1.5
    assert _coerce_float("") is None
    assert _coerce_float("nan") is None


def test_browser_config_tracks_llm_and_inverse_models(tmp_path):
    state = LocalBOState(tmp_path)

    payload = state.update_config(
        {
            "optimizer": "llm",
            "prediction_model": "gpt-4o-mini",
            "inverse_model": "openrouter/mistralai/mistral-7b-instruct:free",
            "llm_samples": 3,
            "inverse_filter": 4,
        }
    )

    assert payload["config"]["optimizer"] == "llm"
    assert payload["config"]["prediction_model"] == "gpt-4o-mini"
    assert payload["config"]["inverse_model"].startswith("openrouter/")
    assert payload["config"]["llm_samples"] == 3
    assert payload["config"]["inverse_filter"] == 4


def test_fewshot_can_use_separate_inverse_model():
    asktell = AskTellFewShotTopk(model="gpt-4o-mini", inverse_model="gpt-4o")

    assert asktell._model == "gpt-4o-mini"
    assert asktell._inverse_model == "gpt-4o"


def test_pool_keeps_embedding_model_setting():
    pool = Pool(["a", "b"], embedding_model="text-embedding-3-small")

    assert pool.embedding_model == "text-embedding-3-small"

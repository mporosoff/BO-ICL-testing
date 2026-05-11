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

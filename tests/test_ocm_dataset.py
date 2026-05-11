import boicl

from boicl.datasets import build_ocm_dataset


def test_build_ocm_dataset_from_committed_raw_csv():
    data = build_ocm_dataset()

    assert list(data.columns) == ["prompt", "completion"]
    assert len(data) == 12708
    assert data["prompt"].str.len().min() > 0
    assert data["completion"].astype(float).between(0, 100).all()
    assert "Mn-Na2WO4/BN" in data["prompt"].iloc[0]
    assert float(data["completion"].iloc[0]) == 5.86


def test_ocm_random_ask_tell_smoke_without_external_services():
    data = build_ocm_dataset().head(8)
    pool = boicl.Pool(data["prompt"].tolist())
    asktell = boicl.AskTellFewShotTopk(selector_k=None)

    asktell.tell(data["prompt"].iloc[0], float(data["completion"].iloc[0]))
    asktell.tell(data["prompt"].iloc[1], float(data["completion"].iloc[1]))
    chosen, scores, means = asktell.ask(pool, aq_fxn="random", k=2)

    assert len(chosen) == 2
    assert len(scores) == 2
    assert len(means) == 2
    assert set(chosen).issubset(set(data["prompt"]))

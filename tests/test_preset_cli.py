"""CLI preset selection is explicit and does not initialize real state in tests."""
import json

import pytest

from boicl import moc_cli


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["init", "--pair"], {}),
        (["init", "--pair", "--study", "source"], {}),
        (["init", "--pair", "--study", "five_point"], {"study": "five_point"}),
    ],
)
def test_cli_selects_matched_study_without_changing_source_default(
    monkeypatch, capsys, arguments, expected
):
    calls = []

    class FakeService:
        def __init__(self, root):
            calls.append(("state", root))

        def create_pair(self, **kwargs):
            calls.append(("pair", kwargs))
            return {"gp": "gp-id", "llm": "llm-id"}

    monkeypatch.setattr(moc_cli, "CampaignService", FakeService)
    assert moc_cli.main(["--state-dir", "isolated-placeholder", *arguments]) == 0
    assert calls == [("state", "isolated-placeholder"), ("pair", expected)]
    assert json.loads(capsys.readouterr().out) == {"gp": "gp-id", "llm": "llm-id"}


@pytest.mark.parametrize("preset", ["moc_five_gp", "moc_five_llm"])
def test_cli_single_preset_remains_available(monkeypatch, capsys, preset):
    calls = []

    class FakeService:
        def __init__(self, root):
            pass

        def create(self, selected):
            calls.append(selected)
            return "campaign-id"

    monkeypatch.setattr(moc_cli, "CampaignService", FakeService)
    assert moc_cli.main(["init", "--preset", preset]) == 0
    assert calls == [preset]
    assert json.loads(capsys.readouterr().out) == {"campaign_id": "campaign-id"}


@pytest.mark.parametrize(
    "arguments",
    [
        ["init", "--pair", "--study", "unknown"],
        ["init", "--study", "five_point"],
    ],
)
def test_cli_invalid_study_usage_stops_before_state_creation(monkeypatch, arguments):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid CLI arguments must not create or open campaign state")

    monkeypatch.setattr(moc_cli, "CampaignService", forbidden)
    with pytest.raises(SystemExit) as error:
        moc_cli.main(arguments)
    assert error.value.code == 2

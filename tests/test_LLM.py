import boicl
from boicl import llm_model
import numpy as np
from abc import ABC
import pytest
import os

from langchain_core.messages import HumanMessage, SystemMessage

np.random.seed(0)
LIVE_API_TESTS = os.environ.get("RUN_LIVE_API_TESTS") == "1"


def test_extract_numeric_prediction_prefers_explicit_prediction_over_bounds():
    response = "The valid range is 0 to 100. Prediction: 5.25"

    assert llm_model.extract_numeric_prediction(response) == pytest.approx(5.25)


def test_extract_numeric_prediction_rejects_bare_range():
    with pytest.raises(ValueError):
        llm_model.extract_numeric_prediction("0 to 100")


def test_extract_numeric_prediction_ignores_confidence_percentage():
    response = "100% confident that the value is 0%"

    assert llm_model.extract_numeric_prediction(response) == pytest.approx(0.0)


def test_extract_numeric_prediction_accepts_numeric_only_percentage():
    assert llm_model.extract_numeric_prediction("3.5%") == pytest.approx(3.5)


def test_make_distribution_preserves_raw_llm_samples_when_collapsed():
    dist = llm_model.make_dd(np.array([0.0, 0.0, 0.0]), np.array([1 / 3, 1 / 3, 1 / 3]))

    assert dist.mean() == pytest.approx(0.0)
    assert dist.raw_samples() == [0.0, 0.0, 0.0]


def test_chat_adapter_omits_unused_generation_parameters(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        llm_model, "ChatOpenAI", lambda **kwargs: recorded.append(kwargs)
    )
    llm_model.get_llm(model_name="gpt-4o", n=5)
    assert recorded[0]["n"] == 5
    assert not {"top_p", "best_of", "logprobs", "top_logprobs"} & set(recorded[0])
    with pytest.raises(ValueError, match="best_of"):
        llm_model.get_llm(model_name="gpt-4o", best_of=5)


def pytest_generate_tests(metafunc):
    if "model_name" in metafunc.fixturenames:
        models = metafunc.cls.models_to_test()
        metafunc.parametrize("model_name", models)


class TestLLM(ABC):
    __test__ = False

    @classmethod
    def models_to_test(cls):
        raise NotImplementedError("`models_to_test` must be implemented in subclasses.")

    def test_completion(self, model_name):
        llm = llm_model.get_llm(model_name=model_name, stop=["\n\n"])
        result, token = llm.predict("How much is 1 + 1? Answer the only the number")
        assert result[0].mean() == pytest.approx(2, 0.1)


class TestOpenAILLM(TestLLM):
    __test__ = True
    pytestmark = pytest.mark.skipif(
        not LIVE_API_TESTS, reason="requires live LLM API access"
    )

    @classmethod
    def models_to_test(cls):
        return ["gpt-3.5-turbo-instruct"]


class TestChatOpenAILLM(TestLLM):
    __test__ = True
    pytestmark = pytest.mark.skipif(
        not LIVE_API_TESTS, reason="requires live LLM API access"
    )

    @classmethod
    def models_to_test(cls):
        return ["gpt-3.5-turbo-0125"]

    def test_parse_response(self, model_name):
        print(model_name)
        prompt = "2 + 2 is"
        answer = 4
        llm = llm_model.get_llm(
            n=3,
            best_of=3,
            temperature=1,
            model_name=model_name,
            top_p=0.99,
            stop=["\n"],
        )

        query = [SystemMessage(content=""), HumanMessage(content=prompt)]

        g = llm.llm.generate([query]).generations
        result = llm.parse_response(g[0])

        assert abs(result.mode().astype(int) - answer) <= 1

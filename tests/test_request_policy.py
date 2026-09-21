from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from boicl.request_policy import ReliableClient, RequestPolicy
import boicl.request_policy as module


class ClockEvent:
    def __init__(self):
        self.now = 0.0
        self.stopped = False
        self.waits = []
        self.stop_during_wait = False

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        if self.stop_during_wait:
            self.stopped = True
        return self.stopped


class ProviderError(Exception):
    def __init__(self, status, headers=None):
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


@pytest.fixture
def clock(monkeypatch):
    value = ClockEvent()
    monkeypatch.setattr(module.time, "monotonic", lambda: value.now)
    return value


def test_spacing_applies_to_each_actual_outgoing_request_and_zero_is_valid(clock):
    policy = RequestPolicy({"request_spacing_s": 0.5}, cancelled=clock)
    starts = []
    for _ in range(3):
        policy.call(lambda **kw: starts.append(clock.now), input=["procedure"])
    assert starts == [0, 0.5, 1.0]
    assert len(policy.log) == 3
    assert all(row["kind"] == "embeddings" for row in policy.log)
    policy = RequestPolicy(
        {"request_spacing_s": 0, "base_cooldown_s": 0}, cancelled=clock
    )
    starts = []
    for _ in range(2):
        policy.call(lambda: starts.append(clock.now))
    assert starts == [1.0, 1.0]


def test_total_attempts_includes_first_and_retry_after_is_honored(clock):
    policy = RequestPolicy(
        {"maximum_attempts": 3, "request_spacing_s": 0.5, "base_cooldown_s": 1},
        cancelled=clock,
    )
    starts = []

    def failed():
        starts.append(clock.now)
        raise ProviderError(429, {"Retry-After": "4"})

    with pytest.raises(ProviderError):
        policy.call(failed)
    assert starts == [0, 4, 8]
    assert [row["attempt"] for row in policy.log] == [1, 2, 3]
    assert "backoff_s" not in policy.log[-1]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_provider_errors_are_never_retried(clock, status):
    policy = RequestPolicy(cancelled=clock)

    def failed():
        raise ProviderError(status)

    with pytest.raises(ProviderError):
        policy.call(failed)
    assert len(policy.log) == 1 and clock.now == 0


@pytest.mark.parametrize(
    "error",
    [ProviderError(code) for code in (408, 409, 429, 500, 502, 503, 504)]
    + [TimeoutError(), ConnectionError()],
)
def test_transient_errors_use_finite_retry_then_return_success(clock, error):
    policy = RequestPolicy(
        {"maximum_attempts": 2, "base_cooldown_s": 2}, cancelled=clock
    )
    calls = []

    def provider():
        calls.append(clock.now)
        if len(calls) == 1:
            raise error
        return "complete"

    assert policy.call(provider) == "complete"
    assert calls == [0, 2]


def test_retry_after_http_date(clock):
    fixed_now = datetime(2026, 9, 20, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    original = module.datetime
    module.datetime = FixedDatetime
    try:
        policy = RequestPolicy(
            {"maximum_attempts": 2, "base_cooldown_s": 1}, cancelled=clock
        )
        attempts = []

        def failed():
            attempts.append(clock.now)
            raise ProviderError(
                503, {"retry-after": format_datetime(fixed_now + timedelta(seconds=30))}
            )

        with pytest.raises(ProviderError):
            policy.call(failed)
        assert attempts == [0, 30]
    finally:
        module.datetime = original


def test_cancellation_before_request_and_during_backoff_stops_scheduling(clock):
    policy = RequestPolicy(cancelled=clock)
    clock.stopped = True
    with pytest.raises(InterruptedError):
        policy.call(lambda: pytest.fail("must not send after cancellation"))
    assert policy.log == []
    clock.stopped = False
    clock.stop_during_wait = True

    def failed():
        raise ProviderError(429)

    with pytest.raises(InterruptedError):
        policy.call(failed)
    assert len(policy.log) == 1


def test_shared_wrapper_covers_chat_and_embedding_and_rejects_nested_sdk_retry(clock):
    calls = []
    create = lambda **kwargs: calls.append(kwargs) or "response"
    sdk = SimpleNamespace(
        max_retries=0,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        embeddings=SimpleNamespace(create=create),
    )
    policy = RequestPolicy(cancelled=clock)
    client = ReliableClient(sdk, policy)
    client.chat.completions.create(
        model="gpt-4o", n=5, messages=[{"role": "user", "content": "example"}]
    )
    client.embeddings.create(
        model="text-embedding-3-large", input=["experimental procedure: exact"]
    )
    assert len(calls) == 2 and len(policy.log) == 2
    assert [row["kind"] for row in policy.log] == ["chat", "embeddings"]
    assert clock.now == 0.5
    assert "messages" not in str(policy.log) and "example" not in str(policy.log)
    sdk.max_retries = 2
    with pytest.raises(ValueError, match="max_retries=0"):
        ReliableClient(sdk, policy)


@pytest.mark.parametrize(
    "settings",
    [
        {"maximum_attempts": 0},
        {"maximum_attempts": 1.5},
        {"request_spacing_s": -1},
        {"base_cooldown_s": float("nan")},
    ],
)
def test_invalid_policy_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        RequestPolicy(settings)

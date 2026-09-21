"""One cancellable, paced attempt budget around each actual provider request."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import threading
import time
from types import SimpleNamespace


class RequestPolicy:
    def __init__(self, settings=None, cancelled=None, log=None):
        settings = settings or {}
        self.attempts = int(settings.get("maximum_attempts", 8))
        self.spacing = float(settings.get("request_spacing_s", 0.5))
        self.cooldown = float(settings.get("base_cooldown_s", 10.0))
        if self.attempts < 1 or self.attempts != float(
            settings.get("maximum_attempts", 8)
        ):
            raise ValueError(
                "maximum_attempts must be a positive integer including the initial request"
            )
        if (
            not math.isfinite(self.spacing)
            or self.spacing < 0
            or not math.isfinite(self.cooldown)
            or self.cooldown < 0
        ):
            raise ValueError(
                "Request spacing and cooldown must be finite and nonnegative"
            )
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.log = log if log is not None else []
        self.last_start = None
        self.lock = threading.Lock()

    def check(self):
        stopped = (
            self.cancelled.is_set()
            if hasattr(self.cancelled, "is_set")
            else self.cancelled()
        )
        if stopped:
            raise InterruptedError("Cancelled before scheduling another request")

    def wait(self, seconds):
        self.check()
        if hasattr(self.cancelled, "wait"):
            if self.cancelled.wait(max(0, seconds)):
                self.check()
        else:
            deadline = time.monotonic() + max(0, seconds)
            while time.monotonic() < deadline:
                self.check()
                time.sleep(min(0.05, deadline - time.monotonic()))
        self.check()

    def call(self, fn, **kwargs):
        for attempt in range(1, self.attempts + 1):
            self.check()
            with self.lock:
                if self.last_start is not None:
                    self.wait(self.spacing - (time.monotonic() - self.last_start))
                self.check()
                started = time.monotonic()
                self.last_start = started
            entry = {
                "attempt": attempt,
                "model": kwargs.get("model"),
                "kind": "embeddings" if "input" in kwargs else "chat",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            self.log.append(entry)
            try:
                result = fn(**kwargs)
                entry.update(status="success", elapsed_s=time.monotonic() - started)
                return result
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                entry.update(
                    status="failed",
                    status_code=status,
                    error_type=type(exc).__name__,
                    elapsed_s=time.monotonic() - started,
                )
                retryable = (
                    status in {408, 409, 429, 500, 502, 503, 504}
                    or isinstance(exc, (TimeoutError, ConnectionError))
                    or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                )
                if not retryable or attempt == self.attempts:
                    raise
                headers = {
                    str(key).lower(): value
                    for key, value in (
                        getattr(getattr(exc, "response", None), "headers", {}) or {}
                    ).items()
                }
                retry_after = headers.get("retry-after")
                delay = self.cooldown * 2 ** (attempt - 1)
                if retry_after:
                    try:
                        parsed = float(retry_after)
                        if math.isfinite(parsed):
                            delay = max(delay, parsed)
                    except ValueError:
                        try:
                            delay = max(
                                delay,
                                (
                                    parsedate_to_datetime(retry_after)
                                    - datetime.now(timezone.utc)
                                ).total_seconds(),
                            )
                        except (ValueError, TypeError):
                            pass
                entry["backoff_s"] = delay
                self.wait(delay)


class ReliableClient:
    """Wrap an SDK client constructed with max_retries=0, avoiding nested retries."""

    def __init__(self, client, policy):
        if getattr(client, "max_retries", 0) != 0:
            raise ValueError(
                "Construct provider clients with max_retries=0 so application attempts are the complete retry budget"
            )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: policy.call(client.chat.completions.create, **kw)
            )
        )
        self.embeddings = SimpleNamespace(
            create=lambda **kw: policy.call(client.embeddings.create, **kw)
        )

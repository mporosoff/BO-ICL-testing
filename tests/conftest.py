import pytest
import os
import socket
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Only explicitly opted-in live tests may load local credentials."""
    if os.environ.get("RUN_LIVE_API_TESTS") == "1":
        load_dotenv()


@pytest.fixture(autouse=True)
def offline_by_default(monkeypatch):
    if os.environ.get("RUN_LIVE_API_TESTS") == "1":
        return
    # Dummy credentials satisfy constructor validation; all outbound network is
    # prohibited so a missed mock fails before a provider can receive anything.
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-not-a-real-key")
    for name in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    original = socket.socket.connect

    def connect(sock, address):
        if isinstance(address, tuple) and address[0] in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            return original(sock, address)
        raise AssertionError(
            "Default tests prohibit external network requests; inject a mocked provider"
        )

    monkeypatch.setattr(socket.socket, "connect", connect)

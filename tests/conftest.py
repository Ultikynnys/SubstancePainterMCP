import pytest

@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    monkeypatch.setenv("SP_MCP_PERMISSIVE_ROOTS", "0")

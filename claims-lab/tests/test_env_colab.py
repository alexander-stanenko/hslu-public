"""cl.setup() in Colab: keys and endpoint come from Colab Secrets, the provider is the class provider (Azure OpenAI).

Colab is simulated with a fake `google.colab.userdata`; no test makes a model call (ping=False).
"""
import os
import sys
import types

import pytest

from claims_lab import models
from claims_lab.env import setup

NAMES = ("CLAIMS_LAB_PROVIDER", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
         "AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "OPENAI_API_VERSION")


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """No keys or provider settings; setup() writes to os.environ, so restore it afterwards."""
    saved = dict(os.environ)
    for name in NAMES:
        os.environ.pop(name, None)
    monkeypatch.chdir(tmp_path)                                   # no .env or .key above the working folder
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "")  # nobody types a key
    yield
    os.environ.clear()
    os.environ.update(saved)


def fake_colab(monkeypatch, secrets: dict):
    def get(name):
        if name not in secrets:
            raise KeyError(f"Secret {name} does not exist.")  # stands in for Colab's SecretNotFoundError
        return secrets[name]
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=get)
    google = sys.modules.get("google") or types.ModuleType("google")  # the parent package may not be installed
    monkeypatch.setattr(google, "colab", colab, raising=False)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def test_colab_secrets_choose_azure_with_course_settings(clean_env, monkeypatch, capsys):
    fake_colab(monkeypatch, {"AZURE_OPENAI_API_KEY": " test-key \n",
                             "AZURE_OPENAI_ENDPOINT": "https://lakeside.example.com/openai/deployments/x/chat/completions?api-version=1"})
    setup(ping=False)
    out = capsys.readouterr().out
    assert models.provider() == "azure_openai"
    assert os.environ["AZURE_OPENAI_API_KEY"] == "test-key"
    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://lakeside.example.com/"
    for name, value in models.COURSE_AZURE.items():
        assert os.environ[name] == value
    assert "API key from Colab Secrets" in out and "Azure endpoint from Colab Secrets" in out
    assert "ready ✔" in out and "NOT ready" not in out
    setup(ping=False)  # running the cell again (every notebook starts with setup)
    assert capsys.readouterr().out.count("from Colab Secrets") == 2


def test_colab_without_secrets_names_both_missing_secrets(clean_env, monkeypatch, capsys):
    fake_colab(monkeypatch, {})
    setup(ping=False)
    out = capsys.readouterr().out
    assert models.provider() == "azure_openai"  # the class provider, so the message names the Azure secrets
    assert "API key from NOWHERE" in out and "add the secret AZURE_OPENAI_ENDPOINT" in out
    assert "NOT ready" in out


def test_colab_with_key_but_no_endpoint_is_not_ready(clean_env, monkeypatch, capsys):
    fake_colab(monkeypatch, {"AZURE_OPENAI_API_KEY": "test-key"})
    setup(ping=False)
    out = capsys.readouterr().out
    assert models.provider() == "azure_openai"
    assert "API key from Colab Secrets" in out and "Azure endpoint from NOWHERE" in out and "NOT ready" in out


def test_outside_colab_without_keys_keeps_the_development_default(clean_env, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails: not in Colab
    setup(ping=False)
    assert models.provider() == "anthropic"
    assert "Azure endpoint" not in capsys.readouterr().out


def test_azure_model_uses_the_course_deployment_by_default(clean_env):
    os.environ.update(CLAIMS_LAB_PROVIDER="azure_openai", AZURE_OPENAI_API_KEY="test-key",
                      AZURE_OPENAI_ENDPOINT="https://lakeside.example.com/")
    m = models.get_model()  # builds the client only; nothing is sent
    assert m.deployment_name == models.COURSE_AZURE["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    assert m.openai_api_version == models.COURSE_AZURE["OPENAI_API_VERSION"]
    emb = models.get_embeddings()
    assert emb.deployment == models.COURSE_AZURE["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

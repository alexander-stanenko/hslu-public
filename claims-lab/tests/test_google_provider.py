"""The course on a Google AI Studio key: Gemini through Google's OpenAI-compatible endpoint.

No test sends a request: setup runs with ping=False, and OpenAI's client is replaced where a call would go out.
"""
import os
import sys

from test_env_colab import clean_env, fake_colab  # noqa: F401  (the fixture and the fake Colab Secrets)

from claims_lab import models
from claims_lab.env import setup


def test_a_google_key_alone_switches_the_course_to_gemini(clean_env, monkeypatch, capsys):
    fake_colab(monkeypatch, {"GOOGLE_API_KEY": " test-google-key \n"})
    setup(ping=False)
    out = capsys.readouterr().out
    assert models.provider() == "google" and os.environ["GOOGLE_API_KEY"] == "test-google-key"
    assert models.model_name() == models.GOOGLE["chat_model"]
    assert "provider google, API key from Colab Secrets" in out and f"Google model {models.GOOGLE['chat_model']}" in out
    assert "ready: yes" in out


def test_an_azure_key_wins_over_a_google_key(clean_env, monkeypatch, capsys):
    fake_colab(monkeypatch, {"GOOGLE_API_KEY": "g", "AZURE_OPENAI_API_KEY": "a", "AZURE_OPENAI_ENDPOINT": "https://x.example.com/"})
    setup(ping=False)
    assert models.provider() == "azure_openai" and models.model_name() == "gpt-5.4-mini"


def test_the_install_cell_choice_is_kept(clean_env, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "google.colab", None)   # not in Colab
    os.environ.update(GOOGLE_API_KEY="g", CLAIMS_LAB_PROVIDER="google", AZURE_OPENAI_API_KEY="a")
    setup(ping=False)
    assert models.provider() == "google" and "ready: yes" in capsys.readouterr().out


def test_gemini_chat_model_uses_langchains_google_integration(clean_env):
    from langchain_google_genai import ChatGoogleGenerativeAI
    os.environ.update(CLAIMS_LAB_PROVIDER="google", GOOGLE_API_KEY="test-google-key")
    m = models.get_model()   # builds the client only; nothing is sent
    assert isinstance(m, ChatGoogleGenerativeAI)   # keeps Gemini's thought signatures between tool calls
    assert m.model.removeprefix("models/") == models.GOOGLE["chat_model"]
    assert m.reasoning_effort == "low" and m.max_output_tokens == 4000 and m.max_retries == 10
    assert models.get_model(effort="high").reasoning_effort == "high"


def test_gemini_embeddings_send_text(clean_env):
    os.environ.update(CLAIMS_LAB_PROVIDER="google", GOOGLE_API_KEY="test-google-key")
    emb = models.get_embeddings()
    assert emb.model == "gemini-embedding-001" and emb.openai_api_base == models.GOOGLE["base_url"]
    assert emb.check_embedding_ctx_length is False


def test_openai_client_goes_to_google_with_the_gemini_model(clean_env):
    from openai.types.chat import ChatCompletion

    import claims_lab as cl
    os.environ.update(CLAIMS_LAB_PROVIDER="google", GOOGLE_API_KEY="test-google-key")
    client = cl.get_client()
    assert str(client._client.base_url) == models.GOOGLE["base_url"]
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return ChatCompletion.model_validate({
            "id": "x", "object": "chat.completion", "created": 0, "model": "gemini-3.8-flash",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000}})
    client._client.chat.completions.create = create
    before = cl.costs.SESSION_METER.cost_usd
    reply = client.chat.completions.create(model=cl.model_name(), messages=[{"role": "user", "content": "Reply with OK."}])
    assert reply.choices[0].message.content == "OK"
    assert sent["model"] == models.GOOGLE["chat_model"] and sent["reasoning_effort"] == "low"
    assert round(cl.costs.SESSION_METER.cost_usd - before, 2) == 0.75   # the Gemini price, not the fallback


def test_gemini_prices_also_for_the_models_prefix():
    from claims_lab.costs import price_for
    assert price_for("gemini-3.5-flash-lite") == price_for("models/gemini-3.5-flash-lite") == (0.30, 2.50)
    assert price_for("gemini-3.8-flash") == (0.75, 3.75)


def test_only_the_policy_docs_server_gets_the_google_key(monkeypatch):
    from claims_lab.mcp_servers import stdio_connections
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    conns = stdio_connections()
    assert conns["policy_docs"]["env"]["GOOGLE_API_KEY"] == "test-google-key"
    assert "GOOGLE_API_KEY" not in conns["claims_systems"]["env"]


def test_gemini_json_in_a_code_fence_is_unwrapped(clean_env):
    import json

    from openai.types.chat import ChatCompletion

    import claims_lab as cl
    os.environ.update(CLAIMS_LAB_PROVIDER="google", GOOGLE_API_KEY="test-google-key")
    client = cl.get_client()

    def create(**kwargs):
        return ChatCompletion.model_validate({
            "id": "x", "object": "chat.completion", "created": 0, "model": "gemini-3.5-flash-lite",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '```json\n{"decision": "fast_track"}\n```'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    client._client.chat.completions.create = create
    reply = client.chat.completions.create(model=cl.model_name(), messages=[{"role": "user", "content": "Decide."}])
    assert json.loads(reply.choices[0].message.content) == {"decision": "fast_track"}

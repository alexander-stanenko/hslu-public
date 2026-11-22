"""One place that decides which LLM provider and model the notebooks use.

Notebooks only call `get_model()` and `get_embeddings()`. The course runs on Azure OpenAI (gpt-5.4-mini) or, with a
Google AI Studio key, on Gemini ('google'). 'offline' is a scripted
stand-in for tests. Anthropic is used only when CLAIMS_LAB_PROVIDER=anthropic is set.
"""
from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .costs import SESSION_METER

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"

# The course's Azure OpenAI deployments and API version (the class provider). Only names that are not secret live
# here; the key and the endpoint come from Colab Secrets or the environment (see `claims_lab.setup`).
COURSE_AZURE = {"AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-5.4-mini",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
                "OPENAI_API_VERSION": "2025-04-01-preview"}

# Gemini with a Google AI Studio key. OpenAI's client (notebook 01, Lab 1) and the embeddings go through Google's
# OpenAI-compatible endpoint; LangChain models use LangChain's Google integration (langchain-google-genai).
# CLAIMS_LAB_GOOGLE_MODEL and CLAIMS_LAB_GOOGLE_EMBEDDING_MODEL override the models.
GOOGLE = {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
          # gemini-3.8-flash answered "high demand" (503) on every try on 22 November 2026; flash-lite answered each time
          "chat_model": "gemini-3.5-flash-lite",
          "embedding_model": "gemini-embedding-001",
          # Gemini 3 models always reason a little; "low" keeps calls quick and cheap, close to gpt-5.4-mini's default
          "reasoning_effort": "low"}


def provider() -> str:
    """'azure_openai' (the course provider and the default), 'google', 'offline' or 'anthropic', from CLAIMS_LAB_PROVIDER.

    No key is used to guess the provider here: an Anthropic key in the environment never switches the course to
    Anthropic. `claims_lab.setup` switches to 'google' when only a Google key is found."""
    return os.environ.get("CLAIMS_LAB_PROVIDER") or "azure_openai"


def google_model() -> str:
    return os.environ.get("CLAIMS_LAB_GOOGLE_MODEL") or GOOGLE["chat_model"]


def model_name() -> str:
    """The name to pass as `model=` to OpenAI's client (`cl.get_client()`): the Azure deployment, or the Gemini model."""
    p = provider()
    if p == "google":
        return google_model()
    if p == "anthropic":
        return os.environ.get("CLAIMS_LAB_MODEL", DEFAULT_ANTHROPIC_MODEL)
    if p == "offline":
        return "offline"
    return os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", COURSE_AZURE["AZURE_OPENAI_CHAT_DEPLOYMENT"])


def google_reasoning_effort() -> str:
    return os.environ.get("CLAIMS_LAB_REASONING_EFFORT") or GOOGLE["reasoning_effort"]


def get_model(model: str | None = None, effort: str | None = None, **kwargs) -> BaseChatModel:
    """Return the course chat model, wired to the session cost meter and with automatic retries.

    Args:
        model: optional model id (Anthropic) or deployment name (Azure). Defaults to the course model.
        effort: how much the model reasons before it answers ("low", "medium", "high"); None keeps the provider's
            default (or CLAIMS_LAB_REASONING_EFFORT). Ignored by the offline model.
        **kwargs: passed to the LangChain chat model.
    """
    kwargs.setdefault("callbacks", [SESSION_METER])
    if effort:
        kwargs["reasoning_effort" if provider() in ("azure_openai", "google") else "effort"] = effort
    if provider() == "offline":
        from .offline import OfflineChatModel
        return OfflineChatModel(callbacks=kwargs["callbacks"])
    kwargs.setdefault("max_retries", 6)
    if provider() == "google":
        # LangChain's Google integration, not ChatOpenAI on the compatible endpoint: Gemini 3 wants each tool call's
        # "thought signature" sent back in the next request, and only the Google integration keeps it
        kwargs.setdefault("max_output_tokens", int(os.environ.get("CLAIMS_LAB_MAX_OUTPUT_TOKENS", 4000)))
        kwargs.setdefault("reasoning_effort", google_reasoning_effort())
        kwargs["max_retries"] = max(kwargs["max_retries"], 10)   # the free tier's per-minute limit needs a longer wait
        return init_chat_model(f"google_genai:{model or google_model()}", google_api_key=os.environ.get("GOOGLE_API_KEY"),
                               **kwargs)
    if provider() == "azure_openai":
        deployment = model or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", COURSE_AZURE["AZURE_OPENAI_CHAT_DEPLOYMENT"])
        # cap each answer (reasoning tokens count as output): a single call can't run away
        kwargs.setdefault("max_tokens", int(os.environ.get("CLAIMS_LAB_MAX_OUTPUT_TOKENS", 4000)))
        if os.environ.get("CLAIMS_LAB_REASONING_EFFORT"):
            kwargs.setdefault("reasoning_effort", os.environ["CLAIMS_LAB_REASONING_EFFORT"])
        return init_chat_model(f"azure_openai:{deployment}", azure_deployment=deployment,
                               api_version=os.environ.get("OPENAI_API_VERSION", COURSE_AZURE["OPENAI_API_VERSION"]), **kwargs)
    return init_chat_model(f"anthropic:{model or os.environ.get('CLAIMS_LAB_MODEL', DEFAULT_ANTHROPIC_MODEL)}", **kwargs)


def get_embeddings():
    """Provider embeddings if configured (Azure or Google), else None (retrieval then uses a local TF-IDF stand-in)."""
    if provider() == "google":
        from langchain_openai import OpenAIEmbeddings
        # check_embedding_ctx_length=False: send the text itself, not OpenAI token ids, which Google does not accept
        return OpenAIEmbeddings(model=os.environ.get("CLAIMS_LAB_GOOGLE_EMBEDDING_MODEL") or GOOGLE["embedding_model"],
                                base_url=GOOGLE["base_url"], api_key=os.environ.get("GOOGLE_API_KEY"),
                                check_embedding_ctx_length=False, max_retries=3)
    if provider() == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", COURSE_AZURE["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"])
        return AzureOpenAIEmbeddings(azure_deployment=deployment,
                                     api_version=os.environ.get("OPENAI_EMBEDDING_API_VERSION", "2023-05-15"), max_retries=3)
    return None


def run_config(max_steps: int = 12, **extra) -> dict:
    """Config for agent.invoke(..., config=run_config()): caps the number of agent steps.

    LangChain's create_agent allows 9,999 steps by default; a confused agent can loop for a long time and
    re-send its growing history on every step. Pass this config to every agent run in the course.
    """
    return {"recursion_limit": 2 * max_steps + 1, **extra}

"""One place that decides which LLM provider and model the notebooks use.

Notebooks only call `get_model()` and `get_embeddings()`. Switching the whole course
from Anthropic to Azure OpenAI is a matter of environment variables (see `claims_lab.setup`).
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


def provider() -> str:
    """'anthropic', 'azure_openai' or 'offline' (from CLAIMS_LAB_PROVIDER, else inferred from available keys)."""
    p = os.environ.get("CLAIMS_LAB_PROVIDER")
    if p:
        return p
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        return "azure_openai"
    return "anthropic"


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
        kwargs["reasoning_effort" if provider() == "azure_openai" else "effort"] = effort
    if provider() == "offline":
        from .offline import OfflineChatModel
        return OfflineChatModel(callbacks=kwargs["callbacks"])
    kwargs.setdefault("max_retries", 6)
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
    """Provider embeddings if configured (Azure), else None (retrieval then uses a local TF-IDF stand-in)."""
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

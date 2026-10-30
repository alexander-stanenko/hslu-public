"""Pre-flight check run at the top of every notebook: key, provider, package versions, data, model ping."""
from __future__ import annotations

import logging
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PINNED = {"langchain": "1.4.7", "langgraph": "1.2.17", "fastmcp": "3.4.11", "mcp": "1.30.0", "langchain-mcp-adapters": "0.3.2"}
KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "azure_openai": "AZURE_OPENAI_API_KEY", "offline": None}


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


_FROM_COLAB: dict[str, str] = {}  # values copied from Colab Secrets, to report where a value came from


def _secret(name: str) -> str | None:
    """Find `name` in the environment or in Colab Secrets (copied into the environment). Returns where it was found."""
    if os.environ.get(name):
        return "Colab Secrets" if _FROM_COLAB.get(name) == os.environ[name] else "environment"
    if _in_colab():
        try:
            from google.colab import userdata
            value = userdata.get(name)
        except Exception:  # secret missing, or "Notebook access" not switched on for this notebook
            value = None
        if value:
            os.environ[name] = _FROM_COLAB[name] = value.strip()
            return "Colab Secrets"
    return None


def _load_key(name: str) -> str | None:
    src = _secret(name)
    if src:
        return src
    for d in [Path.cwd(), *Path.cwd().parents]:  # local development: a .key file next to the notebooks
        if (d / ".key").exists() and name == "ANTHROPIC_API_KEY":
            os.environ[name] = (d / ".key").read_text().strip()
            return f"{d / '.key'}"
    from getpass import getpass
    value = getpass(f"{name} (input hidden): ").strip()
    if value:
        os.environ[name] = value
        return "typed in"
    return None


def _load_dotenv() -> None:
    """Local development: read KEY=VALUE lines from the nearest .env (never overrides existing variables)."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


class _RetryNotice(logging.Handler):
    """Says so when the provider's client waits before a retry (a rate limit or a busy server), at most every 20 s.

    Without it, a rate limit looks like a cell that hangs: the client retries silently up to `max_retries` times."""

    def __init__(self):
        super().__init__(logging.INFO)
        self._last = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        import time
        if "Retrying request" not in record.getMessage() or time.time() - self._last < 20:
            return
        self._last = time.time()
        print("  ⏳ The provider asked us to wait (the shared rate limit, or a busy server). The request is retried "
              "automatically: let the cell run.", flush=True)


def _quiet() -> None:
    for name in ("httpx", "mcp", "fastmcp", "FastMCP", "anthropic", "openai", "langchain_mcp_adapters"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in ("openai._base_client", "anthropic._base_client"):  # only their retry notices, printed by _RetryNotice
        client_log = logging.getLogger(name)
        client_log.setLevel(logging.INFO)
        client_log.propagate = False
        if not any(isinstance(h, _RetryNotice) for h in client_log.handlers):
            client_log.addHandler(_RetryNotice())
    os.environ.setdefault("FASTMCP_SHOW_CLI_BANNER", "false")


def _azure_settings(defaults: dict) -> bool:
    """Endpoint from Colab Secrets or the environment; deployments and API version default to the course's."""
    for k, v in defaults.items():
        os.environ.setdefault(k, v)
    src = _secret("AZURE_OPENAI_ENDPOINT")
    if src:
        # accept a full deployment URL too (https://<resource>.../openai/deployments/...): keep only the resource part
        url = os.environ["AZURE_OPENAI_ENDPOINT"].split("?")[0].split("/openai/")[0].rstrip("/") + "/"
        os.environ["AZURE_OPENAI_ENDPOINT"] = url
        if src == "Colab Secrets":
            _FROM_COLAB["AZURE_OPENAI_ENDPOINT"] = url
    print(f"  {'✔' if src else '✘'} Azure endpoint from {src or 'NOWHERE — add the secret AZURE_OPENAI_ENDPOINT'}, "
          f"deployment {os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT']}")
    return bool(src)


def _scripted_input() -> None:
    """Automated test runs have nobody at the keyboard: input() then answers from CLAIMS_LAB_INPUT ("a|b|..." in turn,
    empty when they run out, which a cell can treat as "decline"). In Colab and Jupyter input() is unchanged."""
    import builtins
    answers = iter(os.environ.get("CLAIMS_LAB_INPUT", "").split("|"))

    def scripted(prompt: str = "", *args, **kwargs) -> str:
        answer = next(answers, "")
        print(f"{prompt}{answer}")
        return answer
    builtins.input = scripted
    try:  # in a Jupyter kernel, input() is the kernel's raw_input, put back before every cell
        from IPython import get_ipython
        kernel = getattr(get_ipython(), "kernel", None)
        if kernel is not None:
            kernel.raw_input = scripted
    except ImportError:
        pass


def setup(provider: str | None = None, ping: bool = True, **azure) -> None:
    """Prepare the notebook: API key, provider, versions, data, a test call to the model.

    Args:
        provider: 'anthropic', 'azure_openai' or 'offline' (default: from CLAIMS_LAB_PROVIDER or available keys).
            'offline' uses a scripted stand-in model: no key, no network, no cost (for testing the notebooks).
        ping: send one tiny request to check that the model answers.
        **azure: optional Azure settings, e.g. endpoint=..., chat_deployment=..., embedding_deployment=..., api_version=...
    """
    _quiet()
    if not _in_colab():
        _load_dotenv()
    if os.environ.get("CLAIMS_LAB_NONINTERACTIVE") == "1":
        _scripted_input()
    for k, v in azure.items():
        os.environ[{"endpoint": "AZURE_OPENAI_ENDPOINT", "chat_deployment": "AZURE_OPENAI_CHAT_DEPLOYMENT",
                    "embedding_deployment": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "api_version": "OPENAI_API_VERSION"}[k]] = v
    if provider:
        os.environ["CLAIMS_LAB_PROVIDER"] = provider
    elif not os.environ.get("CLAIMS_LAB_PROVIDER"):
        # Colab keeps the keys in Colab Secrets, not in the environment: look there before the provider is chosen.
        # In Colab without any key, choose the class provider, so that the check below names the missing secret.
        if not _secret("AZURE_OPENAI_API_KEY") and _in_colab() and not _secret("ANTHROPIC_API_KEY"):
            os.environ["CLAIMS_LAB_PROVIDER"] = "azure_openai"
    from . import models
    from .data import labels, list_packages

    p = models.provider()
    ok = True
    print("claims_lab setup")
    if p == "offline":
        src = "not needed"
        print("  ✔ provider offline: scripted stand-in model, no API calls, answers are NOT from a real LLM")
    else:
        src = _load_key(KEY_NAMES[p])
        print(f"  {'✔' if src else '✘'} provider {p}, API key from {src or 'NOWHERE — add it to Colab Secrets'}")
    ok &= bool(src)
    if p == "azure_openai":
        ok &= _azure_settings(models.COURSE_AZURE)
    for pkg, want in PINNED.items():
        try:
            have = version(pkg)
        except PackageNotFoundError:
            have = "missing"
        if have != want:
            print(f"  ⚠ {pkg} {have} (course tested with {want})")
    print(f"  ✔ data: {len(list_packages('dev'))} dev + {len(list_packages('heldout'))} held-out claim packages, {len(labels())} labels")
    if ping and ok:
        try:
            t0 = time.time()
            m = models.get_model()
            m.invoke("Reply with OK.")
            name = getattr(m, "model", None) or getattr(m, "deployment_name", None) or getattr(m, "model_name", "?")
            print(f"  ✔ model {name} answers ({time.time() - t0:.1f}s)")
        except Exception as e:
            ok = False
            msg = str(e)
            if "credit balance" in msg.lower() or "quota" in msg.lower():
                hint = "the API account has no credit/quota left — tell the instructor"
            elif "401" in msg or "auth" in msg.lower():
                hint = "check the API key"
            elif "429" in msg:
                hint = "rate limit — wait 30 s and re-run"
            else:
                hint = "see the error"
            print(f"  ✘ model call failed ({type(e).__name__}): {hint}\n    {msg[:200]}")
    try:
        from .style import use_hslu_style
        use_hslu_style()
    except Exception:
        pass
    print("  ready ✔" if ok else "  NOT ready — fix the ✘ items above")

"""The notice printed when the provider's client waits before a retry (a rate limit), instead of a silent hang."""
import logging

from claims_lab import env


def test_a_retry_is_announced_once_in_twenty_seconds(capsys):
    env._quiet()
    log = logging.getLogger("openai._base_client")
    for h in log.handlers:
        if isinstance(h, env._RetryNotice):
            h._last = 0.0
    log.info("Retrying request in %f seconds (retry %i of %s)", 0.8, 1, 6)
    log.info("Retrying request in %f seconds (retry %i of %s)", 1.6, 2, 6)
    log.info("Request options: something else")
    out = capsys.readouterr().out
    assert out.count("The provider asked us to wait") == 1


def test_quiet_twice_adds_one_handler():
    env._quiet()
    env._quiet()
    assert sum(isinstance(h, env._RetryNotice) for h in logging.getLogger("openai._base_client").handlers) == 1

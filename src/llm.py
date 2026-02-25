"""
Shared LLM utilities - model factories and rate-limit-aware invocation.

Claude rate limits:
  - Claude Sonnet 4.5 : 50 RPM  |  30,000 ITPM  |  8,000 OTPM
  - Claude Haiku 4.5  : 50 RPM  |  50,000 ITPM  | 10,000 OTPM

`invoke_with_backoff` retries on HTTP 429 (RateLimitError) with exponential backoff: 1 s, 2 s, 4 s, 8 s, 16 s, 32 s (up to 6 attempts, ~63 s total).
"""

import logging
import time

import anthropic
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5" #-20251001"
SONNET_MODEL = "claude-sonnet-4-5"


def get_haiku(temperature: float = 0) -> ChatAnthropic:
    """Return a ChatAnthropic instance using Claude Haiku 4.5 (fast tier)."""
    return ChatAnthropic(model=HAIKU_MODEL, temperature=temperature)


def get_sonnet(temperature: float = 0) -> ChatAnthropic:
    """Return a ChatAnthropic instance using Claude Sonnet 4.5 (smart tier)."""
    return ChatAnthropic(model=SONNET_MODEL, temperature=temperature)


def invoke_with_backoff(llm, messages, max_retries: int = 6):
    """
    Invoke an LLM with exponential backoff on rate-limit errors (HTTP 429).

    Works with both plain ChatAnthropic instances and structured-output wrappers returned by ``llm.with_structured_output(...)``.

    Parameters
    ----------
    llm :
        Any object with an ``.invoke(messages)`` method.
    messages :
        The message list to pass to ``.invoke()``.
    max_retries : int
        Maximum number of attempts before re-raising the error (default 6).

    Returns
    -------
    The return value of ``llm.invoke(messages)``.

    Raises
    ------
    anthropic.RateLimitError
        If all retry attempts are exhausted.
    """
    for attempt in range(max_retries):
        logger.debug("LLM invocation attempt %d/%d", attempt + 1, max_retries)
        try:
            return llm.invoke(messages)
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                logger.error("Rate limit retries exhausted after %d attempts", max_retries)
                raise
            sleep_secs = 2 ** attempt
            logger.warning("Rate limit hit - sleeping %ds before retry", sleep_secs)
            time.sleep(sleep_secs)

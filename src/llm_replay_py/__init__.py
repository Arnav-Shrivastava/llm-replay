"""
llm-replay
==========
Stop paying for the same AI response twice.

Zero-dependency LLM response caching for developers.
Works with any LLM provider: OpenAI, Anthropic, Gemini, and more.

Basic usage:
    from llm_replay_py import replay

    @replay
    def ask_ai(prompt: str) -> str:
        # your openai / anthropic / any LLM call here
        ...

Advanced usage:
    from llm_replay_py import replay, config, stats, clear

    # Configure once at the top of your script
    config(ttl_days=7, verbose=True)

    @replay
    def ask_ai(prompt: str) -> str:
        ...

    # Force a fresh API call (bypass cache)
    ask_ai("hello", force_refresh=True)

    # See how the cache is performing
    stats()

    # Wipe the cache
    clear()
"""

# We import everything from cache.py and re-export it.
# This means users write `from llm_replay_py import replay`
# instead of `from llm_replay_py.cache import replay`.
# The public API lives here. The implementation lives in cache.py.
from llm_replay_py.cache import (
    replay,
    config,
    stats,
    clear,
)

# __all__ explicitly declares what `from llm_replay_py import *` exports.
# This is professional practice - it documents your public API clearly
# and prevents internal helpers from leaking out.
__all__ = [
    "replay",
    "config",
    "stats",
    "clear",
]

# Semantic version following PEP 440 and semver.org convention:
# MAJOR.MINOR.PATCH
# 0.x.x = alpha/beta phase, API may change
# 1.0.0 = stable public API, we promise not to break things
__version__ = "0.1.0"

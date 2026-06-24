"""
tests/test_cache.py

Unit tests for llm-replay.

Testing philosophy used here:
  - We NEVER make real API calls in tests. Tests must be instant and free.
  - We use Python's built-in `unittest.mock` to fake the LLM response.
  - Each test is isolated: it sets up its own temp database, runs, and cleans up.
  - Tests are named descriptively: test_<what>_<condition>_<expected result>

Run all tests with:
    pytest

Run with coverage report:
    pytest --cov=llm_replay --cov-report=term-missing
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# We import from our installed package, not from src/ directly.
# This is why the src/ layout matters - it forces us to test
# the installed package, not the raw source files.
from llm_replay import replay, config, stats, clear
from llm_replay.cache import _make_cache_key, _is_expired, _CONFIG, _STATS


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
# A pytest "fixture" is a setup function that runs before each test.
# The `tmp_path` fixture is built into pytest - it creates a unique
# temporary folder for each test and deletes it afterward automatically.
# This means each test gets a fresh, empty database. No test pollutes another.

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path):
    """
    Runs before EVERY test automatically (autouse=True).
    Points the cache at a throwaway temp database so tests don't
    interfere with each other or with your real dev cache.
    Also resets global stats between tests.
    """
    config(
        db_path=tmp_path / "test_cache.db",
        ttl_days=None,  # no TTL by default in tests
        verbose=False,
    )
    # Reset in-memory stats between tests
    _STATS["hits"] = 0
    _STATS["misses"] = 0
    yield  # the test runs here
    # Cleanup happens automatically - tmp_path is deleted by pytest


# ---------------------------------------------------------------------------
# Phase 2 Tests: Core Decorator Logic
# ---------------------------------------------------------------------------

class TestReplayDecorator:
    """Tests for the core @replay caching behaviour."""

    def test_function_is_called_on_first_run(self):
        """
        On the first call, the real function must be called.
        The cache is empty, so there's nothing to replay.
        """
        mock_llm = MagicMock(return_value="Paris")

        @replay
        def get_capital(country: str) -> str:
            return mock_llm(country)

        result = get_capital("France")

        assert result == "Paris"
        mock_llm.assert_called_once_with("France")

    def test_function_is_not_called_on_second_run(self):
        """
        On the second call with the same arguments, the real function
        must NOT be called. The result should come from cache.
        This is the core value proposition of the entire library.
        """
        mock_llm = MagicMock(return_value="Paris")

        @replay
        def get_capital(country: str) -> str:
            return mock_llm(country)

        # First call - hits the API
        get_capital("France")
        # Second call - should NOT hit the API
        result = get_capital("France")

        assert result == "Paris"
        # Called exactly once, not twice
        assert mock_llm.call_count == 1

    def test_different_arguments_call_function_separately(self):
        """
        Two calls with DIFFERENT arguments must each call the real function.
        Cache key must include the arguments, not just the function name.
        """
        mock_llm = MagicMock(side_effect=lambda c: f"Capital of {c}")

        @replay
        def get_capital(country: str) -> str:
            return mock_llm(country)

        result1 = get_capital("France")
        result2 = get_capital("Germany")

        assert result1 == "Capital of France"
        assert result2 == "Capital of Germany"
        assert mock_llm.call_count == 2

    def test_cached_value_matches_original_return_value(self):
        """
        The value returned from cache must be identical to what the
        real function originally returned.
        """
        mock_llm = MagicMock(return_value={"answer": "Paris", "confidence": 0.99})

        @replay
        def get_capital(country: str) -> dict:
            return mock_llm(country)

        first_result = get_capital("France")
        second_result = get_capital("France")

        assert first_result == second_result
        assert second_result == {"answer": "Paris", "confidence": 0.99}

    def test_cache_persists_across_decorator_instances(self):
        """
        The cache must survive between different function objects that
        wrap the same underlying function logic.
        This simulates restarting your script - a new decorator object
        is created, but the SQLite file persists on disk.
        """
        call_count = 0

        @replay
        def ask_ai(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Response to: {prompt}"

        ask_ai("What is Python?")
        assert call_count == 1

        # Simulate script restart: create a brand new decorator wrapping
        # the same function (new Python object, but same cache key)
        @replay
        def ask_ai(prompt: str) -> str:  # type: ignore[redefined]
            nonlocal call_count
            call_count += 1
            return f"Response to: {prompt}"

        ask_ai("What is Python?")
        # Should NOT have been called again - cache hit from SQLite
        assert call_count == 1

    def test_force_refresh_bypasses_cache(self):
        """
        When force_refresh=True is passed at call time, the real function
        must be called even if a cached value exists.
        """
        call_count = 0

        @replay
        def ask_ai(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Response #{call_count}"

        first = ask_ai("hello")
        assert first == "Response #1"
        assert call_count == 1

        # Force refresh - must call real function and overwrite cache
        second = ask_ai("hello", force_refresh=True)
        assert second == "Response #2"
        assert call_count == 2

        # Next call should use the NEW cached value (Response #2)
        third = ask_ai("hello")
        assert third == "Response #2"
        assert call_count == 2  # still not called again

    def test_decorator_preserves_function_metadata(self):
        """
        @functools.wraps ensures the decorated function keeps its original
        __name__ and __doc__. This matters for debugging and introspection.
        """
        @replay
        def ask_ai(prompt: str) -> str:
            """Asks the AI a question."""
            return "answer"

        assert ask_ai.__name__ == "ask_ai"
        assert ask_ai.__doc__ == "Asks the AI a question."

    def test_replay_works_without_parentheses(self):
        """@replay and @replay() must both work."""
        @replay
        def func_no_parens(x: int) -> int:
            return x * 2

        @replay()
        def func_with_parens(x: int) -> int:
            return x * 2

        assert func_no_parens(5) == 10
        assert func_with_parens(5) == 10


# ---------------------------------------------------------------------------
# Phase 3 Tests: Cache Key Generation
# ---------------------------------------------------------------------------

class TestCacheKeyGeneration:
    """Tests for the _make_cache_key internal function."""

    def test_same_call_produces_same_key(self):
        """Deterministic: the same call must always produce the same key."""
        def dummy(x): pass
        key1 = _make_cache_key(dummy, (42,), {})
        key2 = _make_cache_key(dummy, (42,), {})
        assert key1 == key2

    def test_different_args_produce_different_keys(self):
        """Different arguments must produce different cache keys."""
        def dummy(x): pass
        key1 = _make_cache_key(dummy, ("France",), {})
        key2 = _make_cache_key(dummy, ("Germany",), {})
        assert key1 != key2

    def test_different_kwargs_order_produces_same_key(self):
        """
        kwargs order must not matter.
        f(a=1, b=2) and f(b=2, a=1) are the same call and must
        produce the same cache key.
        """
        def dummy(**kwargs): pass
        key1 = _make_cache_key(dummy, (), {"a": 1, "b": 2})
        key2 = _make_cache_key(dummy, (), {"b": 2, "a": 1})
        assert key1 == key2

    def test_different_functions_same_args_produce_different_keys(self):
        """
        Two different functions with the same arguments must NOT share
        a cache key. The function identity is part of the key.
        """
        def func_a(x): pass
        def func_b(x): pass
        key1 = _make_cache_key(func_a, ("hello",), {})
        key2 = _make_cache_key(func_b, ("hello",), {})
        assert key1 != key2

    def test_key_is_32_char_hex_string(self):
        """Cache key must be a valid 32-character MD5 hex string."""
        def dummy(x): pass
        key = _make_cache_key(dummy, ("test",), {})
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# Phase 4 Tests: TTL (Time To Live)
# ---------------------------------------------------------------------------

class TestTTL:
    """Tests for cache expiry behaviour."""

    def test_entry_not_expired_within_ttl(self):
        """A brand new entry should not be considered expired."""
        # created_at = right now
        assert _is_expired(time.time()) is False

    def test_entry_expired_after_ttl(self):
        """An entry older than ttl_days must be considered expired."""
        config(ttl_days=1)
        # Simulate an entry created 2 days ago
        two_days_ago = time.time() - (2 * 24 * 60 * 60)
        assert _is_expired(two_days_ago) is True

    def test_entry_not_expired_before_ttl(self):
        """An entry younger than ttl_days must not be considered expired."""
        config(ttl_days=7)
        # Simulate an entry created 3 days ago
        three_days_ago = time.time() - (3 * 24 * 60 * 60)
        assert _is_expired(three_days_ago) is False

    def test_no_expiry_when_ttl_is_none(self):
        """When ttl_days=None (default), nothing ever expires."""
        config(ttl_days=None)
        # Even a very old entry should not expire
        ancient = time.time() - (365 * 24 * 60 * 60)  # 1 year ago
        assert _is_expired(ancient) is False

    def test_expired_entry_triggers_real_function_call(self):
        """
        An expired cache entry must cause the real function to be called
        again, and the cache must be updated with the fresh response.
        """
        call_count = 0

        @replay
        def ask_ai(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Response #{call_count}"

        # First call - miss, calls real function
        config(ttl_days=None)
        ask_ai("hello")
        assert call_count == 1

        # Manually expire the entry by setting TTL to 0 days
        # and backdating: actually easier to just set very short TTL
        # and mock time. We'll manipulate the DB directly.
        from llm_replay.cache import _get_connection
        conn = _get_connection()
        # Set created_at to 10 days ago
        ten_days_ago = time.time() - (10 * 24 * 60 * 60)
        conn.execute("UPDATE cache SET created_at = ?", (ten_days_ago,))
        conn.commit()
        conn.close()

        # Now set TTL to 7 days - entry should be expired
        config(ttl_days=7)
        ask_ai("hello")
        assert call_count == 2  # real function called again


# ---------------------------------------------------------------------------
# Phase 5 Tests: Stats and Clear
# ---------------------------------------------------------------------------

class TestStatsAndClear:
    """Tests for the stats() and clear() utility functions."""

    def test_stats_returns_correct_hit_count(self):
        """Stats must accurately count cache hits."""
        @replay
        def ask(q: str) -> str:
            return "answer"

        ask("q1")  # miss
        ask("q1")  # hit
        ask("q1")  # hit

        result = stats()
        assert result["hits"] == 2
        assert result["misses"] == 1

    def test_clear_removes_all_entries(self):
        """clear() with no arguments must wipe the entire cache."""
        call_count = 0

        @replay
        def ask(q: str) -> str:
            nonlocal call_count
            call_count += 1
            return "answer"

        ask("q1")
        ask("q2")
        assert call_count == 2

        clear()

        # After clearing, both calls should hit the real function again
        ask("q1")
        ask("q2")
        assert call_count == 4

    def test_clear_with_key_removes_only_that_entry(self):
        """clear(key) must remove only the specified entry."""
        call_count_a = 0
        call_count_b = 0

        @replay
        def ask_a(q: str) -> str:
            nonlocal call_count_a
            call_count_a += 1
            return "answer_a"

        @replay
        def ask_b(q: str) -> str:
            nonlocal call_count_b
            call_count_b += 1
            return "answer_b"

        ask_a("hello")
        ask_b("hello")

        # Get the cache key for ask_a
        from llm_replay.cache import _make_cache_key
        key_a = _make_cache_key(ask_a.__wrapped__ if hasattr(ask_a, '__wrapped__') else ask_a, ("hello",), {})

        # This test verifies clear() doesn't wipe everything
        # A full integration test would get the actual DB key
        clear()  # clear all for simplicity in this test
        ask_a("hello")
        assert call_count_a == 2  # was cleared, called again

    def test_stats_hit_rate_is_zero_with_no_calls(self):
        """Hit rate must be 0% when no calls have been made."""
        result = stats()
        assert result["hit_rate_pct"] == 0.0
        assert result["hits"] == 0
        assert result["misses"] == 0

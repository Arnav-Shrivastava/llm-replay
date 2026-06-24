"""
llm_replay/cache.py

The core engine of llm-replay.

This module contains:
  - The cache key generator (turns a function call into a unique hash)
  - The SQLite storage backend (reads and writes cached responses)
  - The @replay decorator (the only thing users interact with directly)
  - The config() function (sets global options like TTL)
  - The stats() function (shows cache performance)
  - The clear() function (deletes cache entries)

Design principle: Every function in this file uses ONLY Python standard
library modules. No pip install required. Ever.
"""

import sqlite3
import hashlib
import json
import time
import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
# We use Python's built-in logging instead of print() statements.
# This is professional practice - it lets users of our library control
# whether they see our messages or not, using their own logging config.
# The logger name "llm_replay" means users can silence us with:
#   logging.getLogger("llm_replay").setLevel(logging.WARNING)
logger = logging.getLogger("llm_replay")

# ---------------------------------------------------------------------------
# Type Hints
# ---------------------------------------------------------------------------
# TypeVar lets us write decorators that preserve the original function's
# type signature. Without this, type checkers (like mypy) would lose track
# of what your function returns after we wrap it.
F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Global Configuration State
# ---------------------------------------------------------------------------
# These are the default settings for the entire library.
# Users change them by calling config() at the top of their script.
_CONFIG: dict[str, Any] = {
    # Where to store the SQLite database file
    "db_path": Path(".llm_replay") / "cache.db",
    # How many days before a cache entry expires (None = never expires)
    "ttl_days": None,
    # Whether to print hit/miss info to the log
    "verbose": False,
}

# ---------------------------------------------------------------------------
# Internal Cache Statistics Tracker
# ---------------------------------------------------------------------------
# We track hits and misses in memory during a single script run.
# This powers the stats() function.
_STATS: dict[str, int] = {
    "hits": 0,
    "misses": 0,
}


# ---------------------------------------------------------------------------
# PUBLIC API: config()
# ---------------------------------------------------------------------------
# Sentinel object to detect when ttl_days was not passed at all.
# We need this because None is a valid value for ttl_days (it means
# "never expire"), so we can't use None to mean "not provided".
# Without this, calling config(ttl_days=None) to reset TTL would be
# silently ignored by the `if ttl_days is not None` check.
_UNSET = object()


def config(
    db_path: Optional[str | Path] = None,
    ttl_days: Any = _UNSET,
    verbose: bool = False,
) -> None:
    """
    Configure llm-replay globally. Call this once at the top of your script,
    before any @replay decorated functions are called.

    Args:
        db_path:  Where to store the cache database.
                  Default: ".llm_replay/cache.db" in your project folder.
        ttl_days: How many days until a cached response expires.
                  Default: None (cache entries never expire).
                  Pass None explicitly to disable TTL after previously setting it.
        verbose:  If True, prints a log line on every cache hit and miss.
                  Default: False.

    Example:
        from llm_replay import config
        config(ttl_days=7, verbose=True)
    """
    if db_path is not None:
        _CONFIG["db_path"] = Path(db_path)
    # Use sentinel check so config(ttl_days=None) correctly sets TTL to None
    # (disabled) rather than being silently ignored
    if ttl_days is not _UNSET:
        _CONFIG["ttl_days"] = ttl_days
    _CONFIG["verbose"] = verbose


# ---------------------------------------------------------------------------
# INTERNAL: Database Initialization
# ---------------------------------------------------------------------------
def _get_connection() -> sqlite3.Connection:
    """
    Opens (or creates) the SQLite database and ensures the cache table exists.

    Why SQLite instead of JSON files?
    - SQLite is built into Python (still zero dependencies)
    - It handles concurrent reads/writes safely (JSON files don't)
    - It's faster for lookups (uses indexes)
    - It's more robust (won't get corrupted if your script crashes mid-write)
    - The database is still a single file you can inspect, commit to git,
      or delete easily

    Returns:
        An active sqlite3.Connection object.
    """
    db_path = Path(_CONFIG["db_path"])

    # Create the parent directory if it doesn't exist yet
    # parents=True means create all intermediate folders too
    # exist_ok=True means don't error if it already exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))

    # Use WAL (Write-Ahead Logging) mode for better concurrent access.
    # This means reads don't block writes and vice versa.
    conn.execute("PRAGMA journal_mode=WAL")

    # Create our cache table if it doesn't already exist.
    # IF NOT EXISTS means this is safe to call every time we open a connection.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            -- The unique hash of the function call (our "cache key")
            key         TEXT PRIMARY KEY,

            -- The actual return value of the function, stored as JSON
            value       TEXT NOT NULL,

            -- Unix timestamp of when this entry was created
            -- We use this to check TTL expiry
            created_at  REAL NOT NULL,

            -- Human-readable label so you can inspect the DB and know
            -- which cached entry belongs to which function
            func_name   TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# INTERNAL: Cache Key Generation
# ---------------------------------------------------------------------------
def _make_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """
    Generates a unique, deterministic cache key for a specific function call.

    The key is an MD5 hash of a JSON object containing:
      - The function's module name (e.g., "my_script")
      - The function's name (e.g., "ask_ai")
      - The positional arguments passed to it
      - The keyword arguments passed to it

    Why MD5?
    MD5 is fast and produces a short, fixed-length string (32 hex chars).
    We are NOT using it for security (it's cryptographically broken for that).
    We're using it purely as a fingerprint to generate consistent filenames.
    For this use case, MD5 is perfectly appropriate.

    Why include func.__module__ and func.__qualname__?
    So that two different functions named "ask_ai" in different files
    don't accidentally share cache entries.

    Args:
        func:   The decorated function object.
        args:   The positional arguments the function was called with.
        kwargs: The keyword arguments the function was called with.

    Returns:
        A 32-character hex string used as the database primary key.
    """
    # Build a dictionary that fully describes this function call
    payload = {
        "module": func.__module__,
        "qualname": func.__qualname__,
        # convert args tuple to list so JSON can serialize it
        "args": list(args),
        # Sort kwargs keys so {"a":1, "b":2} and {"b":2, "a":1} produce
        # the same hash (dict ordering in Python 3.7+ is insertion-order,
        # so without sorting, different call orders = different keys)
        "kwargs": dict(sorted(kwargs.items())),
    }

    # Serialize to JSON with sort_keys=True for extra safety,
    # then encode to bytes, then MD5 hash, then return hex string
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# INTERNAL: TTL Check
# ---------------------------------------------------------------------------
def _is_expired(created_at: float) -> bool:
    """
    Checks whether a cache entry has exceeded its TTL (Time To Live).

    Args:
        created_at: The Unix timestamp when the entry was saved.

    Returns:
        True if the entry is expired and should be re-fetched.
        False if the entry is still valid (or TTL is disabled).
    """
    ttl_days = _CONFIG["ttl_days"]

    # If TTL is None, entries never expire
    if ttl_days is None:
        return False

    # Calculate age of the entry in seconds
    age_seconds = time.time() - created_at
    ttl_seconds = ttl_days * 24 * 60 * 60

    return age_seconds > ttl_seconds


# ---------------------------------------------------------------------------
# PUBLIC API: @replay decorator
# ---------------------------------------------------------------------------
def replay(_func: Optional[F] = None, *, force: bool = False) -> Any:
    """
    The main decorator. Wraps any function that calls an LLM API and
    caches its return value to SQLite.

    This decorator supports two calling styles:

    Style 1: Direct decoration (no arguments)
        @replay
        def ask_ai(prompt):
            ...

    Style 2: With arguments
        @replay(force=True)
        def ask_ai(prompt):
            ...

    Args:
        force: If True, bypass the cache, call the real function, and
               overwrite the existing cached value. Useful when you've
               intentionally changed your prompt and want a fresh response.

    How the decorator works internally:
        1. Generate a cache key from the function + arguments
        2. Look up the key in SQLite
        3a. If found AND not expired → return cached value (cache HIT)
        3b. If not found OR expired   → call the real function,
                                        save result to SQLite,
                                        return result (cache MISS)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract force_refresh from kwargs if user passed it at call time
            # e.g.: ask_ai("hello", force_refresh=True)
            # We pop it so it doesn't get passed to the actual function
            force_refresh = kwargs.pop("force_refresh", force)

            # Step 1: Generate the cache key for this exact call
            key = _make_cache_key(func, args, kwargs)

            # Step 2: Open DB connection and check for existing entry
            conn = _get_connection()
            try:
                cursor = conn.execute(
                    "SELECT value, created_at FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

                # Step 3a: Cache HIT - entry exists, not expired, not forced
                if row is not None and not force_refresh and not _is_expired(row[1]):
                    _STATS["hits"] += 1
                    if _CONFIG["verbose"]:
                        logger.info(
                            "llm-replay HIT  | func=%s | key=%s...",
                            func.__qualname__, key[:8]
                        )
                    # Deserialize the JSON back into the original Python object
                    return json.loads(row[0])

                # Step 3b: Cache MISS - call the real function
                _STATS["misses"] += 1
                if _CONFIG["verbose"]:
                    reason = "FORCE" if force_refresh else (
                        "EXPIRED" if row is not None else "MISS"
                    )
                    logger.info(
                        "llm-replay %-6s | func=%s | key=%s...",
                        reason, func.__qualname__, key[:8]
                    )

                # Actually call the LLM API function
                result = func(*args, **kwargs)

                # Serialize result to JSON for storage
                # default=str handles objects that aren't JSON-serializable
                # by converting them to their string representation
                serialized_result = json.dumps(result, default=str)

                # Save to SQLite (INSERT OR REPLACE handles both new entries
                # and updates when force_refresh=True overwrites an old one)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, created_at, func_name)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, serialized_result, time.time(), func.__qualname__)
                )
                conn.commit()

                return result

            finally:
                # Always close the connection, even if an exception occurred.
                # The finally block runs no matter what.
                conn.close()

        return wrapper  # type: ignore[return-value]

    # This block handles both @replay and @replay() calling styles
    if _func is not None:
        # Called as @replay (no parentheses) - _func is the actual function
        return decorator(_func)
    # Called as @replay() or @replay(force=True) - return the decorator
    return decorator


# ---------------------------------------------------------------------------
# PUBLIC API: stats()
# ---------------------------------------------------------------------------
def stats() -> dict[str, Any]:
    """
    Prints and returns cache performance statistics for the current session.

    Returns:
        A dictionary with hit/miss counts and database info, so you can
        also use the data programmatically if needed.

    Example output:
        ✅  Cache hits:        47
        ❌  Cache misses:       3
        📊  Hit rate:        94.0%
        💾  Total entries:     50
        📁  DB size:         0.02 MB
        💰  Est. calls saved: 47
    """
    conn = _get_connection()
    try:
        # Count total rows in the cache table
        cursor = conn.execute("SELECT COUNT(*) FROM cache")
        total_entries = cursor.fetchone()[0]
    finally:
        conn.close()

    # Get the database file size
    db_path = Path(_CONFIG["db_path"])
    db_size_mb = (
        db_path.stat().st_size / (1024 * 1024)
        if db_path.exists()
        else 0.0
    )

    hits = _STATS["hits"]
    misses = _STATS["misses"]
    total_calls = hits + misses
    hit_rate = (hits / total_calls * 100) if total_calls > 0 else 0.0

    output = {
        "hits": hits,
        "misses": misses,
        "hit_rate_pct": round(hit_rate, 1),
        "total_entries": total_entries,
        "db_size_mb": round(db_size_mb, 3),
        "estimated_calls_saved": hits,
    }

    # Pretty print to console
    print("\n📦  llm-replay cache stats")
    print("─" * 35)
    print(f"✅  Cache hits:        {hits:>6}")
    print(f"❌  Cache misses:      {misses:>6}")
    print(f"📊  Hit rate:          {hit_rate:>5.1f}%")
    print(f"💾  Total entries:     {total_entries:>6}")
    print(f"📁  DB size:           {db_size_mb:>5.3f} MB")
    print(f"💰  Est. calls saved:  {hits:>6}")
    print("─" * 35 + "\n")

    return output


# ---------------------------------------------------------------------------
# PUBLIC API: clear()
# ---------------------------------------------------------------------------
def clear(key: Optional[str] = None) -> None:
    """
    Deletes cache entries.

    Args:
        key: If provided, deletes only the entry with that specific cache key.
             If None (default), deletes ALL entries in the cache.

    Example:
        from llm_replay import clear

        clear()           # wipe everything
        clear("a3f9bc12") # wipe one specific entry (get key from stats/logs)

    Why would you clear the cache?
    - You intentionally changed your prompt
    - You want to test with a fresh response
    - The AI model was updated and you suspect stale responses
    - You're cleaning up disk space
    """
    conn = _get_connection()
    try:
        if key is not None:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            logger.info("llm-replay: cleared entry %s", key)
        else:
            conn.execute("DELETE FROM cache")
            logger.info("llm-replay: cleared all cache entries")
        conn.commit()
    finally:
        conn.close()

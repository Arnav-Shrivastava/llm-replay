# llm-replay ⚡

> Stop paying for the same AI response twice.

`llm-replay` is a zero-dependency Python library that caches LLM API responses to disk. Add one line to your existing code. Every repeated call during development is instant and free.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()

---

## The Problem

You're building an LLM app. You keep re-running your script to test a bug. Every run costs money and takes 6 seconds.

```python
# Without llm-replay: every run costs money and takes seconds
def ask_ai(prompt):
    return openai.chat.completions.create(...)  # 💸 $0.01, ⏱ 4 seconds
```

## The Fix

```python
from llm_replay import replay

# With llm-replay: first run costs money, every run after is instant + free
@replay
def ask_ai(prompt):
    return openai.chat.completions.create(...)  # 💸 $0.00, ⏱ 0.001s
```

That's it. One import, one decorator.

---

## Installation

```bash
pip install llm-replay
```

**Zero dependencies.** Uses only Python's standard library (`sqlite3`, `hashlib`, `json`).

---

## Quick Start

### Basic usage

```python
from llm_replay import replay

@replay
def classify_email(email_text: str) -> str:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify this email: {email_text}"}],
    )
    return response.choices[0].message.content

# First run: calls the real API (takes ~4 seconds)
result = classify_email("Please refund my order #12345")

# Second run: served from cache (takes ~0.001 seconds, costs $0)
result = classify_email("Please refund my order #12345")
```

### Works with any LLM provider

```python
from llm_replay import replay

# OpenAI
@replay
def ask_openai(prompt: str) -> str:
    import openai
    ...

# Anthropic
@replay
def ask_claude(prompt: str) -> str:
    import anthropic
    ...

# Google Gemini
@replay
def ask_gemini(prompt: str) -> str:
    import google.generativeai as genai
    ...
```

### Configuration

```python
from llm_replay import config

# Call once at the top of your script
config(
    ttl_days=7,      # Cache expires after 7 days (default: never)
    verbose=True,    # Print hit/miss info to logs (default: False)
    db_path=".cache/llm.db",  # Custom cache location (default: .llm_replay/cache.db)
)
```

### Force a fresh API call

```python
# Bypass the cache for this specific call
result = ask_ai("What is Python?", force_refresh=True)
```

### View cache statistics

```python
from llm_replay import stats

stats()
# Output:
# 📦  llm-replay cache stats
# ───────────────────────────────────
# ✅  Cache hits:            47
# ❌  Cache misses:           3
# 📊  Hit rate:           94.0%
# 💾  Total entries:         50
# 📁  DB size:            0.021 MB
# 💰  Est. calls saved:      47
# ───────────────────────────────────
```

### Clear the cache

```python
from llm_replay import clear

clear()              # Wipe everything
clear("a3f9bc12")   # Wipe one specific entry
```

---

## How It Works

1. You call your LLM function with some arguments
2. `llm-replay` hashes the function name + all arguments into a unique key
3. It looks up that key in a local SQLite database (`.llm_replay/cache.db`)
4. **Cache hit:** returns the stored response instantly
5. **Cache miss:** calls your real function, stores the result, returns it

The cache key includes: function name, all positional args, all keyword args. So `ask_ai("France")` and `ask_ai("Germany")` are stored separately.

---

## When to Use / Not Use

✅ **Use llm-replay for:**
- Development and debugging loops
- Test suites that would otherwise call real APIs
- Demo scripts where reliability matters more than freshness
- RAG pipelines where you're iterating on retrieval, not generation

❌ **Don't use llm-replay for:**
- Prompts that include real-time data (timestamps, live prices)
- User-personalized content that must always be fresh
- Production apps serving live users (consider a proper cache layer instead)

---

## The Cache File

The cache is stored in `.llm_replay/cache.db` — a standard SQLite database.

```
your-project/
├── main.py
├── .llm_replay/
│   └── cache.db    ← SQLite database, inspect with any SQLite viewer
```

**To share the cache with your team:** commit `.llm_replay/cache.db` to git.
**To reset the cache:** delete `.llm_replay/` or call `clear()`.
**To keep the cache private:** add `.llm_replay/` to `.gitignore`.

---

## Comparison with Alternatives

| | llm-replay | GPTCache | LangChain Cache |
|---|---|---|---|
| Zero dependencies | ✅ | ❌ (FAISS, etc.) | ❌ (requires LangChain) |
| Works with raw SDK calls | ✅ | ✅ | ❌ |
| Single file / drop-in | ✅ | ❌ | ❌ |
| Actively maintained | ✅ | ❌ (abandoned) | ✅ |
| Semantic matching | ❌ (by design) | ✅ | ✅ |
| Built for dev iteration | ✅ | ❌ | ❌ |

---

## Contributing

Pull requests welcome. Please add tests for any new functionality.

```bash
git clone https://github.com/Arnav-Shrivastava/llm-replay.git
cd llm-replay
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## License

MIT. See [LICENSE](LICENSE) for details.

---

## Why the name?

Because you're not *caching* — you're *replaying* a conversation that already happened.

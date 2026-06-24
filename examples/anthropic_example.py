"""
examples/anthropic_example.py

A complete working example showing llm-replay with the Anthropic SDK.

To run this example:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"
    python examples/anthropic_example.py
"""

import os
import time
from llm_replay import replay, config, stats

config(ttl_days=7, verbose=True)

import logging
logging.basicConfig(level=logging.INFO)


@replay
def ask_claude(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """
    Sends a prompt to Claude and returns the response text.
    Cached after the first call.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


if __name__ == "__main__":
    print("\n--- First call: hits the Anthropic API ---")
    start = time.time()
    response = ask_claude("What are the three laws of thermodynamics? Be concise.")
    print(f"Response: {response[:100]}...")
    print(f"Time: {time.time() - start:.3f}s")

    print("\n--- Second call: served from cache ---")
    start = time.time()
    response = ask_claude("What are the three laws of thermodynamics? Be concise.")
    print(f"Response: {response[:100]}...")
    print(f"Time: {time.time() - start:.3f}s  ← instant!")

    stats()

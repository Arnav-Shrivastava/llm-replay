"""
examples/openai_example.py

A complete working example showing llm-replay with the OpenAI SDK.

To run this example:
    pip install openai
    export OPENAI_API_KEY="your-key-here"
    python examples/openai_example.py

Run it twice. Watch the second run complete in milliseconds.
"""

import os
import time
from llm_replay_py import replay, config, stats

# Configure the cache at the top of your script, before any @replay functions.
# ttl_days=7 means responses expire after 7 days and get refreshed automatically.
# verbose=True prints a line on every cache hit or miss so you can see it working.
config(ttl_days=7, verbose=True)

import logging
logging.basicConfig(level=logging.INFO)


@replay
def classify_sentiment(text: str) -> dict:
    """
    Classifies the sentiment of a piece of text using GPT-4o-mini.
    Returns a dict with 'sentiment' and 'confidence' keys.

    This function costs ~0.001 cents per call.
    With llm-replay: only costs money the FIRST time you call it.
    """
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sentiment classifier. "
                    "Respond only with valid JSON in the format: "
                    '{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}'
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,  # Deterministic output
    )

    import json
    return json.loads(response.choices[0].message.content)


@replay
def summarize(text: str, max_words: int = 50) -> str:
    """Summarizes text to approximately max_words words."""
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Summarize this in {max_words} words or fewer: {text}",
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    sample_text = (
        "I absolutely loved this product! "
        "It exceeded all my expectations and I would definitely buy it again."
    )

    print("\n--- Run 1: First call, hits the real API ---")
    start = time.time()
    result = classify_sentiment(sample_text)
    elapsed = time.time() - start
    print(f"Result: {result}")
    print(f"Time:   {elapsed:.3f}s")

    print("\n--- Run 2: Second call, served from cache ---")
    start = time.time()
    result = classify_sentiment(sample_text)
    elapsed = time.time() - start
    print(f"Result: {result}")
    print(f"Time:   {elapsed:.3f}s  ← see how fast this is?")

    print("\n--- Force refresh: bypass cache explicitly ---")
    result = classify_sentiment(sample_text, force_refresh=True)
    print(f"Result: {result}")

    # Show performance summary
    stats()

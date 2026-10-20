"""LLM + embedding backends. Works with local Ollama or any OpenAI-compatible API.

Configure with environment variables (or Streamlit secrets, which are exposed as env vars):

  LLM_PROVIDER     ollama (default) | openai
  CHAT_MODEL       default: llama3.2 (ollama) / gpt-4o-mini (openai)
  EMBED_PROVIDER   defaults to LLM_PROVIDER
  EMBED_MODEL      default: nomic-embed-text (ollama) / text-embedding-3-small (openai)

  OLLAMA_HOST      e.g. http://localhost:11434
  OPENAI_API_KEY, OPENAI_BASE_URL          chat endpoint (Groq, OpenRouter, OpenAI, ...)
  EMBED_API_KEY, EMBED_BASE_URL            optional separate embeddings endpoint
"""
import os
import re

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", LLM_PROVIDER).lower()

CHAT_MODEL = os.getenv("CHAT_MODEL") or ("llama3.2" if LLM_PROVIDER == "ollama" else "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL") or (
    "nomic-embed-text" if EMBED_PROVIDER == "ollama" else "text-embedding-3-small")

# nomic-embed-text expects task prefixes; other models do not.
_USE_NOMIC_PREFIX = EMBED_PROVIDER == "ollama" and "nomic" in EMBED_MODEL


def embed_tag():
    """Identifies the embedding space. Vectors from different models are not comparable."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", f"{EMBED_PROVIDER}-{EMBED_MODEL}").strip("-").lower()


def _ollama_client():
    from ollama import Client
    host = os.getenv("OLLAMA_HOST")
    return Client(host=host) if host else Client()


def _openai_client(embeddings=False):
    from openai import OpenAI
    if embeddings:
        key = os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
        base = os.getenv("EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    else:
        key = os.getenv("OPENAI_API_KEY")
        base = os.getenv("OPENAI_BASE_URL")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key, base_url=base or None)


def chat_stream(messages):
    """Yield text chunks for the assistant reply."""
    if LLM_PROVIDER == "ollama":
        for chunk in _ollama_client().chat(model=CHAT_MODEL, messages=messages, stream=True):
            yield chunk.message.content or ""
    else:
        stream = _openai_client().chat.completions.create(
            model=CHAT_MODEL, messages=messages, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def embed_texts(texts, kind="document"):
    """Embed a list of strings. kind is 'document' or 'query'."""
    if not texts:
        return []
    if _USE_NOMIC_PREFIX:
        texts = [f"search_{kind}: {t}" for t in texts]
    if EMBED_PROVIDER == "ollama":
        return _ollama_client().embed(model=EMBED_MODEL, input=texts).embeddings
    resp = _openai_client(embeddings=True).embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def describe():
    return f"{LLM_PROVIDER}:{CHAT_MODEL} · embeddings {EMBED_PROVIDER}:{EMBED_MODEL}"

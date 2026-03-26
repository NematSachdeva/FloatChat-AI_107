"""LLM provider abstraction with cloud/local support and safe fallbacks."""

from typing import List, Dict, Any

import config


def _chat_with_provider(provider: str, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
    provider = provider.lower().strip()

    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    if provider == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        from openai import OpenAI

        headers: Dict[str, str] = {}
        if config.OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
        if config.OPENROUTER_APP_NAME:
            headers["X-Title"] = config.OPENROUTER_APP_NAME

        client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            default_headers=headers or None,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    if provider == "ollama":
        import ollama as _ollama

        client = _ollama.Client(host=config.OLLAMA_HOST)
        resp = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
        return resp["message"]["content"].strip()

    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def chat_completion(messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.1) -> str:
    """Run a chat completion using configured provider with simple fallback order."""

    requested_provider = (config.LLM_PROVIDER or "ollama").lower().strip()
    providers = [requested_provider]

    # If cloud provider fails, fall back to local Ollama if available.
    if requested_provider != "ollama":
        providers.append("ollama")

    last_error = None
    for provider in providers:
        try:
            return _chat_with_provider(
                provider=provider,
                model=config.LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"LLM request failed for providers {providers}: {last_error}")

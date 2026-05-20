import asyncio
import os
import httpx
from fastapi import HTTPException
from utils.logger import log_error, log_info

MISTRAL_API_BASE = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"
GEMINI_LLM_MODEL = "gemini-2.5-flash"


async def _call_gemini_llm(messages: list[dict], json_mode: bool) -> str:
    """Gemini fallback for when Mistral is rate-limited."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=502, detail="Gemini API key not configured for LLM fallback.")

    try:
        from google import genai
    except ImportError:
        raise HTTPException(status_code=502, detail="google-genai package not installed.")

    client = genai.Client(api_key=api_key)

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    other_parts  = [m["content"] for m in messages if m["role"] != "system"]
    prompt = "\n\n".join(system_parts + other_parts)
    if json_mode:
        prompt += "\n\nIMPORTANT: Return ONLY valid JSON with no markdown fences."

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_LLM_MODEL,
            contents=[{"parts": [{"text": prompt}]}],
        )
        text = response.text or ""
        log_info("Gemini LLM fallback response", model=GEMINI_LLM_MODEL, chars=len(text))
        return text
    except Exception as exc:
        log_error("Gemini LLM fallback error", model=GEMINI_LLM_MODEL, error=str(exc))
        raise HTTPException(status_code=502, detail=f"Gemini LLM fallback failed: {exc}")


async def call_llm(
    user_content: str,
    system_content: str = "You are a professional Zimbabwean educator.",
    json_mode: bool = True,
    history: list[dict] | None = None,
) -> str:
    """
    Calls the Mistral API and returns the raw text content of the reply.
    Falls back to Gemini on 429 (rate limit / capacity exceeded).

    Args:
        user_content:   The latest user message.
        system_content: System-level instruction for the model.
        json_mode:      When True (default), forces JSON output.
                        When False, returns free-form text (e.g. Markdown).
        history:        Optional prior turns as [{role, content}] dicts.
                        Inserted between the system message and the current
                        user message so multi-turn chat works correctly.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Mistral API Key not found in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_instruction = (
        f"{system_content} Always return responses ONLY in valid JSON format with no markdown fences."
        if json_mode
        else system_content
    )

    messages: list[dict] = [{"role": "system", "content": system_instruction}]

    if history:
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_content})

    payload: dict = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "max_tokens": 32768,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                MISTRAL_API_BASE, json=payload, headers=headers, timeout=120.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            log_error(f"Mistral HTTP error {status}", body=e.response.text)
            if status == 429:
                log_info("Mistral rate-limited — falling back to Gemini LLM", model=GEMINI_LLM_MODEL)
                return await _call_gemini_llm(messages, json_mode)
            raise HTTPException(status_code=502, detail="Failed to reach the AI Brain.")
        except Exception as e:
            log_error(f"Mistral error [{type(e).__name__}]", error=repr(e))
            raise HTTPException(status_code=502, detail="Failed to reach the AI Brain.")
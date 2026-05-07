import os
import httpx
from fastapi import HTTPException

MISTRAL_API_BASE = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"


async def call_llm(
    user_content: str,
    system_content: str = "You are a professional Zimbabwean educator.",
    json_mode: bool = True,
    history: list[dict] | None = None,
) -> str:
    """
    Calls the Mistral API and returns the raw text content of the reply.

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

    # Inject prior conversation turns so the model has context
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
            # Extract the actual text content from the Mistral response envelope
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            print(f"Mistral HTTP error {e.response.status_code}: {e.response.text}")
            raise HTTPException(status_code=502, detail="Failed to reach the AI Brain.")
        except Exception as e:
            print(f"Mistral error [{type(e).__name__}]: {repr(e)}")
            raise HTTPException(status_code=502, detail="Failed to reach the AI Brain.")
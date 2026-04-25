import os
import httpx
from fastapi import HTTPException

MISTRAL_API_BASE = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"


async def call_llm(user_content: str, system_content: str = "You are a professional Zimbabwean educator.") -> str:
    """
    Calls the Mistral API and returns the raw text content of the reply.
    Forces JSON output so callers can parse structured responses reliably.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Mistral API Key not found in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {
                "role": "system",
                "content": f"{system_content} Always return responses ONLY in valid JSON format with no markdown fences.",
            },
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 32768,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                MISTRAL_API_BASE, json=payload, headers=headers, timeout=45.0
            )
            response.raise_for_status()
            data = response.json()
            # Extract the actual text content from the Mistral response envelope
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            print(f"Mistral HTTP error {e.response.status_code}: {e.response.text}")
            raise HTTPException(status_code=502, detail="Failed to reach the AI Brain.")
        except Exception as e:
            print(f"Mistral error: {str(e)}")
            raise HTTPException(status_code=502, detail="Failed to reach the AI Brain.")
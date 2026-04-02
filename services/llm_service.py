import os
import httpx
import json
from fastapi import HTTPException

# The Brain's Connection Details
MISTRAL_API_BASE = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest" # High intelligence, low cost

async def call_llm(user_content: str, system_content: str = "You are a professional Zimbabwean educator."):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Mistral API Key not found in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # In the grand scheme, we force Mistral to give us JSON so the frontend doesn't break.
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {
                "role": "system", 
                # We combine your persona with the "Must return JSON" rule
                "content": f"{system_content} Return responses ONLY in valid JSON format."
            },
            {"role": "user", "content": user_content}
        ],
        "response_format": {"type": "json_object"}
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(MISTRAL_API_BASE, json=payload, headers=headers, timeout=45.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Mistral Error: {str(e)}")
            raise HTTPException(status_code=502, detail="Failed to reach the AI Brain.")

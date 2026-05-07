import json
from services.llm_service import call_llm

SYSTEM_PROMPT = (
    "You are an expert at parsing exam papers. "
    "Extract every discrete question from the provided text and return a JSON array. "
    "Each element must have the keys: question_text (string), marks_hint (integer or null), "
    "topic_hint (string or null). "
    "Return only the JSON array with no extra commentary."
)

MAX_QUESTIONS = 150


async def extract_questions_from_text(text: str) -> list[dict]:
    try:
        raw = await call_llm(text, system_content=SYSTEM_PROMPT)
        data = json.loads(raw)
        # The LLM may return {"questions": [...]} or a bare array wrapped in an object
        if isinstance(data, list):
            questions = data
        elif isinstance(data, dict):
            # Try common wrapper keys
            for key in ("questions", "items", "data", "result"):
                if isinstance(data.get(key), list):
                    questions = data[key]
                    break
            else:
                questions = []
        else:
            questions = []

        return questions[:MAX_QUESTIONS]
    except Exception:
        return []

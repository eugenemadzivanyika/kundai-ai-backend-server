import json
from services.llm_service import call_llm
from services.asag_service import parse_json_from_ai


async def generate_reteach_card(topic: str, subtopic: str, subject: str, avg_mastery: float) -> dict:
    mastery_pct = round(avg_mastery * 100)

    system_prompt = (
        "You are an expert Zimbabwean secondary school teacher. "
        "You write clear, practical re-teaching materials for topics where students are struggling."
    )

    user_prompt = f"""
A class is underperforming on the following topic.

Subject: {subject}
Topic: {topic}
Subtopic: {subtopic}
Class average mastery: {mastery_pct}%

Generate a re-teach card with exactly two parts:

1. interventionScript — A 2-minute spoken script (150–250 words) for the teacher to deliver to the class.
   Use plain, friendly language. Reference common misconceptions. End with a summary statement.

2. exitTicket — Exactly 3 questions to check understanding after the intervention.
   Mix at least one MCQ (4 options) and at least one short_answer question.
   Each question must have a correct answer.

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "interventionScript": "...",
  "exitTicket": [
    {{
      "question": "...",
      "type": "mcq",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "A. ..."
    }},
    {{
      "question": "...",
      "type": "short_answer",
      "options": [],
      "answer": "..."
    }},
    {{
      "question": "...",
      "type": "mcq",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "B. ..."
    }}
  ]
}}
"""
    response = await call_llm(user_prompt, system_content=system_prompt, json_mode=True)
    parsed = parse_json_from_ai(response)

    return {
        "topic": topic,
        "subtopic": subtopic,
        "avg_mastery": avg_mastery,
        "difficulty_score": max(0.0, min(1.0, 1.0 - avg_mastery)),
        "intervention_script": parsed.get("interventionScript", ""),
        "exit_ticket": parsed.get("exitTicket", [])[:3],
    }

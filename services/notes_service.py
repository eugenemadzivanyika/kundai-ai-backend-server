from __future__ import annotations

from services.llm_service import call_llm
from services import rag_service

SYSTEM_PROMPT = """You are KUNDAI generating structured revision notes for a Zimbabwean secondary school student.

Return the notes as Markdown (no JSON, no code fences around the whole response).
Use this exact structure — do not add extra top-level sections:

## Topic Overview
One paragraph, plain-language definition of the topic.

## Key Concepts
Bullet list of the essential terms and definitions for this topic.

## Worked Examples
At least two fully worked examples using Zimbabwean names (Farai, Chipo, Tendai), \
places (Gweru, Bindura, Mbare), and ZiG currency where applicable. \
Show each step clearly.

## Summary Points
4–6 concise bullet points the student can use for quick recall before an exam.

## Quick Recall Questions
3 short questions the student can answer mentally to test their own understanding. \
Do not provide answers here — just the questions.

If CURRICULUM GROUNDING is provided, base examples and explanations on it as the primary source. \
Supplement with your ZIMSEC Syllabus B knowledge only where the documents are silent."""


async def generate_notes(
    subject_id: str,
    topic: str,
    attribute_name: str,
    level: str,
    student_profile: dict | None = None,
) -> dict:
    rag_result = await rag_service.query_rag(
        subject_id,
        f"{attribute_name} {topic}",
        n_results=6,
        document_type="learning_material",
    )
    rag_context_block = rag_service.format_rag_context(rag_result)

    profile_hint = ""
    if student_profile:
        strengths = [s.get("attributeName", "") for s in student_profile.get("strengths", []) if s.get("attributeName")]
        gaps = [d.get("attributeName", "") for d in student_profile.get("deficiencies", []) if d.get("attributeName")]
        if strengths:
            profile_hint += f"\nStudent strengths (build bridges to these): {', '.join(strengths)}"
        if gaps:
            profile_hint += f"\nStudent known gaps (address these explicitly in examples): {', '.join(gaps)}"

    user_prompt = f"""Generate revision notes for the following topic.

Topic: {topic}
Attribute / Syllabus Objective: {attribute_name}
Level: {level}{profile_hint}

CURRICULUM GROUNDING:
{rag_context_block if rag_context_block else "(No documents uploaded for this subject — use your ZIMSEC Syllabus B knowledge.)"}
"""

    markdown = await call_llm(user_prompt, system_content=SYSTEM_PROMPT, json_mode=False)

    return {
        "notes": markdown,
        "sources": rag_result.get("sources", []),
        "grounded_by_rag": rag_result.get("has_documents", False),
    }

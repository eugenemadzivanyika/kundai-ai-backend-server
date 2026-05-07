from __future__ import annotations

from services.llm_service import call_llm
from services import rag_service

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_NOTES_SYSTEM_PROMPT = """You are KUNDAI generating structured revision notes for a Zimbabwean secondary school student.

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

_CHAT_SYSTEM_TEMPLATE = """You are KUNDAI, a ZIMSEC tutor helping a student understand a specific topic.

The student is reading the notes shown below. Answer their question using those notes as your \
primary source. Do not introduce unrelated concepts. Keep the answer concise (3–5 sentences max) \
unless a worked example is genuinely needed to clarify the point.

Use plain, friendly language suitable for a Zimbabwean secondary school student. \
Where helpful, reference the specific worked examples, names, or numbers already in the notes \
so the student can connect your answer back to what they are reading.

## NOTES THE STUDENT IS READING
{notes_context}
{rag_section}"""

# ---------------------------------------------------------------------------
# generate_notes
# ---------------------------------------------------------------------------

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

    markdown = await call_llm(user_prompt, system_content=_NOTES_SYSTEM_PROMPT, json_mode=False)

    return {
        "notes": markdown,
        "sources": rag_result.get("sources", []),
        "grounded_by_rag": rag_result.get("has_documents", False),
    }


# ---------------------------------------------------------------------------
# answer_topic_question  (Phase 1 — Notes-Aware Topic Chat)
# ---------------------------------------------------------------------------

# Maximum characters of notes to include in each chat turn.
# Mistral's 32k context handles ~1 000 tokens comfortably; 3 000 chars ≈ 750 tokens.
_MAX_NOTES_CHARS = 3_000


async def answer_topic_question(
    subject_id: str,
    topic: str,
    notes_context: str,
    question: str,
    history: list[dict],
) -> dict:
    """
    Answers a student's question in the context of the notes they are currently reading.

    Args:
        subject_id:    MongoDB _id of the course — used for supplementary RAG lookup.
        topic:         Human-readable topic title (used in the RAG query).
        notes_context: The Markdown notes the student is currently reading.
        question:      The student's latest chat message.
        history:       Prior turns [{role: 'user'|'assistant', content: str}].

    Returns:
        {answer: str, grounded_by_rag: bool}
    """
    # 1. Supplementary RAG: pull related curriculum chunks in case the notes
    #    don't fully cover the question.
    rag_result = await rag_service.query_rag(
        subject_id,
        question,
        n_results=4,
        document_type="learning_material",
    )

    # 2. Truncate notes to avoid blowing the context window on very long notes.
    truncated_notes = notes_context[:_MAX_NOTES_CHARS]
    if len(notes_context) > _MAX_NOTES_CHARS:
        truncated_notes += "\n\n*(notes truncated for brevity)*"

    # 3. Build the rag section only when documents exist.
    if rag_result.get("has_documents"):
        rag_block = rag_service.format_rag_context(rag_result)
        rag_section = f"\n\n## ADDITIONAL CURRICULUM CONTEXT\n{rag_block}"
    else:
        rag_section = ""

    # 4. Assemble the system prompt with notes and optional RAG content injected.
    system_prompt = _CHAT_SYSTEM_TEMPLATE.format(
        notes_context=truncated_notes,
        rag_section=rag_section,
    )

    # 5. Call the LLM with the full conversation history for multi-turn coherence.
    answer = await call_llm(
        question,
        system_content=system_prompt,
        history=history,
        json_mode=False,
    )

    return {
        "answer": answer,
        "grounded_by_rag": rag_result.get("has_documents", False),
    }
# from click import prompt
from services.llm_service import call_llm
from services import rag_service
import asyncio
import json
import random
import re


def parse_json_from_ai(response):
    """
    Handles both:
     - a plain string (from the updated llm_service.call_llm which returns text directly)
     - the raw Mistral response dict (legacy callers)
    """
    try:
        if isinstance(response, str):
            content = response
        elif isinstance(response, dict) and 'choices' in response:
            content = response['choices'][0]['message']['content']
        else:
            content = str(response)

        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```[a-z]*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)

        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(content)
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"error": "Invalid AI JSON", "raw": content if 'content' in locals() else str(e)}


# ─── 1. PERSONALIZED THEORY ──────────────────────────────────────────────────

async def generate_personalized_theory(payload: dict) -> dict:
    attr = payload.get("attribute_details", {})
    mastery = payload.get("initial_mastery", 0.1)
    misconceptions = payload.get("misconceptions", [])
    subject_id = payload.get("subject_id")

    if subject_id:
        rag_result = await rag_service.query_rag(
            subject_id,
            f"{attr.get('name', '')} {attr.get('description', '')}",
            n_results=4,
            document_type="learning_material",
        )
    else:
        rag_result = {"chunks": [], "sources": [], "distances": [], "has_documents": False}
    rag_context_block = rag_service.format_rag_context(rag_result)

    system_prompt = """You are KundAI, an expert ZIMSEC tutor.
    Your goal is to explain complex math concepts using relatable Zimbabwean scenarios
    (like Kombi fares, tuckshop profit, or market prices).

    SURGICAL INSTRUCTION: If 'misconceptions' are provided, start the module by
    gently addressing these specific errors. Use the 'Aha!' moment technique to
    show why the previous logic was a common pitfall.

    If CURRICULUM GROUNDING is provided, base your examples, figures, and explanations \
on it as your primary source. Supplement with your own ZIMSEC knowledge only where the \
documents are silent.

    Always use Markdown for formatting. Include 2 Checkpoint questions."""

    gap_context = ""
    if misconceptions:
        gap_context = f"\nRECENT ERRORS IDENTIFIED: {', '.join(misconceptions)}"

    user_prompt = f"""
    Topic: {attr.get('name')} ({attr.get('attribute_id')})
    Level: {attr.get('level')}
    Description: {attr.get('description')}
    Prerequisites: {attr.get('prerequisites')}
    Student Mastery: {mastery:.2f}
    {gap_context}

    TASK: Generate a 'Mini-Module' theory block.
    1. Address the specific identified misconceptions first.
    2. Use a relatable Zim context (e.g., measuring a round vegetable garden or a soccer center circle).
    3. Explain how this builds on prerequisites.

    CURRICULUM GROUNDING:
    {rag_context_block if rag_context_block else "Use your ZIMSEC knowledge — no curriculum documents uploaded for this subject."}

    Return JSON format: {{ "title": "...", "content": "..." }}
    """

    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)


# ─── 2. PRACTICE (Focuses on Hints and Step-by-Step) ─────────────────────────

async def generate_practice_set(payload: dict) -> dict:
    profile = payload.get("profile", {})
    deficiencies = profile.get("deficiencies", [])
    subject_id = payload.get("subject_id")

    main_topic = deficiencies[0].get('attributeName', 'Mathematics') if deficiencies else "Mathematics"

    if subject_id:
        query = " ".join(d.get("attributeName", "") for d in deficiencies[:3] if d.get("attributeName"))
        rag_result = await rag_service.query_rag(
            subject_id, query or main_topic, n_results=4, document_type="learning_material"
        )
    else:
        rag_result = {"chunks": [], "sources": [], "distances": [], "has_documents": False}
    rag_context_block = rag_service.format_rag_context(rag_result)

    system_prompt = """You are KUNDAI's Practice Engine.
    Generate practice problems where the 'hints' are specifically designed to
    stop the student from making the logical errors identified in their profile.

    The 'steps' should provide a ZIMSEC-aligned walkthrough."""

    user_prompt = f"""
    TARGET DEFICIENCIES: {json.dumps(deficiencies)}

    TASK: Generate 5 localized practice problems.
    For each problem, include:
    1. A 'hint' that addresses the specific misconception.
    2. A 'tutor_explanation' for the hint (what the coach says if they ask 'Why that hint?')
    3. Step-by-step solution logic.

    CURRICULUM GROUNDING:
    {rag_context_block if rag_context_block else "Use your ZIMSEC knowledge — no curriculum documents uploaded for this subject."}

    STRICT JSON SCHEMA:
    {{
      "title": "Surgical Practice: {main_topic}",
      "problems": [
        {{
          "question": "The problem text",
          "hint": "The student-facing hint",
          "tutor_explanation": "Internal logic for the AI coach about this hint",
          "steps": ["Step 1...", "Step 2..."],
          "final_answer": "The final result"
        }}
      ]
    }}
    """

    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)


# ─── 3. MASTERY QUIZ ─────────────────────────────────────────────────────────

async def generate_mastery_quiz(payload: dict) -> dict:
    attr = payload.get("attribute_details", {})
    misconceptions = payload.get("misconceptions", [])
    subject_id = payload.get("subject_id")

    if subject_id:
        rag_result = await rag_service.query_rag(
            subject_id,
            f"{attr.get('name', '')} {attr.get('description', '')}",
            n_results=4,
            document_type="learning_material",
        )
    else:
        rag_result = {"chunks": [], "sources": [], "distances": [], "has_documents": False}
    rag_context_block = rag_service.format_rag_context(rag_result)

    system_prompt = """You are KUNDAI, a strict ZIMSEC Examiner.
    Create rigorous multiple-choice questions that mimic the Paper 1 style.

    SURGICAL INSTRUCTION: If 'misconceptions' are provided, at least one 'distractor'
    (wrong option) per question MUST be the result of that specific misconception.
    The 'explanation' should explain why that specific distractor is a trap."""

    gap_context = f"\nPAST MISCONCEPTIONS: {', '.join(misconceptions)}" if misconceptions else ""

    user_prompt = f"""
    Topic: {attr.get('name')} | Level: {attr.get('level')}
    {gap_context}

    Task: Create a 5-question Multiple Choice Quiz.
    Include traps based on the listed misconceptions.

    CURRICULUM GROUNDING:
    {rag_context_block if rag_context_block else "Use your ZIMSEC knowledge — no curriculum documents uploaded for this subject."}

    STRICT JSON SCHEMA:
    {{
      "title": "Mastery Check: {attr.get('name')}",
      "questions": [
        {{
          "question": "...",
          "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
          "correct_answer": "...",
          "explanation": "..."
        }}
      ]
    }}
    """
    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)


# ─── 4. UNIT / SUBJECT CHALLENGE ─────────────────────────────────────────────

async def generate_unit_challenge(payload: dict) -> dict:
    """
    payload:
      attributes   - list of CourseAttribute dicts for the unit
      unit_title   - display name of the unit
      subject_name - display name of the subject
      count        - number of questions to generate (default 10)
      difficulty   - 'easy' | 'medium' | 'hard' (default 'medium')
    """
    attributes   = payload.get("attributes", [])
    unit_title   = payload.get("unit_title", "Mathematics Unit")
    subject_name = payload.get("subject_name", "Mathematics")
    count        = payload.get("count", 10)
    difficulty   = payload.get("difficulty", "medium")
    subject_id   = payload.get("subject_id")

    attr_lines = []
    for a in attributes:
        attr_lines.append(
            f"- {a.get('name', 'Topic')} ({a.get('attribute_id', '')}): {a.get('description', '')}"
        )
    attr_summary = "\n".join(attr_lines) if attr_lines else f"General {unit_title} content"

    if subject_id:
        query = f"{unit_title} {subject_name}"
        rag_result = await rag_service.query_rag(
            subject_id, query, n_results=4, document_type="learning_material"
        )
    else:
        rag_result = {"chunks": [], "sources": [], "distances": [], "has_documents": False}
    rag_context_block = rag_service.format_rag_context(rag_result)

    system_prompt = f"""You are KUNDAI, a ZIMSEC Mathematics tutor creating a unit challenge for students.
Generate {count} multiple-choice questions covering the unit: "{unit_title}" from {subject_name}.
Use Zimbabwean context (names like Farai/Chipo, ZiG currency, local settings).
Difficulty level: {difficulty}.
Each question must have exactly 4 options (A, B, C, D) with one correct answer."""

    user_prompt = f"""
UNIT: {unit_title}
ATTRIBUTES COVERED:
{attr_summary}

CURRICULUM GROUNDING:
{rag_context_block if rag_context_block else "Use your ZIMSEC knowledge — no curriculum documents uploaded for this subject."}

Generate exactly {count} multiple-choice questions spread across these attributes.

RETURN ONLY this JSON — no markdown, no extra text:
{{
  "title": "{unit_title} Challenge",
  "questions": [
    {{
      "id": "q1",
      "prompt": "The question text here",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "correctOptionIndex": 0,
      "explanation": "Brief explanation of the correct answer"
    }}
  ]
}}

correctOptionIndex is the 0-based index into the options array (0=A, 1=B, 2=C, 3=D).
"""

    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)


# ─── 5. PERSONALISED SUBJECT CHALLENGE (Phase 3) ─────────────────────────────

async def generate_subject_challenge(payload: dict) -> dict:
    """
    Generates a personalised subject-level challenge weighted toward the student's
    weakest attributes.

    payload:
      subject_id       - MongoDB _id of the course (used for RAG)
      subject_name     - display name
      student_attrs    - list of {name, attribute_id, description, mastery}
                         where mastery is a float 0–1 from StudentAttribute.currentMastery
      count            - total questions to generate (default 12)
      difficulty       - 'easy' | 'medium' | 'hard' (default 'medium')

    Returns:
      { title, questions: [{id, prompt, options, correctOptionIndex, explanation}],
        weak_topic_count: int }
    """
    subject_id   = payload.get("subject_id")
    subject_name = payload.get("subject_name", "the subject")
    attrs        = payload.get("student_attrs", [])
    count        = int(payload.get("count", 12))
    difficulty   = payload.get("difficulty", "medium")

    if not attrs:
        return {
            "title": f"{subject_name} Challenge",
            "questions": [],
            "weak_topic_count": 0,
            "error": "No attributes provided",
        }

    # ── Allocate questions: bottom 40 % of attrs (by mastery) get 60 % of questions ──
    sorted_attrs = sorted(attrs, key=lambda a: float(a.get("mastery", 0)))

    split       = max(1, len(sorted_attrs) * 4 // 10)
    weak_attrs  = sorted_attrs[:split]
    other_attrs = sorted_attrs[split:]

    weak_count  = min(count, round(count * 0.6))
    other_count = count - weak_count

    # ── Inner helper: generate n questions for a given attr subset ──
    async def _gen(attr_list: list, n: int) -> list:
        if not attr_list or n == 0:
            return []

        rag_query = " ".join(a.get("name", "") for a in attr_list[:3])
        if subject_id:
            rag_result = await rag_service.query_rag(
                subject_id, rag_query, n_results=4, document_type="learning_material"
            )
        else:
            rag_result = {"chunks": [], "has_documents": False}
        rag_block = rag_service.format_rag_context(rag_result)

        attr_lines = [
            f"- {a.get('name', 'Topic')} (mastery {round(float(a.get('mastery', 0)) * 100)}%)"
            for a in attr_list
        ]

        system_prompt = (
            f"You are KUNDAI creating a personalised challenge for a student.\n"
            f"Generate {n} multiple-choice questions on: "
            f"{', '.join(a.get('name', '') for a in attr_list)}.\n"
            f"Use Zimbabwean context (names like Farai/Chipo, ZiG currency, local places).\n"
            f"Difficulty: {difficulty}. Each question has exactly 4 options (A–D), one correct."
        )

        user_prompt = f"""SUBJECT: {subject_name}
TOPICS (with student mastery):
{chr(10).join(attr_lines)}

CURRICULUM GROUNDING:
{rag_block if rag_block else "Use ZIMSEC knowledge — no documents uploaded for this subject."}

Return ONLY this JSON:
{{
  "questions": [
    {{
      "id": "q1",
      "prompt": "...",
      "options": ["A text", "B text", "C text", "D text"],
      "correctOptionIndex": 0,
      "explanation": "..."
    }}
  ]
}}"""

        raw    = await call_llm(user_prompt, system_content=system_prompt)
        parsed = parse_json_from_ai(raw)
        return parsed.get("questions", [])

    # ── Run both batches in parallel ──
    weak_qs, other_qs = await asyncio.gather(
        _gen(weak_attrs, weak_count),
        _gen(other_attrs, other_count),
    )

    all_questions = (weak_qs + other_qs)[:count]
    random.shuffle(all_questions)

    # Re-index question ids after shuffle so they are sequential
    for i, q in enumerate(all_questions, start=1):
        q["id"] = f"q{i}"

    return {
        "title": f"{subject_name} — Personalised Challenge",
        "questions": all_questions,
        "weak_topic_count": len(weak_attrs),
    }
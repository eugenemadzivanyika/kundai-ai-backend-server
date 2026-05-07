from services.llm_service import call_llm
from services import rag_service
import json
import re

def _fix_invalid_json_escapes(s: str) -> str:
    """
    Mistral sometimes emits bare LaTeX backslashes (e.g. \leq, \geq) that are
    invalid JSON escape sequences.  This pass doubles every backslash that is
    NOT followed by a recognised JSON escape character so that json.loads()
    can succeed.  Valid sequences kept as-is: \" \\ \/ \b \f \n \r \t \\uXXXX
    """
    valid = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in valid:
                out.append('\\')
                out.append(nxt)
                i += 2
                if nxt == 'u':          # keep the 4 hex digits verbatim
                    out.append(s[i:i + 4])
                    i += 4
            else:
                out.append('\\\\')      # escape the rogue backslash
                out.append(nxt)
                i += 2
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


def _rescue_truncated_json(text: str) -> dict | None:
    """
    When the model output is cut off mid-question (token limit hit), try to recover
    all *complete* questions by closing the JSON array at the last clean boundary.
    A clean boundary is a position where three consecutive closing braces appear —
    these close the innermost value, solution_schema, and the question object itself.
    """
    arr_m = re.search(r'"questions"\s*:\s*\[', text)
    if not arr_m:
        return None

    name_m = re.search(r'"assessment_name"\s*:\s*"([^"]*)"', text)
    prefix = f'{{"assessment_name": "{name_m.group(1) if name_m else "Assessment"}", "questions": ['

    # Collect every position that ends with }}} — a likely question boundary
    candidate_ends = [m.end() for m in re.finditer(r'\}\s*\}\s*\}', text[arr_m.start():])]

    for offset in reversed(candidate_ends):
        attempt = text[arr_m.start(): arr_m.start() + offset] + '\n  ]\n}'
        for payload in [attempt, _fix_invalid_json_escapes(attempt)]:
            try:
                result = json.loads(payload)
                if isinstance(result.get('questions'), list) and result['questions']:
                    print(f"[rescue] recovered {len(result['questions'])} question(s) from truncated response")
                    return result
            except Exception:
                continue

    return None


def parse_json_from_ai(response: str):
    # call_llm already extracts the content string from the API envelope
    candidates = [response]

    # Also try with invalid escape sequences repaired (catches bare \leq, \geq, etc.)
    fixed = _fix_invalid_json_escapes(response)
    if fixed != response:
        candidates.append(fixed)

    for text in candidates:
        try:
            match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            payload = match.group(0) if match else text
            return json.loads(payload)
        except Exception:
            continue

    # Last resort: the response was likely truncated — recover whatever complete questions exist
    rescued = _rescue_truncated_json(response)
    if rescued:
        return rescued

    print(f"Extraction Error: could not parse AI response after escape repair and rescue")
    return {"error": "Invalid AI JSON", "raw": response}

DIFFICULTY_GUIDE = {
    "easy": {
        "label": "Easy — Formative / Classwork level",
        "description": (
            "Target Form 1-2 learners or basic recall. "
            "Questions test direct recall and single-concept application only. "
            "One or two steps maximum. No combined concepts. "
            "A learner who has attended class and read their notes should answer correctly. "
            "Equivalent to textbook exercise questions, NOT test or exam standard."
        ),
        "scoring": "MCQ/T-F: 1 mark each. Short answer: 1-3 marks. Essay: not recommended at this level.",
    },
    "medium": {
        "label": "Medium — Internal School Test / End-of-Term level",
        "description": (
            "Target Form 3 or internal school exam standard. "
            "Questions require applying concepts to slightly unfamiliar situations. "
            "May combine two related concepts. Requires some reasoning and 2-4 steps. "
            "Equivalent to a school test or class assessment, NOT the ZIMSEC final exam. "
            "A well-prepared learner should score 60-75%."
        ),
        "scoring": "MCQ/T-F: 1-2 marks each. Short answer: 3-6 marks. Essay: 8-10 marks.",
    },
    "hard": {
        "label": "Hard — ZIMSEC O-Level Examination Standard (4008/4028)",
        "description": (
            "Questions must match the rigor of the actual ZIMSEC O-Level Mathematics paper. "
            "Target Form 4 examination level. Non-routine problems that require combining multiple concepts. "
            "Multi-step solutions with logical progression. "
            "Questions should differentiate A-grade from B-grade candidates. "
            "Include real-world application contexts, proof, or deduction where appropriate. "
            "A well-prepared candidate should find these challenging but fair."
        ),
        "scoring": "MCQ/T-F: 2 marks each. Short answer: 5-10 marks. Essay: 10-15 marks.",
    },
}

PAPER_STYLE_GUIDE = {
    "paper1": {
        "label": "Paper 1 style",
        "description": (
            "Questions must be SHORT and DIRECT — single-step or two-step at most. "
            "No calculator context. Tests immediate recall and straightforward application. "
            "Each question should be answerable in under 2 minutes. "
            "Avoid multi-part questions. Keep working minimal and answers concise. "
            "FORBIDDEN: Do NOT generate multiple_choice or true_false questions. "
            "Only short_answer is allowed."
        ),
    },
    "paper2": {
        "label": "Paper 2 style",
        "description": (
            "Questions must be STRUCTURED and MULTI-STEP. "
            "Short answer questions (short_answer type) mirror Section A — structured, "
            "requiring some working shown, 3-10 marks each. "
            "Essay/long questions (essay type) mirror Section B — extended multi-step problems "
            "requiring full working, clearly laid-out solutions, 10-15 marks each. "
            "Questions may be multi-part (e.g. (a), (b), (c)). "
            "FORBIDDEN: Do NOT generate multiple_choice or true_false questions. "
            "Only short_answer and essay are allowed."
        ),
    },
    "both": {
        "label": "Mixed Paper 1 and Paper 2 style",
        "description": (
            "Generate a mix: short_answer questions should follow Paper 1 style "
            "(direct, single-step, no calculator context), while essay questions should "
            "follow Paper 2 style (multi-step, show full working, multi-part where appropriate)."
        ),
    },
}

async def generate_syllabus_questions(payload: dict) -> dict:
    # 1. UNPACK ALL CONTEXT FIELDS
    context           = payload.get("syllabus_context", [])
    meta              = payload.get("meta_context", {})
    count             = payload.get("count", 5)
    max_score         = payload.get("maxScore", 100)
    difficulty        = payload.get("difficulty", "medium")
    distribution      = payload.get("distribution", {})
    math_paper_type   = payload.get("math_paper_type")  # "paper1" | "paper2" | "both" | None
    rag_context_block = payload.get("rag_context_block")  # pre-formatted style guide from Node RAG layer
    subject_id        = payload.get("subject_id")

    # 2. DIFFICULTY & PAPER STYLE CONTEXT
    diff_guide   = DIFFICULTY_GUIDE.get(difficulty, DIFFICULTY_GUIDE["medium"])
    paper_style  = PAPER_STYLE_GUIDE.get(math_paper_type) if math_paper_type else None

    difficulty_block = (
        f"\nDIFFICULTY — {diff_guide['label'].upper()}:\n"
        f"{diff_guide['description']}\n"
        f"Scoring guidance for this level: {diff_guide['scoring']}"
    )
    paper_style_block = (
        f"\nPAPER STYLE — {paper_style['label'].upper()}:\n{paper_style['description']}"
        if paper_style else ""
    )

    # 3. MongoDB QuestionBank RAG style guide (injected verbatim when available)
    rag_block = f"\n\n{rag_context_block}" if rag_context_block else ""

    # 3b. ChromaDB question paper RAG (from teacher-uploaded PDFs)
    doc_rag_block = ""
    if subject_id:
        query = " ".join(a.get("name", "") for a in context[:3] if a.get("name"))
        if query:
            doc_rag = await rag_service.query_rag(
                subject_id, query, n_results=6, document_type="question_paper"
            )
            doc_rag_block = rag_service.format_rag_context(doc_rag)

    uploaded_examples_block = (
        f"\n\nUPLOADED PAST PAPER EXAMPLES (from teacher-uploaded PDFs):\n"
        f"{doc_rag_block}\n"
        f"Use these examples alongside the style guide above when constructing questions."
        if doc_rag_block else ""
    )

    # 4. SYSTEM PERSONA
    system_prompt = f"""You are a Senior ZIMSEC Examination Officer for Mathematics Syllabus B.
    Your task is to create a formal '{meta.get('assessment_type', 'Exercise')}' titled '{meta.get('name')}'.

    CRITICAL CONTEXT:
    - Target Level: {meta.get('level', 'General ZIMSEC')}
    - Teacher Constraints: {meta.get('description', 'Standard ZIMSEC assessment.')}
    - Localization: Use Zimbabwean names (Farai, Chipo), locations (Gweru, Bindura), and ZiG currency.
    - Tone: Formal, rigorous ZIMSEC examination tone (4008/4028 standard).{difficulty_block}{paper_style_block}{rag_block}{uploaded_examples_block}"""

    # 5. TYPE MIX CALCULATION
    allowed_types = [q_type for q_type, pct in distribution.items() if pct > 0]
    type_constraints = []
    for q_type, percentage in distribution.items():
        if percentage > 0:
            num = round((percentage / 100) * count)
            type_constraints.append(f"- {num} {q_type.replace('_', ' ')} questions")
    type_mix_str = "\n".join(type_constraints)
    allowed_types_str = " | ".join(allowed_types) if allowed_types else "short_answer | essay"

    # 6. SYLLABUS OBJECTIVES
    objectives_detail = []
    for a in context:
        detail = (
            f"### OBJECTIVE ID: {a.get('_id') or a.get('attribute_id')}\n"
            f"- TOPIC: {a.get('name')}\n"
            f"- DESCRIPTION: {a.get('description')}\n"
        )
        objectives_detail.append(detail)
    objectives_str = "\n".join(objectives_detail)

    # 7. THE SURGICAL TASK (User Prompt)
    user_prompt = f"""
    Generate exactly {count} questions based on these ZIMSEC objectives:
    {objectives_str}

    REQUIRED QUESTION MIX:
    {type_mix_str}

    TOTAL MARKS: {max_score} points. The sum of all question points MUST equal exactly {max_score}.
    SCORING LOGIC for {difficulty.upper()} difficulty: {diff_guide['scoring']}
    Distribute marks proportionally across the {count} questions to reach the {max_score} total.

    STRICT JSON SCHEMA — every field must match exactly:
    {{
      "assessment_name": "{meta.get('name', 'ZIMSEC Assessment')}",
      "questions": [
        {{
          "primaryAttributeId": "string (MUST match the OBJECTIVE ID used)",
          "questionText": "string — for multipart questions this is the shared context/stem only (e.g. 'The diagram shows triangle ABC where...'). For single questions this is the full question.",
          "questionType": "{allowed_types_str}",
          "options": {{ "A": "str", "B": "str", "C": "str", "D": "str" }},
          "correctAnswer": "MUST be a plain string. Use ONLY for single (non-multipart) questions. Set to empty string '' when parts is non-empty.",
          "points": "number — TOTAL marks for this question. MUST equal sum of part maxPoints when parts is non-empty.",
          "difficulty": "{difficulty}",
          "explanation": "string",
          "solution_schema": {{
            "steps": ["string — one working step per element"],
            "final_answer": "string — set to empty string '' when parts is non-empty"
          }},
          "parts": [
            {{
              "text": "string — the specific sub-question text e.g. '(a) Calculate the area of the triangle.'",
              "type": "short_answer | essay | multiple_choice | true_false",
              "options": [],
              "correctAnswer": "string — the model answer for THIS part only",
              "maxPoints": "number — marks for this part only",
              "solution_schema": {{
                "steps": ["string"],
                "final_answer": "string"
              }}
            }}
          ]
        }}
      ]
    }}

    MULTIPART RULES:
    - Use parts[] ONLY for structured questions that naturally split into labelled sub-tasks (a), (b), (c).
    - When parts[] is non-empty: correctAnswer on the parent MUST be "", points MUST equal sum(part.maxPoints).
    - When parts[] is empty []: correctAnswer on the parent MUST be filled — this is a single question.
    - USE parts for: Paper 2 essay/structured questions, multi-step calculation problems with distinct sub-tasks, questions that award marks per sub-part.
    - DO NOT USE parts for: multiple_choice, true_false, or any question answerable in a single response.

    OTHER RULES:
    - correctAnswer and solution_schema.final_answer are ALWAYS plain strings, never JSON objects or arrays.
    - If questionType is 'short_answer' or 'essay' with no parts, set options to an empty object {{}}.
    - Use LaTeX notation for all mathematical expressions (e.g. $\\\\frac{{1}}{{2}}$, $x^2$).
    - CRITICAL — LaTeX in JSON requires double backslashes. EVERY LaTeX command backslash MUST be doubled: write $\\\\leq$ not $\\leq$, $\\\\geq$ not $\\geq$, $\\\\times$ not $\\times$, $\\\\frac{{a}}{{b}}$ not $\\frac{{a}}{{b}}$. A single backslash before a letter is invalid JSON and will break parsing.
    """

    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)
import re
import json
from typing import Literal
from services.llm_service import call_llm

SYSTEM_PROMPT = (
    "You are classifying an educational document excerpt. "
    "Strong signals for QUESTION PAPER: numbered questions (1. 2. 3.), mark allocations "
    "in brackets like [1] [2] [3] [4], action verbs such as Evaluate/Simplify/Find/Calculate/"
    "Solve/Express/Given/Factorise/Show/Prove at the start of lines, sub-parts labelled a) b) c). "
    "Strong signals for LEARNING MATERIAL: prose explanations, definitions, worked examples "
    "with full solutions shown step-by-step, chapter introductions, syllabi. "
    "A revision book that is MOSTLY questions with mark allocations should be classified as "
    "question_paper even if it has a brief preface. "
    "Return JSON {\"type\": \"question_paper\"} or {\"type\": \"learning_material\"}."
)

# Patterns that strongly indicate a question paper
_MARKS_RE = re.compile(r'\[\d+\]')
_NUMBERED_Q_RE = re.compile(
    r'^\s*\d+[\.\)]\s*(?:a\)?\.?\s+)?'
    r'(?:Evaluate|Simplify|Find|Calculate|Solve|Express|Given|Write|Show|Prove|'
    r'Expand|Factorise|Factorize|Draw|State|Describe|Explain|Convert|Copy|Determine)',
    re.MULTILINE | re.IGNORECASE,
)


def _heuristic_classify(text: str) -> Literal["question_paper", "learning_material"] | None:
    """Returns a classification if heuristic signals are conclusive, else None."""
    marks = _MARKS_RE.findall(text)
    if len(marks) >= 8:
        return "question_paper"

    numbered = _NUMBERED_Q_RE.findall(text)
    if len(numbered) >= 6:
        return "question_paper"

    return None


def _build_sample(full_text: str, words_per_segment: int = 500) -> str:
    """Sample from start, middle, and end so cover/preface pages don't dominate."""
    words = full_text.split()
    total = len(words)
    cap = words_per_segment

    if total <= cap * 3:
        return full_text

    start = words[:cap]
    mid_start = max(cap, total // 2 - cap // 2)
    middle = words[mid_start: mid_start + cap]
    end = words[max(mid_start + cap, total - cap):]

    return " ".join(start) + " [...] " + " ".join(middle) + " [...] " + " ".join(end)


async def classify_document(text: str) -> Literal["question_paper", "learning_material"]:
    # Fast heuristic pass on the full text
    heuristic = _heuristic_classify(text)
    if heuristic is not None:
        return heuristic

    # LLM pass on a multi-part sample so cover pages don't dominate
    try:
        sample = _build_sample(text)
        raw = await call_llm(sample, system_content=SYSTEM_PROMPT)
        data = json.loads(raw)
        doc_type = data.get("type", "learning_material")
        if doc_type in ("question_paper", "learning_material"):
            return doc_type
    except Exception:
        pass

    return "learning_material"

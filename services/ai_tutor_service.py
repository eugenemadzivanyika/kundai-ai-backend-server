from typing import Dict, List, Optional
from services.llm_service import call_llm
from services import rag_service
import json
import re


class AiTutorService:
    """
    KUNDAI: Teaches concepts Feynman-style, personalised to the student's cognitive profile.
    Practice questions (ASAG) own formal assessment — the tutor owns explanation and dialogue.
    """

    async def generate_coaching_response(
        self,
        message: str,
        history: List[Dict],
        context: Dict,
        subject_id: Optional[str] = None,
    ) -> Dict:
        step_data   = context.get("step_data") or {}
        profile     = context.get("profile")   or {}
        canvas      = context.get("canvas")    or "No working provided yet."
        goal        = context.get("goal")      or "this topic"

        current_step   = step_data.get("title", goal)
        checkpoint     = step_data.get("exitCheckpoint") or {}
        expected_logic = checkpoint.get("expectedLogic", "")
        step_content   = step_data.get("content", "")

        if subject_id:
            rag_result = await rag_service.query_rag(
                subject_id,
                f"{current_step} {goal}",
                n_results=5,
                document_type="learning_material",
            )
        else:
            rag_result = {"chunks": [], "sources": [], "distances": [], "has_documents": False}
        rag_context_block = rag_service.format_rag_context(rag_result)

        strengths             = profile.get("strengths", [])
        deficiencies          = profile.get("deficiencies", [])
        suggested_approach    = profile.get("suggestedTutorApproach", "")

        # True when no prior student messages exist — handles both a completely
        # fresh session and one where only the intro message is in history.
        is_first_turn = not any(m.get('role') == 'user' for m in history)

        # Format profile into readable blocks the LLM can reason over
        profile_block = self._format_profile(strengths, deficiencies, suggested_approach)

        if is_first_turn:
            phase_instructions = """
PHASE: TEACH — first contact with this step. Deliver the full lesson before anything else.
Do NOT open with a question. The student came here to learn, not to be tested.

1. Hook — open with a brief, relatable Zimbabwean real-world scenario that makes the
   concept feel familiar before you name it formally. Ground the hook in what you know
   about this student (their strengths, their context, their gaps).

2. Feynman Explanation — starting from zero, explain in this order:
   - WHAT it is (plain-language definition, no assumed knowledge)
   - WHY it matters (real-world relevance, where it appears in ZIMSEC exams)
   - HOW it works (mechanics, rules, step-by-step logic)
   Use numbered steps, short paragraphs, or text-based tables/diagrams where helpful.

3. Personalise the depth using the student profile above:
   - If they have STRENGTHS in related areas → build explicit bridges.
     ("You already know how to do X — this is the same idea, just applied to Y.")
   - If they have KNOWN GAPS in prerequisite concepts → briefly shore up those gaps
     before introducing the new concept. Don't assume they have the foundation.
   - If a suggestedTutorApproach is provided → lead with that strategy.

4. Worked Example — walk through at least one complete example with clearly numbered steps.
   Use Zimbabwean names (Farai, Chipo, Tatenda), places (Gweru, Mbare, Bindura), ZiG currency.
   If there are known deficiencies, design the example to directly target those gaps.

5. Checkpoint Probe — close with exactly ONE open-ended question (not multiple choice,
   not yes/no) that asks the student to explain or apply the concept in their own words.
   Aim the question at the student's known weaknesses — that's where understanding breaks.
   Frame it as curiosity, not a test.
"""
        else:
            phase_instructions = """
PHASE: RESPOND & ADVANCE — the student has replied.
Address their message directly first, then continue building their understanding.

1. Direct Response — answer their question clearly, or acknowledge and gently correct
   their working. Never leave the student's point unaddressed.

2. Diagnose using the profile:
   - If their response shows a KNOWN MISCONCEPTION from their deficiencies → name it
     gently and explain exactly where their logic went wrong, using a fresh analogy.
   - If their response builds on a STRENGTH → affirm it and show them how it connects
     to the current concept.
   - If this is a new error not in their profile → address it directly and concisely.

3. Checkpoint Assessment (silent — the student should not feel evaluated):
   Does the student's latest message or canvas demonstrate the exit checkpoint logic?
   - YES → affirm specifically what they demonstrated (name the concept, praise the logic),
            then set checkpointPassed: true.
   - NO  → end with ONE natural follow-up question nudging them toward the checkpoint.
            Use a fresh angle — never repeat the same question from the previous turn.
            If they have struggled more than twice, give a concrete hint or mini-example
            before asking again.

4. Keep it conversational. The formal assessment happens in practice questions.
   Here the student should feel like they are thinking out loud with a knowledgeable friend.
"""

        system_prompt = f"""
You are KUNDAI, a warm and rigorous ZIMSEC tutor for Zimbabwean secondary school students.
Before you write a single word, read the student profile below. Your explanation, analogies,
examples, and questions must all be shaped by who this specific student is.

═══════════════════════════════════════
STUDENT PROFILE
═══════════════════════════════════════
{profile_block}

═══════════════════════════════════════
CURRENT LESSON
═══════════════════════════════════════
Step: "{current_step}"
Overall Goal: "{goal}"

Learning Material:
{(rag_context_block + ("\n\nSupplementary Notes:\n" + step_content if step_content else "")) if rag_context_block else (step_content if step_content else "(No documents uploaded for this subject — use your ZIMSEC curriculum knowledge.)")}

Exit Checkpoint — what this student must demonstrate before moving on:
"{expected_logic if expected_logic else 'A clear, correct explanation and application of the concept.'}"

Student's Current Working / Canvas:
{canvas}

═══════════════════════════════════════
TEACHING INSTRUCTIONS
═══════════════════════════════════════
{phase_instructions}

CHECKPOINT RULES (always apply):
- Set checkpointPassed: true ONLY when the student clearly and correctly demonstrates
  the exit checkpoint logic — not before.
- Never reveal the exit checkpoint wording to the student.
- Probe for the checkpoint at most once per turn. One question, one angle.
- Do not ask multiple questions in a single response.

TONE & STYLE:
- Warm, encouraging, locally grounded — Zimbabwean names, ZiG currency, local places.
- Treat the student as intelligent and capable.
- Reinforce ZIMSEC exam language when introducing key terms.
- Structure responses clearly — numbered steps and short paragraphs. Avoid unbroken walls of text.

FORMATTING — this message is delivered via WhatsApp. Use ONLY WhatsApp-native formatting:
- *bold* — single asterisk on each side, NO space between asterisk and word (e.g. *index notation*, NOT ** double asterisks **)
- _italic_ — single underscore on each side, no space (e.g. _optional_)
- Numbered lists (1. 2. 3.) and bullet dashes ( - ) render fine
- Do NOT use Markdown headers (##, ###), HTML tags, or any other markup
- Do NOT use LaTeX notation ($...$, \frac, \sqrt, etc.) — WhatsApp cannot render it
- For mathematical expressions use plain Unicode: write 10² not 10^2, use × not \times, use ÷ not \div, ≤ not \leq, etc.

STRICT OUTPUT — return ONLY this JSON object, no markdown wrappers, no extra keys:
{{
  "guidance": "<your full teaching or response message here>",
  "checkpointPassed": <true or false>
}}
"""

        history_text = ""
        if history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(
                f"{m.get('role', '?').upper()}: {m.get('content', '')}"
                for m in history[-6:]
            )

        user_prompt = f"{history_text}\n\nSTUDENT'S LATEST MESSAGE:\n{message}"

        raw = await call_llm(user_content=user_prompt, system_content=system_prompt)
        return self._parse_structured_response(raw)

    async def generate_plan_introduction(
        self,
        profile: Dict,
        plan: Dict,
        subject_id: Optional[str] = None,
    ) -> Dict:
        """
        Called once when a student is first assigned a development plan.
        KUNDAI makes the first move — greeting, explaining why the plan was assigned,
        what they will know by the end, and what to expect. Short, warm, anticipation-building.
        """
        strengths    = profile.get("strengths", [])
        deficiencies = profile.get("deficiencies", [])
        missions     = plan.get("missions", [])

        # Only use human-readable names — never expose attribute ID codes to the student
        gap_names = [
            d["attributeName"] for d in deficiencies
            if d.get("attributeName")
        ]
        strength_names = [
            s["attributeName"] for s in strengths
            if s.get("attributeName")
        ]

        # Build a readable mission overview
        mission_lines = []
        total_steps = 0
        for i, m in enumerate(missions, 1):
            step_count = len(m.get("steps", []))
            total_steps += step_count
            mission_lines.append(
                f"  Mission {i} — {m.get('task', 'Untitled')}: {m.get('objective', '')} ({step_count} steps)"
            )
        mission_overview = "\n".join(mission_lines) if mission_lines else "  (Plan details not available)"

        # Rough time estimate: ~15 min per step on average
        min_time = total_steps * 10
        max_time = total_steps * 15
        time_estimate = f"{min_time}–{max_time} minutes total ({10}–{15} min per step)"

        # Final learning outcome = last mission's objective
        final_outcome = missions[-1].get("objective", "") if missions else ""

        if subject_id and gap_names:
            query = " ".join(gap_names[:3])
            intro_rag = await rag_service.query_rag(
                subject_id, query, n_results=5, document_type="learning_material"
            )
        else:
            intro_rag = {"chunks": [], "sources": [], "distances": [], "has_documents": False}
        intro_rag_block = rag_service.format_rag_context(intro_rag)

        system_prompt = """
You are KUNDAI, a warm and rigorous ZIMSEC tutor for Zimbabwean secondary school students.
You are writing the very first message a student will ever receive from you — the moment they
are assigned their personal development plan.

Your job is to make them feel:
  1. Seen — you know exactly where they are and why they are here.
  2. Safe — their gaps are normal, fixable, and this plan is the path.
  3. Excited — promise them clearly and specifically what they will know by the end.

TONE: Like the opening of a great lecture — confident, warm, purposeful.
Keep it to 4–6 short paragraphs maximum.

OUTPUT FORMAT — CRITICAL:
- Return ONLY the message as plain conversational prose.
- Do NOT output JSON, XML, YAML, or any structured data format.
- Do NOT use headers (no ##, no bold labels like "Greeting:" or "Why:").
- Do NOT use bullet points or numbered lists.
- Separate paragraphs with a blank line.
- WhatsApp bold (*word* — single asterisk, no spaces) is acceptable for emphasising a key term, but use it sparingly. Do NOT use double asterisks (**word**).
- Do NOT use LaTeX notation — WhatsApp cannot render it. Use Unicode characters instead (10², ×, ÷, ≤, ≥).
- Your entire response IS the message — nothing else before or after it.
"""

        user_prompt = f"""
Write KUNDAI's opening message to a student who has just been assigned a development plan.

WHY THEY WERE ASSIGNED THIS PLAN (their specific knowledge gaps from their assessment):
{', '.join(gap_names) if gap_names else 'General concept gaps identified from their assessment.'}

WHAT THEY ALREADY KNOW WELL (their strengths — use these as confidence builders):
{', '.join(strength_names) if strength_names else 'Not yet determined.'}

THE PLAN THEY HAVE BEEN ASSIGNED:
{mission_overview}

ESTIMATED TIME COMMITMENT: {time_estimate}

WHAT THEY WILL BE ABLE TO DO BY THE END OF THIS PLAN:
{final_outcome if final_outcome else 'Master the identified concepts to ZIMSEC examination standard.'}

INSTRUCTIONS:
- Open by greeting the student warmly and personally (you know them, don't be generic).
- Explain honestly but encouragingly why they got this plan — reference their specific gaps
  by name, but frame it positively (gaps are just unfinished learning, not failure).
- Briefly mention their strengths and tell them the plan is built to use what they already know.
- Walk them through the plan in one short paragraph — name the missions, give the step count,
  and give the time estimate so they know exactly what they are committing to.
- Close with a promise: one vivid, specific sentence about what they will be able to do
  by the end. Make them want to start right now.
- End with a single gentle prompt inviting them to begin (not a question — an invitation).
- NEVER mention attribute codes, IDs, or any internal system references (e.g. MATH-F1-ALG-08).
  Use only the human-readable concept names provided above.
"""

        if intro_rag_block:
            user_prompt += f"\n\nCURRICULUM CONTEXT FOR THIS PLAN:\n{intro_rag_block}"
        else:
            user_prompt += "\n\nCURRICULUM CONTEXT FOR THIS PLAN:\n(Use your ZIMSEC knowledge for this subject.)"

        raw = await call_llm(user_content=user_prompt, system_content=system_prompt)
        return {"introduction": self._ensure_plain_text(raw)}

    def _ensure_plain_text(self, raw: str) -> str:
        """
        Safety net: if the LLM returned JSON instead of prose, extract all
        string leaf values and join them into paragraphs.
        """
        text = raw.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # Flatten all string values depth-first
                def collect(obj):
                    if isinstance(obj, str):
                        return [obj]
                    if isinstance(obj, dict):
                        return [s for v in obj.values() for s in collect(v)]
                    if isinstance(obj, list):
                        return [s for v in obj for s in collect(v)]
                    return []
                return "\n\n".join(collect(parsed))
        except (json.JSONDecodeError, ValueError):
            pass
        return text

    def _format_profile(
        self,
        strengths: List[Dict],
        deficiencies: List[Dict],
        suggested_approach: str,
    ) -> str:
        lines = []

        if strengths:
            lines.append("STRENGTHS (concepts this student handles well):")
            for s in strengths:
                name     = s.get("attributeName") or "Unknown concept"
                evidence = s.get("evidence", "")
                lines.append(f"  - {name}" + (f": {evidence}" if evidence else ""))
        else:
            lines.append("STRENGTHS: Not yet determined.")

        lines.append("")

        if deficiencies:
            lines.append("KNOWN GAPS (where this student's understanding breaks down):")
            for d in deficiencies:
                name          = d.get("attributeName") or "Unknown concept"
                misconception = d.get("misconception", "")
                failed_logic  = d.get("failedLogic", "")
                lines.append(f"  - {name}")
                if misconception:
                    lines.append(f"      Misconception : {misconception}")
                if failed_logic:
                    lines.append(f"      Failed logic  : {failed_logic}")
        else:
            lines.append("KNOWN GAPS: Not yet determined.")

        if suggested_approach:
            lines.append("")
            lines.append(f"SUGGESTED APPROACH FROM PRIOR ASSESSMENT: {suggested_approach}")

        return "\n".join(lines)

    def _parse_structured_response(self, content: str) -> Dict:
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                return {
                    "guidance":         str(parsed.get("guidance", content)),
                    "checkpointPassed": bool(parsed.get("checkpointPassed", False)),
                }
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"guidance": content, "checkpointPassed": False}

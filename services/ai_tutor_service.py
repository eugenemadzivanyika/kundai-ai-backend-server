from typing import Dict, List, Optional
from services.llm_service import call_llm
import json
import re


class AiTutorService:
    """
    Generates Socratic coaching responses grounded in the student's current
    development plan step and cognitive profile from their latest Result.
    """

    async def generate_coaching_response(
        self,
        message: str,
        history: List[Dict],
        context: Dict,
    ) -> Dict:
        """
        Returns a dict with keys:
          - guidance (str)       : the tutor's Socratic reply
          - checkpointPassed (bool): whether the student's logic satisfies
                                    the step's exitCheckpoint
        """
        step_data   = context.get("step_data") or {}
        profile     = context.get("profile")   or {}
        canvas      = context.get("canvas")    or "No working provided yet."
        goal        = context.get("goal")      or "this topic"
        mode        = context.get("mode", "socratic")

        current_step      = step_data.get("title", goal)
        expected_logic    = step_data.get("exitCheckpoint", {}).get("expectedLogic", "")
        step_content      = step_data.get("content", "")

        strengths     = profile.get("strengths", [])
        deficiencies  = profile.get("deficiencies", [])

        # Build a coaching persona prompt that is aware of the step's content
        system_prompt = f"""
You are KUNDAI, a warm and rigorous Socratic ZIMSEC tutor for Zimbabwean secondary school students.

CURRENT MISSION STEP: "{current_step}"
STEP CONTENT / LEARNING MATERIAL:
{step_content if step_content else "(No specific content provided — infer from the step title and ZIMSEC curriculum.)"}

EXIT CHECKPOINT (what the student must demonstrate to pass this step):
"{expected_logic if expected_logic else 'A clear, correct explanation and application of the concept.'}"

STUDENT COGNITIVE PROFILE:
- Strengths  : {json.dumps(strengths)  if strengths  else "Not yet determined."}
- Known gaps : {json.dumps(deficiencies) if deficiencies else "Not yet determined."}

STUDENT'S CURRENT WORKING / CANVAS:
{canvas}

COACHING MODE: {"Socratic — ask guiding questions, never reveal the full answer." if mode == "socratic" else "Hint-first — give one concrete hint, then ask a follow-up question."}

YOUR TASK:
1. Read the student's latest message and their canvas carefully.
2. Decide if their working/message demonstrates the EXIT CHECKPOINT logic.
3. Respond with an encouraging, locally relevant (Zimbabwean context) coaching message.
   - If they are on the right track, affirm specifically what is correct and push one step further.
   - If they have a gap, ask ONE guiding question that points them toward the gap without revealing the answer.
   - Reference ZIMSEC exam expectations when relevant.
4. Set checkpointPassed to true ONLY if the student's reasoning fully satisfies the exit checkpoint.

STRICT OUTPUT — return ONLY this JSON object, no markdown, no extra keys:
{{
  "guidance": "<your coaching message here>",
  "checkpointPassed": <true or false>
}}
"""

        # Build a single user turn that includes message history context
        history_text = ""
        if history:
            history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(
                f"{m.get('role','?').upper()}: {m.get('content','')}"
                for m in history[-6:]  # last 6 turns is enough context
            )

        user_prompt = f"{history_text}\n\nSTUDENT'S LATEST MESSAGE:\n{message}"

        raw = await call_llm(user_content=user_prompt, system_content=system_prompt)
        return self._parse_structured_response(raw)

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
        # Fallback: return raw text, never crash
        return {"guidance": content, "checkpointPassed": False}
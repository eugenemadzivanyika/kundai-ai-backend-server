# Kundai/kundai-ai-services-backend/services/ai_tutor_service.py
from typing import Dict, List, Optional
from services.llm_service import call_llm
import json
import re

class AiTutorService:
    def __init__(self):
        self.llm = call_llm

    async def generate_coaching_response(
        self, 
        message: str, 
        history: List[Dict], 
        context: Dict
    ) -> Dict:
        step_data = context.get("step_data", {})
        profile = context.get("profile", {})
        canvas = context.get("canvas", "No work provided yet")
        
        current_step = step_data.get("title", "this concept")
        checkpoint = step_data.get("exitCheckpoint", {})
        
        system_prompt = f"""
        You are KUNDAI, a Socratic ZIMSEC Tutor. 
        MISSION STEP: {current_step}
        EXPECTED LOGIC: "{checkpoint.get('expectedLogic')}"
        
        STUDENT PROFILE:
        - Strengths: {profile.get('strengths')}
        - Gaps: {profile.get('deficiencies')}

        TASK:
        1. Evaluate if the student's work in the Canvas: "{canvas}" fulfills the EXPECTED LOGIC.
        2. Provide a Socratic response to the student's message: "{message}".
        
        STRICT OUTPUT FORMAT (JSON):
        {{
          "guidance": "Your encouraging, localized coaching message here.",
          "checkpointPassed": true/false
        }}
        """

        # Using your existing LLM service
        raw_response = await self.llm.generate_text(
            prompt=message,
            system_instruction=system_prompt,
            history=history
        )
        
        return self._parse_structured_response(raw_response)

    def _parse_structured_response(self, content: str) -> Dict:
        try:
            # Extract JSON from potential markdown wrapping
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return {"guidance": content, "checkpointPassed": False}
        except:
            return {"guidance": content, "checkpointPassed": False}
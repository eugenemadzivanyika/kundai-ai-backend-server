from typing import Dict, List, Optional
from services.llm_service import call_llm
class AiTutorService:
    def __init__(self):
        self.llm = call_llm

    async def generate_coaching_response(
        self, 
        message: str, 
        history: List[Dict], 
        context: Dict
    ) -> str:
        goal = context.get("goal", "general learning")
        step = context.get("step", "current topic")
        canvas = context.get("canvas", "No work provided yet")
        mode = context.get("mode", "socratic")

        # The "Golden Rule" prompt for coaching
        system_prompt = f"""
        You are a highly skilled Socratic AI Tutor. Your goal is to guide the student toward understanding 
        {step} in the context of their goal: "{goal}".

        RULES:
        1. NEVER give the final answer or a full solution.
        2. If the student makes a mistake in their 'Reasoning Canvas', do not point it out directly. 
           Instead, ask a question that leads them to find the error.
        3. Use the student's current work in the Canvas: "{canvas}" to personalize your hint.
        4. If mode is 'socratic', only ask probing questions.
        5. If mode is 'hint', provide a small conceptual nudge then ask a question.
        6. Keep responses concise and encouraging.
        """

        # Construct the payload for your existing LLM service
        response = await self.llm.generate_text(
            prompt=message,
            system_instruction=system_prompt,
            history=history
        )
        
        return response

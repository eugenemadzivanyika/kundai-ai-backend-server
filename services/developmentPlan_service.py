from services.asag_service import parse_json_from_ai
from services.llm_service import call_llm
import json

async def generate_surgical_missions(payload: dict) -> dict:
    profile = payload.get("profile", {})
    strengths = profile.get("strengths", [])
    deficiencies = profile.get("deficiencies", [])
    
    system_prompt = """You are KUNDAI, a High-Performance ZIMSEC Math Coach.
    Your goal is to build a 'Surgical Remediation Path' based on a student's cognitive profile.
    
    You must generate 3 distinct Missions. 
    Mission 1 (Knowledge) MUST use the student's STRENGTHS to bridge into their GAPS.
    Mission 2 (Practice) MUST address the specific logic errors found in the profile.
    Mission 3 (Mastery) MUST be a final ZIMSEC-style evaluation.
    
    Each mission needs 'steps'. Every step must have an 'exitCheckpoint' question.
    The AI Coach will use these to verify understanding before unlocking the next step."""

    user_prompt = f"""
    STUDENT COGNITIVE PROFILE:
    Strengths: {json.dumps(strengths)}
    Deficiencies: {json.dumps(deficiencies)}

    TASK: Generate 3 Missions with granular steps and checkpoints.
    CRITICAL: Step types must ONLY be one of: "Theory", "Interactive_Exercise", or "Hinted_Practice".
    
    OUTPUT FORMAT (EXACT JSON):
    {{
      "missions": [
        {{
          "task": "Mission Title",
          "objective": "High-level goal",
          "steps": [
            {{
              "title": "Step Title",
              "content": "What the AI Coach should explain to the student",
              "type": "Theory", <-- MUST be Theory, Interactive_Exercise, or Hinted_Practice
              "exitCheckpoint": {{
                "question": "A specific question the Coach asks the student in chat",
                "expectedLogic": "The specific reasoning the Coach is looking for"
              }}
            }}
          ]
        }}
      ]
    }}
    """

    response = await call_llm(user_content=user_prompt, system_content=system_prompt) 
    return parse_json_from_ai(response)
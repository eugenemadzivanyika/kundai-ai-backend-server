from services.llm_service import call_llm
import json
import re

def parse_json_from_ai(response):
    try:
        content = response['choices'][0]['message']['content']
        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(content)
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"error": "Invalid AI JSON", "raw": content if 'content' in locals() else str(e)}

async def generate_syllabus_questions(payload: dict) -> dict:
    # 1. UNPACK THE NEW CONTEXT FIELDS
    context      = payload.get("syllabus_context", [])
    meta         = payload.get("meta_context", {})
    count        = payload.get("count", 5)
    max_score    = payload.get("maxScore", 100) # The total points boundary
    difficulty   = payload.get("difficulty", "medium")
    distribution = payload.get("distribution", {}) # e.g., {"multiple_choice": 60, ...}
    
    # 2. ENHANCED SYSTEM PERSONA
    # We inject the teacher's instructions and the specific target level here
    system_prompt = f"""You are a Senior ZIMSEC Examination Officer for Mathematics Syllabus B.
    Your task is to create a formal '{meta.get('assessment_type', 'Exercise')}' titled '{meta.get('name')}'.
    
    CRITICAL CONTEXT:
    - Target Level: {meta.get('level', 'General ZIMSEC')}
    - Teacher Constraints: {meta.get('description', 'Standard ZIMSEC assessment.')}
    - Localization: Use Zimbabwean names (Farai, Chipo), locations (Gweru, Bindura), and ZiG currency.
    - Tone: Formal, rigorous ZIMSEC examination tone (4008/4028 standard)."""

    # 3. TYPE MIX CALCULATION
    # We force the AI to respect the percentage distribution from the ConfigStep
    type_constraints = []
    for q_type, percentage in distribution.items():
        if percentage > 0:
            num = round((percentage / 100) * count)
            type_constraints.append(f"- {num} {q_type.replace('_', ' ')} questions")
    type_mix_str = "\n".join(type_constraints)

    # 4. SYLLABUS OBJECTIVES
    objectives_detail = []
    for a in context:
        detail = (
            f"### OBJECTIVE ID: {a.get('_id') or a.get('attribute_id')}\n"
            f"- TOPIC: {a.get('name')}\n"
            f"- DESCRIPTION: {a.get('description')}\n"
        )
        objectives_detail.append(detail)
    objectives_str = "\n".join(objectives_detail)

    # 5. THE SURGICAL TASK (User Prompt)
    user_prompt = f"""
    Generate exactly {count} questions based on these ZIMSEC objectives:
    {objectives_str}

    REQUIRED QUESTION MIX:
    {type_mix_str}

    TOTAL MARKS FOR PAPER: {max_score} points.
    SCORING LOGIC: Distribute points logically based on difficulty. 
    - MCQs/True-False: 1-2 points each.
    - Short Answer/Structured: 3-10 points based on complexity.
    Ensure the sum of all question points is exactly {max_score}.

    STRICT JSON SCHEMA:
    {{
      "assessment_name": "{meta.get('name', 'ZIMSEC Assessment')}",
      "questions": [
        {{
          "primaryAttributeId": "string (MUST match the OBJECTIVE ID used)",
          "questionText": "string",
          "questionType": "multiple_choice | true_false | short_answer | essay",
          "options": {{ "A": "str", "B": "str", "C": "str", "D": "str" }}, 
          "correctAnswer": "The actual TEXT value (for short_answer) or the KEY (A/B/C/D for MCQ)",
          "points": number,
          "difficulty": "{difficulty}",
          "explanation": "Step-by-step working and ZIMSEC distractor analysis."
        }}
      ]
    }}
    
    NOTE: If questionType is 'short_answer' or 'essay', return an empty object {{}} for 'options'.
    """
    
    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)
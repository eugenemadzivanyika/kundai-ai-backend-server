from services.llm_service import call_llm
import re
import json

async def grade_student_work(payload: dict) -> dict:
    """
    Uses Chain-of-Thought prompting to grade subjective answers against a ZIMSEC rubric.
    """
    content = payload.get("content", "") # The student's answer string
    rubric = payload.get("rubric", {})   # { "question": "...", "maxMarks": 5, "keywords": [...] }
    student_context = payload.get("studentContext", {})

    system_prompt = """You are a Senior ZIMSEC Moderator for Mathematics.
    Your goal is to provide 'Surgical Grading'. 
    Follow this internal 'Chain-of-Thought' process:
    1. ANALYZE: Break down the student's statement into logical claims.
    2. COMPARE: Map these claims against the marking rubric and keywords.
    3. IDENTIFY GAPS: Note specifically what is missing or what misconceptions exist.
    4. SCORE: Assign a mark based on the logic, not just keywords.
    
    Use Zimbabwean context. If a student uses a local analogy that is scientifically 
    sound (e.g., comparing a circuit to water pipes in a township), reward the logic."""

    user_prompt = f"""
    --- ASSESSMENT DATA ---
    Question: {rubric.get('question')}
    Max Marks: {rubric.get('maxMarks')}
    Required Concepts: {", ".join(rubric.get('keywords', []))}
    
    --- STUDENT SUBMISSION ---
    Student's Answer: "{content}"
    Student's Current Mastery: {student_context.get('currentMastery', 'Unknown')}

    --- TASK ---
    Return a JSON object following this EXACT structure:
    {{
      "chainOfThought": "Your step-by-step reasoning for the grade",
      "awardedScore": 0,
      "feedback": "Direct encouraging feedback for the student",
      "misconceptionsFound": ["list", "of", "errors"],
      "confidenceScore": 0.0,
      "masterySignal": true/false
    }}
    """

    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)

def parse_json_from_ai(response):
    try:
        content = response['choices'][0]['message']['content']
        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(content)
    except Exception as e:
        print(f"ASAG Extraction Error: {e}")
        return {"error": "Grading Parse Error", "raw": content}
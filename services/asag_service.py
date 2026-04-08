from services.llm_service import call_llm
import re
import json

async def grade_student_work(payload: dict) -> dict:
    """
    Uses Chain-of-Thought prompting to grade subjective answers against a ZIMSEC rubric 
    and Ground Truth answer.
    """
    content = payload.get("content", "") # The student's answer string
    rubric = payload.get("rubric", {})   # { "question": "...", "maxMarks": 5, "keywords": [...], "correctAnswer": "..." }
    student_context = payload.get("studentContext", {})

    system_prompt = """You are a Senior ZIMSEC Moderator for Mathematics.
    Your goal is to provide 'Surgical Grading' by comparing student work against the 'Expected Answer'.
    
    Follow this internal 'Chain-of-Thought' process:
    1. ANALYZE: Break down the student's statement into logical claims.
    2. COMPARE: Measure the student's claims against the 'Expected Answer' provided.
    3. EVALUATE RUBRIC: Map student logic against required concepts/keywords.
    4. IDENTIFY GAPS: Note specifically what is missing or what misconceptions exist.
    5. SCORE: Assign a mark based on logical equivalence to the expected answer, not just exact wording.
    
    Use Zimbabwean context. If a student uses a local analogy that is mathematically 
    sound (e.g., describing currency conversion using ZiG and USD rates in a local tuckshop), 
    reward the logic if it aligns with the Expected Answer."""

    user_prompt = f"""
    --- ASSESSMENT DATA ---
    Question: {rubric.get('question')}
    Max Marks: {rubric.get('maxMarks')}
    Expected Answer (Ground Truth): {rubric.get('correctAnswer', 'No ground truth provided')}
    Required Concepts: {", ".join(rubric.get('keywords', []))}
    
    --- STUDENT SUBMISSION ---
    Student's Answer: "{content}"
    Student's Current Mastery: {student_context.get('currentMastery', 'Unknown')}

    --- TASK ---
    Evaluate the student's work compared to the Expected Answer.
    Return a JSON object following this EXACT structure:
    {{
      "chainOfThought": "Your step-by-step reasoning for the grade compared to the ground truth",
      "awardedScore": 0.0,
      "feedback": "Direct encouraging feedback for the student explaining where they strayed from the correct answer",
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
        # Remove markdown code blocks if the LLM includes them
        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(content)
    except Exception as e:
        print(f"ASAG Extraction Error: {e}")
        # Return a structured error so the backend doesn't crash
        return {
            "error": "Grading Parse Error", 
            "awardedScore": 0, 
            "feedback": "Error parsing AI response.",
            "chainOfThought": "Failed to extract JSON."
        }
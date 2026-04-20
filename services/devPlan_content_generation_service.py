# from click import prompt
from services.llm_service import call_llm
import json
import re

def parse_json_from_ai(response):
    try:
        content = response['choices'][0]['message']['content']
        # Use regex to find everything between the first { and last }
        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(content)
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {"error": "Invalid AI JSON", "raw": content}

async def generate_personalized_theory(payload: dict) -> dict:
    attr = payload.get("attribute_details", {})
    mastery = payload.get("initial_mastery", 0.1)
    # NEW: Capture the specific misconceptions found during grading
    misconceptions = payload.get("misconceptions", [])
    
    system_prompt = """You are KundAI, an expert ZIMSEC tutor. 
    Your goal is to explain complex math concepts using relatable Zimbabwean scenarios 
    (like Kombi fares, tuckshop profit, or market prices). 
    
    SURGICAL INSTRUCTION: If 'misconceptions' are provided, start the module by 
    gently addressing these specific errors. Use the 'Aha!' moment technique to 
    show why the previous logic was a common pitfall.
    
    Always use Markdown for formatting. Include 2 Checkpoint questions."""

    # We build a specific string to highlight the gaps if they exist
    gap_context = ""
    if misconceptions:
        gap_context = f"\nRECENT ERRORS IDENTIFIED: {', '.join(misconceptions)}"

    user_prompt = f"""
    Topic: {attr.get('name')} ({attr.get('attribute_id')})
    Level: {attr.get('level')}
    Description: {attr.get('description')}
    Prerequisites: {attr.get('prerequisites')}
    Student Mastery: {mastery:.2f}
    {gap_context}

    TASK: Generate a 'Mini-Module' theory block. 
    1. Address the specific identified misconceptions first.
    2. Use a relatable Zim context (e.g., measuring a round vegetable garden or a soccer center circle).
    3. Explain how this builds on prerequisites.
    
    Return JSON format: {{ "title": "...", "content": "..." }}
    """
    
    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)

# --- 2. PRACTICE (Focuses on Hints and Step-by-Step) ---
async def generate_practice_set(payload: dict) -> dict:
    profile = payload.get("profile", {})
    deficiencies = profile.get("deficiencies", [])
    
    # We pull the attribute name from the first deficiency to keep the title relevant
    main_topic = deficiencies[0].get('attributeName', 'Mathematics') if deficiencies else "Mathematics"
    
    system_prompt = """You are KUNDAI's Practice Engine.
    Generate practice problems where the 'hints' are specifically designed to 
    stop the student from making the logical errors identified in their profile.
    
    The 'steps' should provide a ZIMSEC-aligned walkthrough."""

    user_prompt = f"""
    TARGET DEFICIENCIES: {json.dumps(deficiencies)}
    
    TASK: Generate 5 localized practice problems.
    For each problem, include:
    1. A 'hint' that addresses the specific misconception.
    2. A 'tutor_explanation' for the hint (what the coach says if they ask 'Why that hint?')
    3. Step-by-step solution logic.

    STRICT JSON SCHEMA:
    {{
      "title": "Surgical Practice: {main_topic}",
      "problems": [
        {{
          "question": "The problem text",
          "hint": "The student-facing hint",
          "tutor_explanation": "Internal logic for the AI coach about this hint",
          "steps": ["Step 1...", "Step 2..."],
          "final_answer": "The final result"
        }}
      ]
    }}
    """
    
    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)

async def generate_mastery_quiz(payload: dict) -> dict:
    attr = payload.get("attribute_details", {})
    misconceptions = payload.get("misconceptions", [])
    
    system_prompt = """You are KUNDAI, a strict ZIMSEC Examiner. 
    Create rigorous multiple-choice questions that mimic the Paper 1 style. 
    
    SURGICAL INSTRUCTION: If 'misconceptions' are provided, at least one 'distractor' 
    (wrong option) per question MUST be the result of that specific misconception. 
    The 'explanation' should explain why that specific distractor is a trap."""

    gap_context = f"\nPAST MISCONCEPTIONS: {', '.join(misconceptions)}" if misconceptions else ""

    user_prompt = f"""
    Topic: {attr.get('name')} | Level: {attr.get('level')}
    {gap_context}
    
    Task: Create a 5-question Multiple Choice Quiz. 
    Include traps based on the listed misconceptions.
    
    STRICT JSON SCHEMA:
    {{
      "title": "Mastery Check: {attr.get('name')}",
      "questions": [
        {{
          "question": "...",
          "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
          "correct_answer": "...",
          "explanation": "..."
        }}
      ]
    }}
    """
    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)
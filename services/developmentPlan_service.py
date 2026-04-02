from services.llm_service import call_llm
import json

async def generate_surgical_missions(payload: dict) -> dict:
    attr = payload.get("attribute_details", {})
    mastery = payload.get("initial_mastery", 0.1)
    
    # We define the KUNDAI persona to ensure the tone is supportive yet rigorous
    system_prompt = (
        "You are KUNDAI, an expert ZIMSEC Mathematics tutor. Your goal is to create a "
        "remediation path for a student struggling with a specific skill."
    )
    
    user_prompt = f"""
    You are KUNDAI, an expert ZIMSEC Mathematics tutor.
    Topic: {attr.get('name')} ({attr.get('attribute_id')})
    Level: {attr.get('level')}
    Description: {attr.get('description')}
    Prerequisites: {attr.get('prerequisites')}

    The student has a mastery level of {mastery:.2f}. Generate 3 distinct 'missions':
    1. Review (Knowledge) - Focus on: {attr.get('name')}
    2. Practice (Application) - Based on: {attr.get('description')}
    3. Assessment (Evaluation) - ZIMSEC Standard.

    Format the output as a JSON object with a 'missions' key containing an array of objects.
    Each object must have:
    - "task": A clear, actionable title for the student.
    - "objective": The learning goal (e.g., 'Mastering HCF concepts').
    - "status": Always set to 'Pending'.
    """

    # Call the Mistral Brain
    response = await call_llm(user_content=user_prompt, system_content=system_prompt) 
    
    try:
        # Mistral sometimes wraps JSON in markdown blocks; we strip those if present
        content = response['choices'][0]['message']['content']
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        
        return json.loads(content)
    except Exception as e:
        print(f"Parsing Error: {e}")
        return {"missions": [], "error": "AI response was not valid JSON"}
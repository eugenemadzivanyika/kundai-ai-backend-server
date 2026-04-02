from services.ocr_service import perform_ocr
from services.llm_service import call_llm

async def generate_assessment_from_resource(file_id: str):
    file_path = f"uploads/{file_id}"
    
    # 1. Read the document
    raw_text = await perform_ocr(file_path)
    
    # 2. Build the Prompt (The Instructions)
    prompt = f"""
    Based on the following text, generate 3 Multiple Choice Questions.
    Return a JSON object with a key 'questions' containing an array of objects.
    Each object must have: id, stem, options (array of {{id, text}}), and answer.
    
    TEXT:
    {raw_text[:2000]} 
    """
    
    # 3. Ask the Brain
    ai_response = await call_llm(prompt)
    # Extract the string content from Mistral's response
    content_string = ai_response['choices'][0]['message']['content']
    # Turn that string into a real Python Dictionary (JSON)
    quiz_data = json.loads(content_string)
    return quiz_data

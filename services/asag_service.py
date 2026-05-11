from services.llm_service import call_llm
import re
import json

async def grade_student_work(payload: dict) -> dict:
    """
    Grades each question individually and returns a per-question chainOfThought array.
    Accepts a `questions` array (preferred) or falls back to legacy flat `content`.
    """
    questions = payload.get("questions", [])
    raw_rubric = payload.get("rubric", {})
    # rubric may arrive as a list (per-question objects) or a dict (legacy metadata)
    rubric_list = raw_rubric if isinstance(raw_rubric, list) else []
    rubric = raw_rubric if isinstance(raw_rubric, dict) else {}
    student_context = payload.get("studentContext", {})

    system_prompt = """You are a Senior ZIMSEC Moderator for Mathematics.
    Your goal is to provide 'Surgical Grading' by comparing student work against the 'Expected Answer'.

    Follow this internal 'Chain-of-Thought' process for EACH question:
    1. ANALYZE: Break down the student's statement into logical claims.
    2. COMPARE: Measure the student's claims against the 'Expected Answer' provided.
    3. EVALUATE RUBRIC: Map student logic against required concepts/keywords.
    4. IDENTIFY GAPS: Note specifically what is missing or what misconceptions exist.
    5. SCORE: Assign a mark based on logical equivalence to the expected answer, not just exact wording.

    Use Zimbabwean context. If a student uses a local analogy that is mathematically
    sound (e.g., describing currency conversion using ZiG and USD rates in a local tuckshop),
    reward the logic if it aligns with the Expected Answer."""

    if questions:
        questions_block = "\n\n".join([
            f"--- Question {q.get('number', i+1)} (Max: {q.get('maxPoints', 1)} marks) ---\n"
            f"Question: {q.get('text', '')}\n"
            f"Expected Answer: {q.get('correctAnswer', 'Not provided')}\n"
            f"Student Answer: \"{q.get('studentAnswer', '')}\""
            for i, q in enumerate(questions)
        ])
        total_marks = sum(q.get('maxPoints', 1) for q in questions)
        question_count = len(questions)
    else:
        # Legacy fallback — build from flat content + rubric list if available
        questions_block = payload.get("content", "")
        if rubric_list:
            total_marks = sum(q.get('maxMarks', q.get('maxPoints', 1)) for q in rubric_list)
            question_count = len(rubric_list)
        else:
            total_marks = rubric.get('maxMarks', 10)
            question_count = 1

    # Build per-question score examples to anchor the LLM on marks, not percentages
    score_examples = ""
    if questions:
        score_examples = "\n    SCORING EXAMPLES (marks, NOT percentages):\n"
        for q in questions[:3]:
            mp = q.get('maxPoints', 1)
            score_examples += (
                f"    - A question worth {mp} marks: full credit = {mp}, half credit = {mp/2:.1f}, "
                f"no credit = 0  (NEVER return {100} or {50} — those are percentages, not marks)\n"
            )

    user_prompt = f"""
    --- ASSESSMENT: {rubric.get('paperName', rubric.get('question', 'Assessment'))} ---
    Total Available Marks: {total_marks}
    Student's Current Mastery: {student_context.get('currentMastery', 'Unknown')}

    --- QUESTIONS AND STUDENT ANSWERS ---
    {questions_block}

    --- CRITICAL SCORING RULE ---
    "score" is ALWAYS in RAW MARKS, never a percentage.
    A question worth 6 marks: award between 0 and 6 (e.g. 4.0, not 67).
    A question worth 8 marks: award between 0 and 8 (e.g. 8.0 for full marks, not 100).
    score MUST be <= maxScore. Do NOT return percentages.
    {score_examples}
    --- TASK ---
    Grade EACH question individually.
    Return a JSON object with this EXACT structure:
    {{
      "chainOfThought": [
        {{
          "question": 1,
          "score": 4.0,
          "maxScore": 6,
          "analysis": "One sentence: what the student's answer claims",
          "comparison": "One sentence: how it maps to the expected answer",
          "evaluation": "One sentence: rubric judgement and reasoning",
          "gaps": "Specific gap or 'None' if fully correct"
        }}
      ],
      "awardedScore": <sum of all question scores in marks>,
      "feedback": "Direct encouraging feedback explaining where they strayed from correct answers",
      "misconceptionsFound": ["list", "of", "key errors"],
      "confidenceScore": <0.0 to 1.0>,
      "masterySignal": true/false
    }}
    The chainOfThought array must have exactly {question_count} entries, one per question.
    Every "score" value must be a number between 0 and its corresponding "maxScore".
    """

    response = await call_llm(user_prompt, system_content=system_prompt)
    result = parse_json_from_ai(response)

    # Normalise any scores the LLM returned as percentages (score > maxScore is impossible in marks)
    if isinstance(result.get("chainOfThought"), list):
        q_map = {q.get('number', i + 1): q.get('maxPoints', 1) for i, q in enumerate(questions)}
        corrected_total = 0.0
        for step in result["chainOfThought"]:
            max_pts = q_map.get(step.get("question"), step.get("maxScore", 1))
            raw = step.get("score", 0)
            if raw > max_pts:
                # LLM returned a percentage — convert to marks
                step["score"] = round((raw / 100.0) * max_pts, 1)
            step["score"] = min(step["score"], max_pts)   # hard cap
            step["maxScore"] = max_pts
            corrected_total += step["score"]
        # Recalculate awardedScore from the corrected per-question scores
        result["awardedScore"] = round(corrected_total, 1)

    # Guarantee chainOfThought is always a list
    if isinstance(result.get("chainOfThought"), str):
        cot_text = result["chainOfThought"]
        if questions:
            result["chainOfThought"] = [
                {
                    "question": i + 1,
                    "score": 0,
                    "maxScore": q.get("maxPoints", 1),
                    "analysis": cot_text if i == 0 else "See question 1 analysis.",
                    "comparison": "",
                    "evaluation": "",
                    "gaps": "Could not parse per-question breakdown."
                }
                for i, q in enumerate(questions)
            ]
        else:
            result["chainOfThought"] = [{"question": 1, "score": 0, "maxScore": total_marks,
                                          "analysis": cot_text, "comparison": "", "evaluation": "", "gaps": ""}]

    return result

async def build_cognitive_profile(payload: dict, grading_output: dict) -> dict:
    """
    Analyzes student logic using full text context and attribute names.
    """
    questions = payload.get("questions", [])
    raw_rubric = payload.get("rubric", {})
    rubric = raw_rubric if isinstance(raw_rubric, dict) else {}
    attribute_context = rubric.get("attributeContext", "No specific attribute names provided")

    if questions:
        formatted_work = "\n\n".join([
            f"Q{q.get('number', i+1)} [{q.get('attributeId', '')}]: {q.get('studentAnswer', '')}"
            for i, q in enumerate(questions)
        ])
    else:
        formatted_work = payload.get("content", "")

    system_prompt = """You are an Educational Psychologist and ZIMSEC Curriculum Expert.
    Your task is to build a 'Cognitive Profile' that maps student logic to Syllabus Attributes.

    CRITICAL INSTRUCTION: Use the human-readable Attribute Names provided in the context.
    Do not just return IDs; explain the 'Cognitive Gap' in relation to the topic name."""

    user_prompt = f"""
    --- SYLLABUS CONTEXT ---
    Target Attributes for this paper: {attribute_context}

    --- STUDENT WORK ---
    {formatted_work}

    --- INITIAL GRADING FEEDBACK ---
    Feedback: {grading_output.get('feedback')}
    Errors found: {grading_output.get('misconceptionsFound')}

    --- TASK ---
    Return a JSON object exactly like this:
    {{
      "strengths": [
        {{ "attributeId": "The ID", "attributeName": "The Name", "evidence": "Why they are good at this" }}
      ],
      "deficiencies": [
        {{
          "attributeId": "The ID",
          "attributeName": "The Name",
          "misconception": "Short label of the error",
          "failedLogic": "Deep dive into the student's wrong logic path"
        }}
      ],
      "suggestedTutorApproach": "One sentence strategy for the AI Coach"
    }}
    """

    response = await call_llm(user_prompt, system_content=system_prompt)
    return parse_json_from_ai(response)

def parse_json_from_ai(response):
    try:
        content = response
        match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(content)
    except Exception as e:
        print(f"ASAG Extraction Error: {e}")
        return {
            "error": "Grading Parse Error",
            "awardedScore": 0,
            "feedback": "Error parsing AI response.",
            "chainOfThought": [],
            "misconceptionsFound": [],
            "confidenceScore": 0.5
        }

def update_mastery(payload: dict) -> dict:
    """
    Calculates the new probability of mastery using standard BKT.
    Formula: P(Ln|Result) = P(Ln-1|Result) + (1 - P(Ln-1|Result)) * P(T)
    """
    # 1. Extract inputs sent from Node.js
    p_ln_minus_1 = payload.get("prev_mastery", 0.1)  # Previous Mastery (L0)
    correct = payload.get("correct", False)          # Result: True/False
    
    # 2. Get BKT Parameters (Defaults based on ZIMSEC Syllabus B)
    p_g = payload.get("p_guess", 0.2)   # Prob of guessing right
    p_s = payload.get("p_slip", 0.1)    # Prob of slipping up
    p_t = payload.get("p_transit", 0.1) # Prob of learning (Transition)

    # 3. Step A: Calculate the 'Prior' (Mastery given the current response)
    if correct:
        # Probability student knew it, given they got it right
        p_known_given_obs = (p_ln_minus_1 * (1 - p_s)) / ((p_ln_minus_1 * (1 - p_s)) + ((1 - p_ln_minus_1) * p_g))
    else:
        # Probability student knew it, given they got it wrong
        p_known_given_obs = (p_ln_minus_1 * p_s) / ((p_ln_minus_1 * p_s) + ((1 - p_ln_minus_1) * (1 - p_g)))

    # 4. Step B: Account for Learning (Transition)
    # The new mastery is the probability they knew it + the probability they just learned it
    new_mastery = p_known_given_obs + (1 - p_known_given_obs) * p_t

    final_score = min(new_mastery, 0.9999)
    return {
        "new_mastery": round(final_score, 4),
        "attribute_id": payload.get("attribute_id"),
        "growth": round(final_score - p_ln_minus_1, 4),
        "timestamp": payload.get("timestamp")
    }
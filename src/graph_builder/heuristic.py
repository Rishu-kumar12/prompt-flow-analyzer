ACTION_KEYWORDS = [
    "calculate", "sort", "filter", "merge", "combine",
    "analyze", "report", "generate", "extract", "process"
]

def apply_heuristic(src_text, tgt_text, base_sim):
    """
    Compute heuristic influence score based on action keywords.
    """
    score = base_sim
    if any(k in src_text.lower() for k in ACTION_KEYWORDS):
        score += 0.05
    return min(score, 1.0)

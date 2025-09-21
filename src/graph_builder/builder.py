import numpy as np
from .similarity import cosine_sim_matrix
from .heuristic import apply_heuristic
from .llm_influence import llama_influence_score

def top_k_next_edges(sentences, emb_matrix, top_k=3, candidate_window=None,
                     softmax_temp=1.0, intra_paragraph_boost=0.1,
                     use_llm=False, llm_weight=0.3):
    """
    Build top-k probable next-instruction edges with paragraph preference
    and optional LLM high-level influence.
    """
    n = len(sentences)
    edges = []

    # Cosine similarity
    sim = cosine_sim_matrix(emb_matrix)

    # Paragraph boost
    for i in range(n):
        for j in range(i+1, n):
            if sentences[i]["paragraph_id"] == sentences[j]["paragraph_id"]:
                sim[i, j] += intra_paragraph_boost

    for i in range(n):
        if candidate_window is None:
            candidates = list(range(i+1, n))
        else:
            end = min(n, i+1+candidate_window)
            candidates = list(range(i+1, end))

        if not candidates:
            continue

        # Heuristic influence
        sims = np.array([apply_heuristic(sentences[i]["text"], sentences[j]["text"], sim[i, j])
                         for j in candidates])

        # LLM influence
        if use_llm:
            for idx, j in enumerate(candidates):
                llm_score = llama_influence_score(sentences[i]["text"], sentences[j]["text"])
                sims[idx] = (1 - llm_weight) * sims[idx] + llm_weight * llm_score

        # Top-k selection
        kk = min(top_k, len(candidates))
        top_idx = sims.argsort()[::-1][:kk]
        top_candidates = [candidates[t] for t in top_idx]
        top_sims = sims[top_idx].astype(float)

        # Softmax probabilities
        logits = (top_sims + 1.0) / softmax_temp
        exps = np.exp(logits - np.max(logits))
        probs = exps / exps.sum()

        # Build edges
        for tgt, p, sim_val in zip(top_candidates, probs, top_sims):
            reason = "high similarity"
            if sentences[i]["paragraph_id"] == sentences[tgt]["paragraph_id"]:
                reason += ", same paragraph"
            if use_llm:
                reason += ", LLM influence"
            edges.append({
                "from": i,
                "to": tgt,
                "score": float(p),
                "sim": float(sim_val),
                "reason": reason
            })

    return edges

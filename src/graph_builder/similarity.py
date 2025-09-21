import numpy as np

def cosine_sim_matrix(emb_matrix):
    """
    Compute cosine similarity matrix from embedding vectors.
    """
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    emb_norm = emb_matrix / norms
    return emb_norm.dot(emb_norm.T)

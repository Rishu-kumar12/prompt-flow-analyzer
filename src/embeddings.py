from sentence_transformers import SentenceTransformer
import numpy as np

def load_model(model_name="all-MiniLM-L6-v2", device="cpu"):
    """
    Load a small sentence-transformer model (local or HF).
    """
    model = SentenceTransformer(model_name, device=device)
    return model

def embed_sentences(model, sentences):
    """
    Compute embeddings for a list of sentences (structured dicts with paragraph info).
    Returns numpy array of shape (num_sentences, embedding_dim)
    """
    texts = [s["text"] for s in sentences]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return normalize_embeddings(embeddings)

def normalize_embeddings(emb):
    """
    L2-normalize embeddings for cosine similarity.
    """
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return emb / norms

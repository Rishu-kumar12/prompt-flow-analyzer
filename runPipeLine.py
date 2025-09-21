from src.splitter import split_sentences_with_paragraphs
from src.embeddings import load_model, embed_sentences
from src.graph_builder.builder import top_k_next_edges
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# ---- Step 1: Load text ----
with open("data/long_prompt.txt", "r", encoding="utf-8") as f:
    long_prompt = f.read()

# ---- Step 2: Split sentences with paragraph info ----
# Returns list of dicts: {"paragraph_id": int, "sent_id": int, "text": str}
sentences = split_sentences_with_paragraphs(long_prompt)
print("Number of sentences:", len(sentences))

# ---- Step 3: Load small embedding model ----
model = load_model("all-MiniLM-L6-v2")

# ---- Step 4: Compute embeddings ----
embs = embed_sentences(model, sentences)

# ---- Step 5: Build top-3 edges with optional LLM influence ----
edges = top_k_next_edges(
    sentences,
    embs,
    top_k=3,
    candidate_window=200,
    softmax_temp=0.8,
    intra_paragraph_boost=0.1,
    use_llm=True,            # Toggle LLM influence here
    llm_weight=0.3           # How much weight LLM has in scoring
)

# ---- Step 6: Save JSON ----
output = {
    "sentences": sentences,
    "edges": edges
}

output_file = "outputs/result.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved {output_file} with {len(sentences)} nodes and {len(edges)} edges.")

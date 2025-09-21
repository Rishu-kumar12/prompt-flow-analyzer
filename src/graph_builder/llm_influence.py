import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

def llama_influence_score(src_text, tgt_text, api_key=OPENROUTER_API_KEY, model="meta-llama/llama-3.3-70b-instruct:free"):
    """
    Call LLaMA 3.3 via OpenRouter to get influence score between two instructions.
    Returns 0-1.
    """
    prompt = f"""
    You are an AI assistant. Given the following instructions:

    Source instruction: "{src_text}"
    Target instruction: "{tgt_text}"

    Rate how much the source instruction influences the target instruction on a scale from 0 (no influence) to 1 (strong influence). 
    Respond only with a numeric value between 0 and 1.
    """
    try:
        response = requests.post(
            url=OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            })
        )
        result = response.json()
        score = float(result["choices"][0]["message"]["content"].strip())
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        print("LLM call failed:", e)
        return 0.0

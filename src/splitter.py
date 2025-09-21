import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

def split_paragraphs(text):
    """
    Split text into paragraphs by detecting double newlines.
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def split_sentences_with_paragraphs(text):
    """
    Split a long text into structured sentences with paragraph info.
    Returns list of dicts: {"paragraph_id": int, "sent_id": int, "text": str}
    """
    paragraphs = split_paragraphs(text)
    results = []

    for p_id, paragraph in enumerate(paragraphs):
        sents = [s.strip() for s in sent_tokenize(paragraph) if s.strip()]
        for s_id, sent in enumerate(sents):
            results.append({
                "paragraph_id": p_id,
                "sent_id": s_id,
                "text": sent
            })

    return results


if __name__ == "__main__":
    test_text = """Write a summary. Include bullet points. Keep it professional.

    Explain the limitations. Suggest improvements."""
    
    structured_sents = split_sentences_with_paragraphs(test_text)
    for item in structured_sents:
        print(item)

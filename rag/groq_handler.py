
import os
from openai import OpenAI
import requests

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def embed_text(text: str) -> list:
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace embedding error: {response.status_code} - {response.text}")
    return response.json()[0]
# Set up the Groq client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Define model names
LLM_MODEL_ID = "mixtral-8x7b-32768"    


def generate_response(prompt: str, context: str) -> str:
    """Generate a response from Groq LLM using the given context and prompt."""
    full_prompt = f"""Answer the question based on the context below:

Context:
{context}

Question: {prompt}
"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Groq LLM error: {e}")

def expand_query(original_query: str, n: int = 3) -> list:
    """Generate n reworded queries for fusion RAG."""
    prompt = f"Generate {n} different rewordings of this question for semantic search: {original_query}"
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content
        return [q.strip("-• ") for q in raw.strip().split("\n") if q.strip()]
    except Exception as e:
        raise RuntimeError(f"Groq expand_query error: {e}")

def extract_metadata(text: str) -> dict:
    """Extract topic, entities, and section_title from text for self-query RAG."""
    prompt = f"""
Extract structured metadata from the following passage.
Return a Python dictionary with keys: topic (str), entities (list), and section_title (str).

Passage:
{text}
"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[{"role": "user", "content": prompt}]
        )
        return eval(response.choices[0].message.content.strip())
    except Exception as e:
        return {"topic": None, "entities": [], "section_title": None}
def generate_query_variants(original_query: str, n: int = 3) -> list:
    """Alias for expand_query."""
    return expand_query(original_query, n)
import os
from openai import OpenAI

# Initialize OpenRouter client

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Model choices
EMBED_MODEL_ID = "openai/text-embedding-3-small"
LLM_MODEL_ID = "mistralai/mistral-7b-instruct"  # or use "anthropic/claude-3-haiku"

def embed_text(text: str) -> list:
    """Generate an embedding vector using OpenRouter."""
    response = client.embeddings.create(
        model=EMBED_MODEL_ID,
        input=text
    )
    return response.data[0].embedding

def generate_response(prompt: str, context: str) -> str:
    """Generate an LLM response using OpenRouter."""
    full_prompt = f"""Answer the question based on the context below:

Context:
{context}

Question: {prompt}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def expand_query(original_query: str, n: int = 3) -> list:
    """Generate n rewordings for fusion RAG."""
    prompt = f"Generate {n} different rewordings of this question for semantic search:\n{original_query}"
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    raw = response.choices[0].message.content
    return [q.strip("-• \n") for q in raw.strip().split("\n") if q.strip()]

def extract_metadata(text: str) -> dict:
    """Extract metadata for self-query RAG."""
    prompt = f"""
Extract structured metadata from the following passage.
Return a Python dictionary with keys: topic (str), entities (list), and section_title (str).

Passage:
{text}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        return eval(response.choices[0].message.content.strip())
    except Exception:
        return {"topic": None, "entities": [], "section_title": None}

def generate_query_variants(original_query: str, n: int = 3) -> list:
    """Alias for expand_query."""
    return expand_query(original_query, n)

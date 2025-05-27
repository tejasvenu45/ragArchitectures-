import google.generativeai as genai
import os

# Configure the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Define model IDs
EMBED_MODEL_ID = "models/embedding-001"
LLM_MODEL_ID = "models/gemini-2.0-flash"

# Create only the LLM model instance (embedding doesn't require instantiation)
llm_model = genai.GenerativeModel(LLM_MODEL_ID)

def embed_text(text: str) -> list:
    """Generate an embedding vector from the given text."""
    response = genai.embed_content(
        model=EMBED_MODEL_ID,
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

def generate_response(prompt: str, context: str) -> str:
    """Use Gemini to answer a prompt with given context."""
    full_prompt = f"Answer the question based on context below:\n\nContext:\n{context}\n\nQuestion: {prompt}"
    response = llm_model.generate_content(full_prompt)
    return response.text.strip()

def expand_query(original_query: str, n: int = 3) -> list:
    """Generate n sub-queries for fusion RAG."""
    response = llm_model.generate_content(
        f"Generate {n} different rewordings of this question for semantic search: {original_query}"
    )
    return [q.strip("-• \n") for q in response.text.strip().split("\n") if q.strip()]

def extract_metadata(text: str) -> dict:
    """Extract structured metadata for self-query RAG using Gemini."""
    prompt = f"""
    Extract structured metadata from the following passage.
    Return a JSON with fields: topic, entities, and section_title.

    Passage:
    {text}
    """
    response = llm_model.generate_content(prompt)
    try:
        return eval(response.text.strip())
    except:
        return {"topic": None, "entities": [], "section_title": None}
def generate_query_variants(original_query: str, n: int = 3) -> list:
    """Generate n sub-queries for fusion RAG."""
    response = llm_model.generate_content(
        f"Generate {n} different rewordings of this question for semantic search: {original_query}"
    )
    return [q.strip("-• \n") for q in response.text.strip().split("\n") if q.strip()]


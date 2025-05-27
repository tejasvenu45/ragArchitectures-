from fastapi import APIRouter, UploadFile, File, HTTPException
from rag.utils import extract_text_by_page, generate_uuid, timer
from rag.groq_handler import embed_text, generate_response
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
import os
import tempfile
from typing import List
import uuid
import time

from pydantic import BaseModel

class QueryRequest(BaseModel):
    collection: str
    question: str
    top_k: int = 5

router = APIRouter()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_PREFIX = "simplerag_"

def create_collection(name: str):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

def add_chunks_to_qdrant(collection_name: str, chunks: List[dict]):
    points = []
    for chunk in chunks:
        vector = embed_text(chunk["text"])
        points.append(
            PointStruct(
                id=uuid.uuid4().int >> 64,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"]
                }
            )
        )
    client.upsert(collection_name=collection_name, points=points)

@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.file.read())
            file_path = tmp.name

        pages = extract_text_by_page(file_path)
        chunks = [{"text": text, "page": page} for page, text in pages]
        collection_name = COLLECTION_PREFIX + file.filename.replace(".pdf", "").replace(" ", "_")

        create_collection(collection_name)
        add_chunks_to_qdrant(collection_name, chunks)

        return {"status": "uploaded", "pages": len(pages), "collection": collection_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/query")
def query_pdf(request: QueryRequest):
    collection_name = COLLECTION_PREFIX + request.collection
    try:
        q_vector = embed_text(request.question)
        start = time.time()
        search_result = client.search(
            collection_name=collection_name,
            query_vector=q_vector,
            limit=request.top_k,
        )
        end = time.time()

        retrieved_chunks = [point.payload["text"] for point in search_result]
        context = "\n\n".join(retrieved_chunks)
        answer = generate_response(request.question, context)

        time_taken = round(end - start, 4)

        precision = round(request.top_k / len(search_result), 2) if search_result else 0
        recall = "Estimation N/A (no labels)"
        relevance = eval_answer_relevance(request.question, answer)
        context_score = eval_context_relevance(request.question, retrieved_chunks)

        return {
            "question": request.question,
            "answer": answer,
            "response_time": time_taken,
            "retrieved_chunks": retrieved_chunks,
            "metrics": {
                "precision@k": precision,
                "recall@k": recall,
                "answer_relevance_score": relevance,
                "context_relevance_score": context_score
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Evaluation helpers (basic heuristics using LLMs) ---

def eval_answer_relevance(question: str, answer: str) -> float:
    """Use Gemini to rate the answer's relevance to the question (0–1)."""
    prompt = f"""
    Rate the relevance of the following answer to the question on a scale of 0 to 1:

    Question: {question}

    Answer: {answer}

    Only return the number.
    """
    try:
        from rag.gemini_handler import llm_model
        res = llm_model.generate_content(prompt)
        score = float(res.text.strip())
        return round(score, 2)
    except:
        return 0.0

def eval_context_relevance(question: str, chunks: List[str]) -> float:
    """Heuristically measure average similarity to question."""
    from rag.gemini_handler import embedding_model
    try:
        q_vec = embed_text(question)
        total_sim = 0
        for chunk in chunks:
            chunk_vec = embed_text(chunk)
            sim = cosine_similarity(q_vec, chunk_vec)
            total_sim += sim
        return round(total_sim / len(chunks), 3)
    except:
        return 0.0

def cosine_similarity(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

from fastapi import APIRouter, UploadFile, File, HTTPException
from rag.utils import extract_text_by_page
from rag.groq_handler import embed_text, generate_response, generate_query_variants
from rag.evaluation_metrics import answer_relevance_score, coverage_score
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import os
import tempfile
from typing import List
import uuid
import time
from collections import defaultdict
from pydantic import BaseModel

router = APIRouter()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_PREFIX = "fusionrag_"

# Inline request model
class FusionQueryRequest(BaseModel):
    collection: str
    question: str
    num_queries: int = 3
    top_k: int = 3

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
                payload={"text": chunk["text"], "page": chunk["page"]}
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
def query_pdf(request: FusionQueryRequest):
    collection_name = COLLECTION_PREFIX + request.collection
    try:
        queries = generate_query_variants(request.question)
        queries.append(request.question)  # include original

        retrieved = []
        start = time.time()
        for q in queries:
            vector = embed_text(q)
            result = client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=request.top_k
            )
            retrieved.extend(result)
        end = time.time()

        # Deduplicate by payload text
        seen = set()
        fused_chunks = []
        for item in retrieved:
            txt = item.payload["text"]
            if txt not in seen:
                fused_chunks.append(txt)
                seen.add(txt)

        context = "\n\n".join(fused_chunks)
        answer = generate_response(request.question, context)

        # Evaluation metrics
        relevance = answer_relevance_score(answer, fused_chunks)
        coverage = coverage_score(answer, fused_chunks)

        return {
            "question": request.question,
            "query_variants": queries,
            "answer": answer,
            "response_time": round(end - start, 4),
            "retrieved_chunks": fused_chunks,
            "metrics": {
                "deduplicated_chunks": len(fused_chunks),
                "raw_retrieved": len(retrieved),
                "fusion_gain": round(len(fused_chunks) / len(retrieved), 2),
                "answer_relevance_score": relevance,
                "coverage_score": coverage
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

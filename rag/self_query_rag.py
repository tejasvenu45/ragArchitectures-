from fastapi import APIRouter, UploadFile, File, HTTPException
from rag.utils import extract_text_by_page, generate_uuid
from rag.groq_handler import embed_text, generate_response, extract_metadata
from qdrant_client import QdrantClient
from rag.evaluation_metrics import answer_relevance_score, coverage_score

from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
import os
import tempfile
from typing import List
import uuid

from pydantic import BaseModel
from typing import Optional
import time

class QueryRequest(BaseModel):
    collection: str
    question: str
    topic: Optional[str] = None
    section: Optional[str] = None
    top_k: Optional[int] = 5


router = APIRouter()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_PREFIX = "selfrag_"


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
                    "page": chunk["page"],
                    "topic": chunk["topic"],
                    "entities": chunk["entities"],
                    "section_title": chunk["section_title"]
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
        chunks = []
        for page, text in pages:
            metadata = extract_metadata(text)
            chunks.append({
                "text": text,
                "page": page,
                **metadata
            })

        collection_name = COLLECTION_PREFIX + file.filename.replace(".pdf", "").replace(" ", "_")

        create_collection(collection_name)
        add_chunks_to_qdrant(collection_name, chunks)

        return {"status": "uploaded", "pages": len(pages), "collection": collection_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
def query_pdf(req: QueryRequest):
    collection_name = COLLECTION_PREFIX + req.collection
    try:
        q_vector = embed_text(req.question)
        start = time.time()

        filters = []
        if req.topic:
            filters.append(FieldCondition(key="topic", match=MatchValue(value=req.topic)))
        if req.section:
            filters.append(FieldCondition(key="section_title", match=MatchValue(value=req.section)))

        query_filter = Filter(must=filters) if filters else None

        search_result = client.search(
            collection_name=collection_name,
            query_vector=q_vector,
            limit=req.top_k,
            query_filter=query_filter
        )
        end = time.time()

        retrieved_chunks = [point.payload["text"] for point in search_result]
        context = "\n\n".join(retrieved_chunks)
        answer = generate_response(req.question, context)
        relevance = answer_relevance_score(answer, retrieved_chunks)
        coverage = coverage_score(answer, retrieved_chunks)
        time_taken = round(end - start, 4)

        return {
            "question": req.question,
            "answer": answer,
            "response_time": time_taken,
            "retrieved_chunks": retrieved_chunks,
            "relevance": relevance,
            "coverage": coverage,
            "metadata_filter": {"topic": req.topic, "section_title": req.section}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
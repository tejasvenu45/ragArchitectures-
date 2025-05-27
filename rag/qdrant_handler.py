from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

router = APIRouter()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

@router.get("/collections")
def list_collections():
    """List all Qdrant collections."""
    collections = client.get_collections()
    return {"collections": [c.name for c in collections.collections]}

@router.delete("/collections/{name}")
def delete_collection(name: str):
    """Delete a single Qdrant collection."""
    try:
        client.delete_collection(collection_name=name)
        return {"status": "deleted", "collection": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear-all")
def clear_all_collections():
    """Danger: Delete ALL collections!"""
    collections = client.get_collections().collections
    for c in collections:
        client.delete_collection(c.name)
    return {"status": "all collections deleted"}

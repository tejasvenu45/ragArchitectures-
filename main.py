from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

# Init app
app = FastAPI(
    title="Multi-RAG PDF QA System",
    description="Supports Simple, Self-query, and Fusion RAG using Gemini + Qdrant",
    version="1.0.0",
)

# Enable CORS if using with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes (modular, will add later)
from rag.simple_rag import router as simple_router
from rag.self_query_rag import router as self_query_router
from rag.fusion_rag import router as fusion_router
from rag.qdrant_handler import router as qdrant_router

# Register routers
app.include_router(simple_router, prefix="/simple", tags=["Simple RAG"])
app.include_router(self_query_router, prefix="/self", tags=["Self-query RAG"])
app.include_router(fusion_router, prefix="/fusion", tags=["Fusion RAG"])
app.include_router(qdrant_router, prefix="/qdrant", tags=["Collection Management"])

@app.get("/")
def root():
    return {"message": "Multi-RAG PDF backend is running!"}

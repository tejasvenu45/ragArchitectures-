#  RAG Architectures - Multi-RAG Backend (FastAPI + Qdrant + Groq/HuggingFace)

This project implements three **Retrieval-Augmented Generation (RAG)** architectures using **FastAPI**, **Qdrant** vector store, and **Groq** (for LLM) + **HuggingFace** (for Embeddings). It enables querying and document uploading for context-aware responses from LLMs.

---

##  RAG Architectures Implemented

### 1.  Simple RAG

* **Retrieves top-K chunks** from the vector store based on the user query.
* Concatenates chunks into a context prompt for the LLM.
* **Use Case**: General-purpose document QA.

### 2. Self-query RAG

* Enhances retrieval by extracting metadata (topic, entities, section title) from chunks.
* Filters vector search using metadata to retrieve more relevant context.
* **Use Case**: Long documents with structured sections (e.g., research papers, policies).

### 3.  Fusion RAG (Query Fusion / Rewriting)

* Generates multiple semantically similar queries (n variants).
* Runs each through vector search and merges all results using **reciprocal rank fusion (RRF)**.
* **Use Case**: Short/ambiguous user queries, FAQs, or searches that benefit from query expansion.

---

## ⚙️ Tech Stack

* **FastAPI**: Backend framework
* **Qdrant**: Vector DB for chunk search
* **Groq**: LLM inference (e.g., `mixtral-8x7b-32768`)
* **HuggingFace Inference API**: For embeddings (e.g., `all-MiniLM-L6-v2`)
* **PyMuPDF / pdfminer**: PDF parsing
* **dotenv**: Secure API key management

---

##  How to Run

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/ragArchitectures.git
cd ragArchitectures
```

### 2. Create `.env`

```env
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
HF_API_TOKEN=your_huggingface_token
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start FastAPI server

```bash
uvicorn main:app --reload
```

> API will be available at: [http://localhost:8000](http://localhost:8000)

---

##  API Documentation

###  Upload Endpoints

#### `POST /simple/upload`

Upload a PDF file and index it into Qdrant.

```bash
curl -X POST "http://localhost:8000/simple/upload" -F "file=@/path/to/file.pdf"
```

#### `POST /self/upload`

Upload a PDF and extract metadata for self-query indexing.

```bash
curl -X POST "http://localhost:8000/self/upload" -F "file=@/path/to/file.pdf"
```

#### `POST /fusion/upload`

Upload PDF for Fusion RAG (uses standard chunking, but fusion on querying).

```bash
curl -X POST "http://localhost:8000/fusion/upload" -F "file=@/path/to/file.pdf"
```

---

###  Query Endpoints

#### `POST /simple/query`

```json
{
  "collection": "your_collection_name",
  "question": "What is the summary?",
  "top_k": 5
}
```

#### `POST /self/query`

```json
{
  "collection": "your_collection_name",
  "question": "What did the policy say about AI ethics?",
  "top_k": 5
}
```

#### `POST /fusion/query`

```json
{
  "collection": "your_collection_name",
  "question": "Achievements of Virat in T20",
  "num_queries": 3,
  "top_k": 4
}
```

> Returns: Final LLM answer + retrieved chunks + evaluation metrics (precision, recall, coverage).

---

###  Utility Endpoints

#### `DELETE /delete/{collection}`

Delete all documents in a given collection.

```bash
curl -X DELETE "http://localhost:8000/delete/virat"
```

#### `POST /clear-all`

Clear all documents from Qdrant (DANGER zone).

---

##  Deployment (Render)

1. Add your `.env` variables via Render Dashboard.
2. Use this `render.yaml`:

```yaml
services:
  - type: web
    name: rag-backend
    runtime: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn main:app --host 0.0.0.0 --port 10000"
    envVars:
      - key: QDRANT_URL
        fromDatabase: qdrant
      - key: QDRANT_API_KEY
        value: your_key
      - key: GROQ_API_KEY
        value: your_key
      - key: HF_API_TOKEN
        value: your_key
```

> Optionally add a `Dockerfile` if you want tighter control.

---

## RAG Evaluation (in response JSON)

* `retrieved_chunks`: All top chunks used in the LLM prompt.
* `precision / recall`: Measured based on overlap with GT if provided.
* `query_variants`: Shows how the system reformulates queries (in fusion).
* `metadata`: Shows how the system parses topic/entity/section info (in self-query).

---

##  Folder Structure

```
ragArchitectures/
│
├── main.py
├── simple_rag.py
├── self_query_rag.py
├── fusion_rag.py
├── embedding_handler.py  ← HuggingFace
├── groq_handler.py       ← Groq LLM
├── utils/
│   ├── pdf_parser.py
│   ├── metrics.py
├── requirements.txt
└── .env
```

---

##  Why This Matters

> RAG is the backbone of scalable document QA.

* Simple RAG for speed.
* Self-query for structured docs.
* Fusion RAG for ambiguous questions.

Combine them all for **adaptive retrieval based on context**.

---

##  Questions?

Open an issue or ping me at \[[tenacioutejas@gmail.com](mailto:tenacioutejas@gmail.com)]

---

##  Inspired by

* LangChain
* Qdrant Examples
* Groq LLMs
* HuggingFace Embedding APIs

---
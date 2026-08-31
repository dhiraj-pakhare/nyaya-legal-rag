# Nyaya Legal RAG

Nyaya Legal RAG is an open-source, structure-aware statutory retrieval and question-answering system engineered specifically for Indian criminal jurisprudence. It provides grounded legal reasoning across the **Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)** and the classification provisions of the **Bharatiya Nyaya Sanhita, 2023 (BNS)**.

The system incorporates coordinate-aware Gazette PDF extraction, hierarchical statutory chunking, dense vector retrieval (`BAAI/bge-base-en-v1.5`), sparse lexical retrieval (`BM25Okapi`), reciprocal rank fusion, deterministic section intent routing, cross-encoder reranking (`cross-encoder/ms-marco-TinyBERT-L-2-v2`), calibrated multi-factor confidence scoring, and strict citation-guarded generation with programmatic AST validation.

---

## Corpus Scope & Source-PDF Note

> [!IMPORTANT]
> **Source PDF Identity & Statutory Coverage**:
> The primary source file supplied in this repository is named `BNS bare act 2023.pdf` (249 pages). However, inspection of the official Gazette publication reveals that this document contains the enactment of Act No. 46 of 2023—the **Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)**.
>
> The ingested corpus is structured as follows:
> - **Pages 1–157 (BNSS Substantive Procedural Code)**: 39 Chapters, Sections 1 to 531, including all subsections, provisos, and explanations (555 canonical chunks).
> - **Pages 158–189 (The First Schedule)**: The classification table for offences under the **Bharatiya Nyaya Sanhita, 2023 (BNS)** (Sections 1–356), defining offence names, statutory punishment ranges, cognizability, bailability, and triable courts (472 canonical chunks).
> - **Pages 190–249 (The Second Schedule)**: 58 statutory procedural forms under the BNSS.
>
> **Substantive BNS Limitation**:
> The substantive text of the Bharatiya Nyaya Sanhita (Act No. 45 of 2023)—which defines definitions and elements of penal offences—is **not** present in the supplied source PDF. Consequently, queries targeting BNS provisions are grounded strictly and exclusively in the authentic classification and punishment evidence available within the First Schedule table.
>
> In accordance with zero-fabrication safety invariants, the system refuses queries when sufficient statutory evidence is absent, rather than fabricating text or relabeling BNSS procedural provisions as BNS substantive sections.

---

## Current Implementation Status

| Component / Milestone | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Coordinate-Aware Gazette PDF Parser & AST Structure Detector | Complete & Verified |
| **Phase 2** | Structure-Preserving Statutory Chunking + BGE Embeddings + Qdrant Indexing | Complete & Verified |
| **Phase 3** | BM25 Lexical Retrieval + Reciprocal Rank Fusion + Deterministic Section Routing | Complete & Verified |
| **Phase 4** | Cross-Encoder Reranking + Multi-Factor Confidence Scoring + Refusal Engine | Complete & Verified |
| **Phase 5** | Citation-Guarded LLM Generation & Programmatic AST Claim Verification | Complete & Verified |
| **Phase 6** | Multi-Tenant User Document RAG & Cryptographic Privacy Isolation | Complete & Verified |
| **Phase 7** | Statutory Forms Second Schedule Parser (58 Forms & Field Schema) | Complete & Verified |
| **Phase 8** | FastAPI Gateway + Rate Limiting + Prometheus Metrics + SSE Token Streaming | Complete & Verified |
| **Forms Export** | Automated PDF Form Slicer, ZIP Exporter, and Metadata Manifest | Complete & Verified |
| **Background Ops** | Asynchronous Document Ingestion Job Worker & Status Polling | Complete & Verified |
| **Frontend** | React 19 + TypeScript Chat UI, Interactive Citation Chips & Evidence Modal | Complete & Verified |

---

## Architecture

```
                                  [ User Query ]
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │  Deterministic Intent Detector  │
                        └────────────────┬────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │ Exact Match                               │ Conceptual / Hybrid
                   ▼                                           ▼
      ┌─────────────────────────┐                 ┌─────────────────────────┐
      │  Exact Section Lookup   │                 │   Dual-Stage Retrieval  │
      │   (O(1) in-memory dict) │                 │  Dense (BGE) + BM25Okapi│
      └────────────┬────────────┘                 └────────────┬────────────┘
                   │                                           │
                   │                                           ▼
                   │                              ┌─────────────────────────┐
                   │                              │  Reciprocal Rank Fusion │
                   │                              │       (RRF k=60)        │
                   │                              └────────────┬────────────┘
                   │                                           │
                   │                                           ▼
                   │                              ┌─────────────────────────┐
                   │                              │  Cross-Encoder Reranker │
                   │                              │     (Top-10 -> Top-5)   │
                   │                              └────────────┬────────────┘
                   │                                           │
                   │                                           ▼
                   │                              ┌─────────────────────────┐
                   │                              │ Multi-Factor Confidence │
                   │                              │    Gating (θ* = 0.75)   │
                   │                              └────────────┬────────────┘
                   │                                           │
                   ├───────────────────┬───────────────────────┘
                   │                   │ Score < 0.75
                   │                   ▼
                   │      ┌─────────────────────────┐
                   │      │ Grounded Refusal Engine │
                   │      └─────────────────────────┘
                   ▼
      ┌─────────────────────────┐
      │ LLM Answer Generation   │
      │ (Ollama / Local Models) │
      └────────────┬────────────┘
                   │
                   ▼
      ┌─────────────────────────┐
      │ Programmatic Validator  │◄── Enforces AST citation format
      │  (Sentence AST Check)   │    and statutory evidence match
      └────────────┬────────────┘
                   │
         ┌─────────┴─────────┐
         │ Valid             │ Invalid
         ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│ Stream SSE Post- │  │ Single Retry or  │
│ Validation Tokens│  │ Safe Refusal     │
└──────────────────┘  └──────────────────┘
```

### 1. Document Ingestion
- Ingests the 249-page official Gazette PDF using `pdfplumber` with bounding-box geometry.
- Dynamically strips recurring Gazette running headers, page numbers, and publication notices.
- Discovers and clusters marginal notes with sub-millisecond coordinate association, binding statutory section headings directly to their substantive section bodies.

### 2. Statutory Structure Parsing
- Recursively constructs an abstract syntax tree (AST) matching the legislative hierarchy:
  - 39 Chapters (`I` to `XXXIX`)
  - 531 Substantive Sections
  - 749 Subsections
  - 137 Statutory Provisos
  - 31 Explanations
- Parses the 472 BNS First Schedule offence entries into discrete tabular records containing section number, offence title, punishment, cognizability, bailability, and triable court.

### 3. Structure-Aware Chunking
- Preserves section atomicity: statutory sections under 3,200 characters remain intact.
- Longer sections are partitioned strictly at subsection boundaries with contextual breadcrumb headers:
  `[Act: BNSS | Chapter V | Section 35 | Subsection (1)(c)]`
- Produces exactly 1,027 canonical statutory chunks (555 substantive BNSS chunks + 472 BNS First Schedule chunks).

### 4. Retrieval (Dense + Sparse + RRF)
- **Dense Vector Search**: Encodes statutory text using `BAAI/bge-base-en-v1.5` (768 dimensions, L2-normalized cosine distance). Asymmetric query instructions (`Represent this sentence for searching relevant passages: `) align query semantics with passage vectors in Qdrant (`nyaya_legal_corpus`).
- **Sparse Lexical Search**: Custom BM25 search index using legal tokenization that generates synthetic section variants (e.g., `s.103` expands to `['s103', 'sec103', '103']`).
- **Reciprocal Rank Fusion (RRF)**: Merges top-25 dense and top-25 BM25 candidates with constant $k=60$:
  $$RRF(d) = \sum_{m \in \{dense, bm25\}} \frac{1}{k + r_m(d)}$$

### 5. Exact Section Routing
- Deterministic regex parser intercepts queries explicitly targeting statutory sections (e.g., `"What is Section 103 BNSS?"` or `"BNS s.303(2)"`).
- Enforces strict Act matching to eliminate cross-statute ambiguity, resolving directly to the target statutory chunk with zero retrieval overhead.

### 6. Reranking and Confidence Gating
- Reranks top-10 RRF candidates down to top-5 using a cross-encoder (`cross-encoder/ms-marco-TinyBERT-L-2-v2`).
- Multi-factor confidence score combines top cross-encoder relevance, the margin over the second-ranked candidate, and retriever agreement.
- Queries falling below the calibrated threshold ($\theta^* = 0.75$) trigger structured legal refusals rather than ungrounded generation.

### 7. Citation Validation and Evidence Display
- LLM outputs are held in memory and evaluated sentence-by-sentence by `CitationValidator` before token emission.
- Verifies every legal claim against the retrieved statutory evidence.
- Sentences containing statutory quotes with internal punctuation (e.g., `"Offence: Theft. Cognizable. Non-bailable. Any Magistrate"`) are handled quotation-aware to prevent false rejections.
- Passes verified `source_text` directly to client DTOs, enabling users to click citation chips in the UI to inspect the exact statutory passage.

### 8. User Document RAG
- Multi-tenant file ingestion supporting user-uploaded legal briefs, complaints, and contracts (`PDF`, `DOCX`, `TXT`).
- Cryptographically isolated tenant collections in Qdrant (`nyaya_user_documents`) partitioned by session scope.
- Enforces distinct citation tagging (`[DOC p.4]`) and prevents user document vectors from contaminating statutory search spaces.

### 9. API and Streaming
- Built on FastAPI with strict Pydantic schemas.
- Non-blocking Server-Sent Events (SSE) streaming at `/api/v1/query/stream`.
- Background asynchronous document processing with state-machine polling (`/api/v1/documents/{doc_id}/status`).
- In-memory rate limiting with `429 Too Many Requests` and `Retry-After` headers.
- Prometheus exposition endpoint at `/api/v1/metrics` tracking request latency, token counts, and refusal distributions.

### 10. Frontend Application
- Modern dark judicial UI built with React 19, TypeScript, and Vite.
- Real-time token streaming with SSE error recovery.
- Interactive citation chips (`[BNS s.103]`, `[BNSS s.35]`, `[DOC p.2]`, `[BNSS Second Schedule, Form 1]`).
- Evidence drawer displaying citation tag, Act name, section number, page range, chunk ID, verification badge, and verbatim statutory source text.
- User document manager with drag-and-drop upload and background processing indicators.
- Statutory forms catalog with PDF preview and bulk ZIP download.

### 11. Security and Isolation
- Prompt injection hardening: retrieved statutory and document text is explicitly delimited as untrusted data in LLM prompts.
- Programmatic AST validation rejects outputs if an LLM is hijacked into emitting uncited assertions.
- Multi-tenant tenant session isolation with Bearer authorization and token validation.

---

## Key Capabilities

- **Strict Evidence Grounding**: Answers are bounded by verified statutory excerpts. Unsupported claims are eliminated before emission.
- **Dual Criminal Code Navigation**: Seamlessly distinguishes procedural BNSS mechanisms from substantive BNS offence classifications.
- **Interactive Provenance**: Every citation chip in the frontend UI is clickable, opening a verification drawer displaying the exact retrieved statutory excerpt and Gazette page numbers.
- **Statutory Forms Repository**: Parses, catalogs, and exports all 58 statutory forms from the Second Schedule of the BNSS as discrete PDFs.
- **Production Observability**: Built-in `/health` diagnostic probe and `/api/v1/metrics` Prometheus exporter for operational visibility.

---

## Benchmark / Evaluation Results

Historical evaluation conducted on the curated 30-query golden set ([`eval/golden_set.jsonl`](eval/golden_set.jsonl)) across five retrieval configurations ([`eval_results_phase4.json`](eval_results_phase4.json)):

| Configuration | Recall@5 | Recall@10 | MRR | Out-of-Scope Refusal Rate | False Refusal Rate | p50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: Dense Only (`bge-base-en-v1.5`)** | 87.0% | 87.0% | 0.7261 | 100.0% | 100.0% | 230.7 ms |
| **Config B: BM25 Only (`BM25Okapi`)** | 100.0% | 100.0% | 0.8058 | 28.6% | 0.0% | 4.1 ms |
| **Config C: Hybrid (`Dense + BM25 + RRF`)** | 95.7% | 95.7% | 0.8326 | 100.0% | 100.0% | 216.4 ms |
| **Config D: Hybrid + Cross-Encoder Reranking** | 100.0% | 100.0% | 0.8913 | 100.0% | 17.4% | 308.4 ms |
| **Config E: Auto (`Intent + Exact + Hybrid + Reranker`)** | **100.0%** | **100.0%** | **0.9130** | **100.0%** | **0.0%** | **260.1 ms** |

*Note: In Config E, intent detection routes explicit section queries directly via exact lookup, achieving 100% recall and an MRR of 0.9130 while maintaining a 100% refusal rate on out-of-scope queries.*

---

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+ and npm
- Local Ollama instance (default model: `qwen2.5:3b` or `llama3.2`) or OpenAI-compatible API
- Source Gazette PDF: `BNS bare act 2023.pdf` placed in the repository root directory

### Backend Setup
```bash
# 1. Clone repository
git clone https://github.com/dhiraj-pakhare/nyaya-legal-rag.git
cd nyaya-legal-rag

# 2. Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env

# 5. Ingest Gazette PDF into Qdrant vector storage
python3 scripts/ingest.py

# 6. Extract all 58 statutory forms into discrete PDFs
python3 scripts/extract_forms.py
```

### Frontend Setup
```bash
# Navigate to frontend directory and install dependencies
cd frontend
npm install
cd ..
```

---

## Running the Application

### 1. Start the Backend API
From the repository root:
```bash
PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000
```
- API Documentation (Swagger UI): `http://localhost:8000/docs`
- Health Diagnostic Endpoint: `http://localhost:8000/health`
- Prometheus Metrics: `http://localhost:8000/api/v1/metrics`

### 2. Start the Frontend Dev Server
In a separate terminal:
```bash
cd frontend
npm run dev
```
- Application Web UI: `http://localhost:5173/chat` (or `http://localhost:5174/chat` if port 5173 is in use)

---

## Docker Architecture & Containerized Deployment

Nyaya Legal RAG provides a multi-container Docker Compose architecture for production deployment, defining isolated services, persistent volumes, healthchecks, and internal networking:

```
┌────────────────────────────────────────────────────────┐
│               Host Browser / Client                    │
└───────────────┬────────────────────────┬───────────────┘
                │ :5173                  │ :8000
                ▼                        ▼
┌──────────────────────────┐   ┌─────────────────────────┐
│ nyaya-frontend (Nginx)   │───│ nyaya-api (FastAPI)     │
└──────────────────────────┘   └───────────┬─────────────┘
                                           │
         ┌─────────────────────────────────┼────────────────────────┐
         ▼                                 ▼                        ▼
┌──────────────────────────┐   ┌─────────────────────────┐   ┌──────────────────────────┐
│ nyaya-worker (Worker)    │   │ nyaya-qdrant (Vector DB)│   │ nyaya-redis (Queue)      │
│ Ingestion & Queue Poller │   │ Port 6333, Volume       │   │ Internal (not exposed)   │
└──────────────────────────┘   └─────────────────────────┘   └──────────────────────────┘
```

### Services & Port Configuration
| Service | Image / Build | Container Name | Host Port | Internal Port | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`api`** | Multi-stage `Dockerfile` (Python 3.11-slim) | `nyaya-api` | `8000` | `8000` | FastAPI gateway, hybrid search, citation validation |
| **`worker`** | Shared `Dockerfile` (Python 3.11-slim) | `nyaya-worker` | — | — | Standalone background ingestion & task processor |
| **`qdrant`** | `qdrant/qdrant:v1.13.2` | `nyaya-qdrant` | `6333` | `6333` | Vector database (1,027 statutory points + user docs) |
| **`redis`** | `redis:7-alpine` | `nyaya-redis` | — | `6379` | Message broker & queue (internal only, not exposed) |
| **`frontend`** | Multi-stage `frontend/Dockerfile` (Nginx) | `nyaya-frontend` | `5173` | `80` | Production React SPA with reverse proxy & SSE support |

### Persistent Named Volumes
- **`nyaya_qdrant_data`**: Persists the Qdrant vector database (`/qdrant/storage`) across container restarts.
- **`nyaya_redis_data`**: Persists Redis queue state and keys (`/data`).
- **`nyaya_app_data`**: Shared volume (`/app/data`) for extracted statutory forms (`data/forms/`) and tenant documents.

### Docker Setup Instructions

#### 1. Prerequisites
- Docker Engine 24+ and Docker Compose v2.20+
- Host Ollama instance running with model `qwen2.5:3b` (or configure external API key)
- Source Gazette PDF: `BNS bare act 2023.pdf` placed in the repository root

#### 2. Build the Docker Images
```bash
docker compose build
```

#### 3. Start the Stack
```bash
docker compose up -d
```

#### 4. Run the Idempotent Bootstrap Script
Populates the containerized Qdrant instance with all 1,027 statutory vectors and extracts the 58 Second Schedule forms:
```bash
docker compose exec api /app/scripts/bootstrap.sh
```
*(If already initialized, the bootstrap script verifies counts and completes in under 2 seconds without duplicating data).*

#### 5. Verify Service Health
```bash
docker compose ps
```
Confirm that all 5 services report `Up (healthy)`.

#### 6. Access the Application
- **Frontend Web UI**: `http://localhost:5173/chat`
- **FastAPI Documentation (Swagger)**: `http://localhost:8000/docs`
- **Health Diagnostic Probe**: `http://localhost:8000/health` (and `http://localhost:8000/api/v1/health`)
- **Prometheus Metrics**: `http://localhost:8000/api/v1/metrics`
- **Qdrant Dashboard / Health**: `http://localhost:6333/dashboard` and `http://localhost:6333/healthz`

#### 7. Inspecting Logs
```bash
# View aggregated real-time logs
docker compose logs -f

# Inspect specific services
docker compose logs -f api
docker compose logs -f worker
```

#### 8. Stopping and Restarting the Stack
```bash
# Graceful shutdown (preserves all vector data and extracted forms)
docker compose down

# Restart the stack
docker compose up -d

# Rebuild containers after code modifications
docker compose up -d --build

# CAUTION: Reset all persistent volumes and data (clean wipe)
docker compose down -v
```

---

## Verification

### Automated Backend Test Suite
Run the full backend test suite:
```bash
PYTHONPATH=. pytest backend/tests/ -v
```
**Current Verified Result**:
```
================== 249 passed, 1 warning in 281.09s ==================
```
*(1 warning corresponds to standard local embedded Qdrant payload indexing notice).*

### Frontend Typecheck & Build
```bash
cd frontend
npm run lint
npm run build
```
Builds the static bundle using TypeScript (`tsc -b`) and Vite with zero errors.

### Health Probe Verification
```bash
curl -s http://localhost:8000/health
```
**Response**:
```json
{"status":"UP","timestamp":"2026-08-31T17:43:41.705854+00:00","version":"1.0.0"}
```

---

## Repository Structure

```
nyaya-legal-rag/
├── backend/
│   ├── app/
│   │   ├── api/                  # FastAPI routers, schemas, deps, rate limiting
│   │   │   ├── routes/           # Query, chat, documents, forms, health, metrics
│   │   │   └── schemas/          # Pydantic DTOs for requests and citations
│   │   ├── core/                 # App config, embeddings, Qdrant repo, metrics
│   │   ├── document_rag/         # User document RAG pipeline, parsing, isolation
│   │   ├── forms/                # Statutory forms parser, repository, PDF exporter
│   │   ├── generation/           # LLM providers, prompt templates, citation validator
│   │   ├── ingestion/            # PDF layout extractor, structure detector, chunker
│   │   ├── retrieval/            # BM25, exact section lookup, RRF, reranker, pipeline
│   │   ├── services/             # LegalQueryService coordinator
│   │   ├── workers/              # Ingestion worker, job manager, standalone worker
│   │   │   ├── ingestion_worker.py
│   │   │   ├── job_manager.py
│   │   │   └── worker.py         # Standalone background worker process
│   │   └── main.py               # Application factory and middleware assembly
│   └── tests/                    # 249 unit, integration, and security tests
├── frontend/
│   ├── src/
│   │   ├── api/                  # API client and TypeScript DTO definitions
│   │   ├── components/           # CitationChips, CitationSourceModal, Navbar, Sidebar
│   │   ├── hooks/                # useSSEStream, useDocuments, useAuth
│   │   ├── views/                # ChatView, DocumentsView, FormsView
│   │   ├── App.tsx               # Root view router
│   │   └── index.css             # Judicial theme design system
│   ├── Dockerfile                # Multi-stage production Nginx frontend image
│   ├── nginx.conf                # Nginx reverse proxy configuration for SPA & SSE
│   ├── package.json              # React 19, TypeScript, Vite scripts
│   └── vite.config.ts            # Vite dev proxy configuration
├── eval/
│   └── golden_set.jsonl          # 30-query golden evaluation benchmark set
├── scripts/
│   ├── bootstrap.sh              # Idempotent system bootstrap & ingestion script
│   ├── ingest.py                 # Corpus extraction and Qdrant ingestion script
│   ├── extract_forms.py          # Statutory forms PDF extractor script
│   ├── calibrate_threshold.py    # Multi-factor confidence calibration sweep
│   └── benchmark_reranking.py    # 5-configuration retrieval & reranking benchmark
├── ARCHITECTURE.md               # Detailed system design and AST specifications
├── DECISIONS.md                  # Architectural Decision Records (ADRs)
├── Dockerfile                    # Multi-stage production API and Worker image
├── docker-compose.yml            # Multi-container Compose definition
├── IMPLEMENTATION_PLAN.md        # Development milestone documentation
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment configuration template
```

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): Comprehensive system architecture, statutory AST specifications, and retrieval data-flow designs.
- [DECISIONS.md](DECISIONS.md): Architectural Decision Records (ADRs) detailing trade-offs, parser evaluations, and design rationales.
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): Complete development milestone tracker and roadmap history.

---

## Important Limitations

1. **Substantive BNS Penal Code Scope**: The supplied source PDF (`BNS bare act 2023.pdf`) is the Gazette of Act No. 46 of 2023 (BNSS). Penal offence queries for the BNS rely on the authentic First Schedule classification table (punishment, cognizability, bailability, triable court). Substantive offence definitions under BNS Act No. 45 of 2023 are not present in this corpus.
2. **Local Model Hardware Constraints**: When utilizing local Ollama models (e.g., `qwen2.5:3b`), generation latency depends directly on host CPU/GPU availability.
3. **Statutory Forms Pre-Filling**: Statutory forms extracted from the Second Schedule are read-only procedural templates. Dynamic interactive form completion with case metadata is outside the current scope.

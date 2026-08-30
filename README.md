# Nyaya Legal RAG

**Nyaya Legal RAG** is an open-source, structure-aware statutory retrieval system engineered specifically for Indian criminal jurisprudence—the **Bharatiya Nyaya Sanhita, 2023 (BNS)** and the **Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)**.

The system features coordinate-aware Gazette PDF parsing, hierarchical statutory chunking, dense vector retrieval (`BAAI/bge-base-en-v1.5`), sparse keyword retrieval (`BM25Okapi`), reciprocal rank fusion, deterministic section-number intent routing, cross-encoder reranking (`ms-marco-TinyBERT-L-2-v2` / `BAAI/bge-reranker-base`), multi-factor confidence scoring, and an empirically calibrated zero-hallucination refusal mechanism.

---

## Current Implementation Status

| Phase / Part | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | **PDF Inspection & Structure-Aware Statutory Parser** | :white_check_mark: **Complete & Validated** |
| **Phase 2** | **Statutory Chunking + BGE Embeddings + Qdrant Indexing** | :white_check_mark: **Complete & Validated** |
| **Phase 3** | **BM25 Keyword Retrieval + RRF + Exact Section Intent Routing** | :white_check_mark: **Complete & Validated** |
| **Phase 4** | **Cross-Encoder Reranking + Multi-Factor Confidence + Refusal Engine** | :white_check_mark: **Complete & Validated** |
| **Phase 5** | **Citation-Guarded LLM Generation & Safety (AST Verification)** | :white_check_mark: **Complete & Validated** |
| **Phase 6** | **Multi-Tenant User Document RAG & Privacy Isolation** | :white_check_mark: **Complete & Validated** |
| **Phase 7** | **Statutory Forms Second Schedule Parser (58 Forms)** | :white_check_mark: **Complete & Validated** |
| **Phase 8** | **FastAPI Gateway + Hardened Security + SSE Streaming** | :white_check_mark: **Complete & Validated** |
| **Part B** | **Statutory Forms PDF Exporter + Manifest + Download & ZIP API** | :white_check_mark: **Complete & Validated** |
| **Part D** | **Asynchronous Background Ingestion + Job State Machine & Status Probe** | :white_check_mark: **Complete & Validated** |



---

## Key Architecture & Capabilities (Phases 1–4)

1. **Coordinate-Aware Gazette PDF Ingestion**:
   - Parses the exact 249-page Gazette PDF (`BNS bare act 2023.pdf`).
   - Dynamic boundary isolation separating Gazette headers, footers, colophons, marginal notes, and body text.
   - Detects all 39 Chapters (I–XXXIX), 531 Substantive Sections (1–531), 749 Subsections, 137 Provisos, and 31 Explanations.
   - Parses 472 First Schedule offence classification table entries (BNS Sections 1–356).

2. **Structure-Aware Statutory Chunking**:
   - Preserves section atomicity ($\le 3,200$ chars intact).
   - Long sections split only at subsection boundaries with contextual statutory breadcrumbs.
   - Generates 1,027 canonical chunks with deterministic IDs.

3. **Dense Vector Indexing (`BAAI/bge-base-en-v1.5` + Qdrant)**:
   - 768-dimensional L2-normalized embeddings stored in Qdrant (`nyaya_legal_corpus`).
   - Asymmetric query instruction prefixing.
   - Deterministic UUIDv5 point IDs for idempotent re-indexing.

4. **Hybrid Retrieval with Reciprocal Rank Fusion**:
   - Combines Dense semantic search ($k=25$) and BM25 sparse search ($k=25$) via RRF ($k=60$).
   - Custom legal tokenizer with synthetic section expansion (`s.103` $\rightarrow$ `['s103', 'sec103', '103']`).

5. **Deterministic Section Intent Routing**:
   - Detects section-lookup queries (e.g. `"What is section 103 BNS?"`, `"BNS s.103"`, `"Explain section 35"`) and routes directly to $O(1)$ exact statutory lookup.

6. **Cross-Encoder Reranking & Calibrated Refusal Engine**:
   - Cross-encoder reranks top-10 RRF candidates to top-5 with Sigmoid-normalized scoring.
   - Multi-factor confidence combining top relevance score, margin, and dual-retriever agreement.
   - Empirically calibrated threshold ($\theta^* = 0.75$) delivering **100.0% Out-of-Scope Refusal Rate** and **0.0% False Refusal Rate**.

---

## Quantitative Benchmark Results (30-Query Golden Set)

| Configuration | Recall@5 | Recall@10 | MRR | Out-of-Scope Refusal Rate | In-Scope False Refusal Rate | p50 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: Dense Only (`bge-base-en-v1.5`)** | 87.0% | 87.0% | 0.7261 | 100.0% | 100.0% | 230.7 ms |
| **Config B: BM25 Only (`BM25Okapi`)** | 100.0% | 100.0% | 0.8058 | 28.6% | 0.0% | 4.1 ms |
| **Config C: Hybrid (`Dense + BM25 + RRF`)** | 95.7% | 95.7% | 0.8326 | 100.0% | 100.0% | 216.4 ms |
| **Config D: Hybrid + Cross-Encoder Reranking** | 100.0% | 100.0% | 0.8913 | 100.0% | 17.4% | 308.4 ms |
| **Config E: Auto (`Intent + Exact + Hybrid + Reranker`)** | **100.0%** | **100.0%** | **0.9130** | **100.0%** | **0.0%** | **260.1 ms** |

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- Source Gazette PDF: `BNS bare act 2023.pdf` placed in the root repository directory.

### Setup Instructions
```bash
# 1. Clone repository
git clone <repo-url>
cd nyaya-legal-rag

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env

# 5. Ingest PDF and index corpus into Qdrant
python3 scripts/ingest.py

# 6. Extract all 58 statutory forms into discrete PDFs and manifest
python3 scripts/extract_forms.py
```

---

## Running Verification & Benchmarks

```bash
# Run complete test suite (224 tests passing, 100% green)
python3 -m pytest backend/tests/ -v

# Run confidence threshold calibration sweep
python3 scripts/calibrate_threshold.py

# Run comprehensive 5-configuration retrieval & reranking benchmark
python3 scripts/benchmark_reranking.py
```


---

## Documentation Links

- [ARCHITECTURE.md](file:///Users/_dhirajp2004_/Desktop/nyaya-legal-rag:/ARCHITECTURE.md): System architecture, statutory AST specifications, and data flow.
- [DECISIONS.md](file:///Users/_dhirajp2004_/Desktop/nyaya-legal-rag:/DECISIONS.md): Architectural Decision Records (ADRs) with technical rationale and empirical evidence.
- [IMPLEMENTATION_PLAN.md](file:///Users/_dhirajp2004_/Desktop/nyaya-legal-rag:/IMPLEMENTATION_PLAN.md): End-to-end phased development roadmap.

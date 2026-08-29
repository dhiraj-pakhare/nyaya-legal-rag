# Phased Implementation Plan — Nyaya Legal RAG

This implementation plan details the step-by-step engineering roadmap for building the complete Nyaya Legal RAG system in strict compliance with the DhronAI technical assignment requirements (Parts A–F).

---

## 1. Phased Roadmap

---

### PHASE 1: PDF Inspection & Structure-Aware Parser
- **Objective**: Parse the exact supplied `BNS bare act 2023.pdf` (249 pages), stripping Gazette running headers/footers, extracting marginal notes (section titles), parsing substantive Sections 1–531, parsing **The First Schedule (Pages 158–189) BNS Offence Table**, and constructing a clean statutory document model.
- **Files/Modules Expected**:
  - `backend/app/ingestion/pdf_parser.py`
  - `backend/app/ingestion/first_schedule_parser.py` (Parser for Schedule I BNS offences, punishments, and triable courts)
  - `backend/app/ingestion/models.py` (Pydantic AST for Chapter, Section, Subsection, Proviso, ScheduleEntry, Form)
  - `backend/app/ingestion/cleaner.py` (De-hyphenation, Gazette header/footer removal)
- **Dependencies**: `pypdf`, `pdfplumber` / `pypdf2`, `pydantic`
- **Tests Required**:
  - `backend/tests/test_pdf_parser.py`: Unit test extracting Section 1, Section 2, Section 35, Section 63, Section 141, Section 187, and Section 531 from fixture pages.
  - Unit test verifying The First Schedule BNS entries (e.g. BNS s.105, s.63) are correctly extracted into structured records.
  - Test verifying Gazette footers (`THE GAZETTE OF INDIA EXTRAORDINARY`) and horizontal lines (`______`) are completely filtered.
- **Acceptance Criteria**: All 531 substantive procedural sections and all First Schedule BNS offence classifications are extracted into structured objects without losing text or omitting marginal note titles.
- **Likely Failure Modes**:
  - Marginal notes (section titles) detached from section numbers.
  - Missing the First Schedule offences table causing zero retrieval on BNS penal queries.
  - Form feed characters and page breaks splitting legal definitions in Section 2.
  - Hyphenated words across line breaks (e.g. `inves-` / `tigation`) remaining broken.

---

### PHASE 2: Chunking & Metadata Validation
- **Objective**: Implement statutory section-aware chunking where the section is the atomic unit, splitting long sections only at subsection `(1)` / clause `(a)` boundaries, structuring First Schedule BNS offence chunks, and guaranteeing that Provisos, Exceptions, Explanations, and Illustrations remain attached to their parent section.
- **Files/Modules Expected**:
  - `backend/app/ingestion/chunker.py`
  - `backend/app/ingestion/schedule_chunker.py`
  - `backend/app/ingestion/schema.py` (Statutory chunk metadata schema supporting both BNS & BNSS)
  - `backend/app/ingestion/cross_ref.py` (Regex extractor for cross-statute references)
- **Dependencies**: `pydantic`, `regex`
- **Tests Required**:
  - `backend/tests/test_chunker.py`: Unit test verifying that Provisos stay attached to parent clauses.
  - Unit test verifying Illustrations and Explanations are never orphaned.
  - Test verifying short sections are not split and long sections are split cleanly at sub-clauses.
  - Test verifying both BNS offence chunks and BNSS procedural chunks conform to schema.
- **Acceptance Criteria**: Chunk metadata conforms 100% to the assignment specification schema (`act`, `act_short`, `chapter`, `chapter_title`, `section_number`, `section_title`, `subsection`, `clause`, `has_proviso`, `has_exception`, `has_illustration`, `page_start`, `page_end`, `chunk_id`).
- **Likely Failure Modes**:
  - Overlapping chunker duplicating middle sentences instead of providing hierarchical contextual prefixes.
  - Provisos appearing in separate isolated chunks causing inverted legal meanings.

---

### PHASE 3: Open-Source Embeddings & Qdrant Vector Storage
- **Objective**: Set up open-source embedding pipeline (`BAAI/bge-base-en-v1.5`) and index the statutory chunks into a local Docker-compatible Qdrant instance.
- **Files/Modules Expected**:
  - `backend/app/retrieval/embeddings.py` (Embedding model wrapper with query/passage prefix handling)
  - `backend/app/retrieval/vector_store.py` (Qdrant client, schema initialization, batch upsert)
  - `scripts/ingest.py` (One-shot CLI ingestion script for BNS PDF)
- **Dependencies**: `qdrant-client`, `sentence-transformers`, `torch` (CPU), `numpy`
- **Tests Required**:
  - `backend/tests/test_vector_store.py`: Integration test spinning up in-memory Qdrant instance, inserting fixture chunks, and asserting dense cosine nearest-neighbor retrieval.
- **Acceptance Criteria**:
  - Ingestion embeds all substantive and schedule chunks in $<60\text{s}$ on CPU.
  - Vectors stored with full metadata payloads and cosine distance metric.
- **Likely Failure Modes**:
  - Forgetting the mandatory query prefix (`"Represent this sentence for searching relevant passages: "`) for BGE embeddings, silently degrading recall.
  - Running embedding generation synchronously on every container boot rather than via one-time bootstrap.

---

### PHASE 4: Hybrid Retrieval & Deterministic Section Lookup
- **Objective**: Implement hybrid dense ($k=25$) + BM25 ($k=25$) retrieval fused with Reciprocal Rank Fusion ($k_{rrf}=30, k_{rrf\_const}=60$) alongside a deterministic section-number intent bypass.
- **Files/Modules Expected**:
  - `backend/app/retrieval/bm25.py` (BM25 index builder and query engine with disk caching)
  - `backend/app/retrieval/hybrid.py` (RRF rank fusion combining dense and sparse results)
  - `backend/app/retrieval/deterministic.py` (Exact section pattern detector and payload lookup)
  - `backend/app/retrieval/router.py` (Query router handling `BNS s.X`, `BNSS s.X`, and unqualified queries)
- **Dependencies**: `rank-bm25`, `numpy`
- **Tests Required**:
  - `backend/tests/test_retrieval.py`: Test that `"section 35"` deterministically returns BNSS Section 35 at Rank 1.
  - Test that `"BNS section 105"` deterministically returns BNS Schedule 1 Section 105 entry at Rank 1.
  - Test that semantic queries retrieve conceptually relevant sections.
- **Acceptance Criteria**:
  - Direct section queries return the exact statutory section 100% of the time.
  - RRF combines dense and sparse candidate lists smoothly without score distortion.
- **Likely Failure Modes**:
  - Regex missing compound section notations like `s. 35(1)(a)`.
  - BM25 tokenizer failing on legal numbering or punctuation.

---

### PHASE 5: Cross-Encoder Reranking & Provisional Confidence Refusal
- **Objective**: Add cross-encoder reranking ($k_{in}=10 \rightarrow k_{out}=5$) using `cross-encoder/ms-marco-MiniLM-L-6-v2` and implement a configurable confidence threshold with an empirical calibration harness.
- **Files/Modules Expected**:
  - `backend/app/retrieval/reranker.py` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - `backend/app/retrieval/refusal.py` (Configurable confidence threshold estimator & refusal arbiter)
- **Dependencies**: `sentence-transformers`, `scipy`
- **Tests Required**:
  - `backend/tests/test_refusal.py`: Test asserting out-of-scope query `"What is the punishment for jaywalking in Ohio?"` triggers immediate refusal below threshold.
  - Test asserting in-scope queries pass confidence validation.
- **Acceptance Criteria**:
  - Threshold is configurable via `CONFIDENCE_THRESHOLD` (provisional default: $\theta = 0.35$).
  - Refusal triggers on low confidence before invoking LLM generation.
- **Likely Failure Modes**:
  - Hardcoding a single un-calibrated threshold without measuring the Precision-Recall curve.

---

### PHASE 6: Buffered Citation Validation & Zero-Unvalidated Streaming
- **Objective**: Enforce the non-negotiable citation contract with server-side buffering: complete LLM generation, programmatically validate all cited sections against retrieved context chunk IDs, and stream verified text with citation chips to the client.
- **Files/Modules Expected**:
  - `backend/app/llm/citation_guard.py` (Programmatic AST citation validator & regeneration handler)
  - `backend/app/llm/prompts.py` (Statutory legal prompt templates with citation instructions)
  - `backend/app/llm/provider.py` (Abstract `LLMProvider` interface and adapters)
- **Dependencies**: `pydantic`, `httpx`, `groq`, `openai`
- **Tests Required**:
  - `backend/tests/test_citation_guard.py`: Test that a mock response citing non-existent `[BNS s.999]` is caught and triggers regeneration/refusal before client delivery.
- **Acceptance Criteria**:
  - Zero unvalidated legal claims are exposed to the user.
  - Every legal claim has a verified inline citation.
  - Verified citations link directly to chunk metadata (page, section, chapter) for UI drawer display.
- **Likely Failure Modes**:
  - Streaming unvalidated tokens directly to the client, exposing hallucinations.

---

### PHASE 7: Evaluation Suite, Calibration & Golden Set (Part F)
- **Objective**: Build golden evaluation dataset (`eval/golden_set.jsonl`) with 30 annotated queries (15 lookup, 10 reasoning, 5 must-refuse), calibrate the confidence threshold, and benchmark 3 retrieval configurations.
- **Files/Modules Expected**:
  - `eval/golden_set.jsonl`
  - `eval/run_eval.py` (Evaluation runner with `--config [dense|hybrid|rerank]` and threshold sweeping)
  - `eval/results/comparison_report.json`
- **Dependencies**: `pytest`, `pandas`, `tabulate`
- **Tests Required**:
  - Automated CI test running evaluation against golden set.
- **Acceptance Criteria**:
  - Generates comparative markdown table comparing Config A (Dense), Config B (Hybrid RRF), and Config C (Hybrid + Reranking).
  - Reports Recall@5, Recall@10, MRR, Citation Accuracy %, Refusal Rate %, and split p50/p95 latency (Retrieval vs Generation).
- **Likely Failure Modes**:
  - Subjective adjectives in README instead of concrete numeric benchmark metrics.

---

### PHASE 8: User Document RAG, Multi-Corpus Routing & Isolation
- **Objective**: Implement isolated user document ingestion, session-scoped Qdrant filtering, prompt injection defense, and multi-corpus routing.
- **Files/Modules Expected**:
  - `backend/app/ingestion/user_doc_parser.py`
  - `backend/app/retrieval/multi_corpus.py` (Dual routing between Statute index & Session index)
  - `backend/app/core/security.py` (Prompt injection scan & XML framing)
- **Dependencies**: `pypdf`, `qdrant-client`
- **Tests Required**:
  - `backend/tests/test_session_isolation.py`: Test asserting Session A cannot retrieve documents uploaded by Session B.
  - `backend/tests/test_prompt_injection.py`: Test verifying injected override strings inside uploaded PDF are treated strictly as untrusted data.
- **Acceptance Criteria**:
  - Document questions query session index; statute questions query BNS index; compliance questions query both with distinct citation chips.
  - Cross-user document access returns 404.
- **Likely Failure Modes**:
  - Uploaded document vectors leaking into global statute searches.

---

### PHASE 9: Statutory Forms Extraction Pipeline (Pages 190–249)
- **Objective**: Extract all 58 statutory forms from pages 190–249 into individual vector-grade PDFs, scrape titles dynamically, collate multi-page forms (Form 33 over pages 222–224), and generate `forms_manifest.json`.
- **Files/Modules Expected**:
  - `backend/app/forms/extractor.py`
  - `backend/app/forms/manifest.py`
  - `scripts/extract_forms.py`
- **Dependencies**: `pypdf`, `pdfplumber` / `fitz`, `pytesseract` (fallback)
- **Tests Required**:
  - `backend/tests/test_forms.py`: Test verifying Form 33 has page count = 3 and Form 12 is extracted with correct scraped title.
  - Test verifying manifest SHA-256 and byte sizes are generated properly.
- **Acceptance Criteria**:
  - All 58 forms generated with deterministic names `FORM-<num>_<slug>.pdf`.
  - `forms_manifest.json` emitted with form number, scraped title, page range, SHA-256, byte size, confidence score, and `needs_review` flag.
  - Zero hardcoded title lists.
- **Likely Failure Modes**:
  - Naive 1-page loop splitting multi-page Form 33 into 3 broken fragments.
  - Hardcoded title strings causing automatic zero on Part B.

---

### PHASE 10: FastAPI Backend & Async Task Worker
- **Objective**: Expose all required REST and SSE endpoints with async background task queue, rate limiting, and Prometheus metric telemetry.
- **Files/Modules Expected**:
  - `backend/app/main.py`
  - `backend/app/api/v1/chat.py` (`POST /api/v1/chat` with SSE)
  - `backend/app/api/v1/documents.py` (`POST /upload`, `GET /status`, `GET /`, `DELETE /{id}`)
  - `backend/app/api/v1/forms.py` (`GET /`, `GET /{id}/download`, `GET /download-all`, `GET /search`)
  - `backend/app/api/v1/feedback.py`, `health.py`, `metrics.py`
  - `backend/app/workers/ingest_worker.py`
- **Dependencies**: `fastapi`, `uvicorn`, `sse-starlette`, `prometheus-client`, `slowapi`
- **Tests Required**:
  - `backend/tests/test_api_endpoints.py`: End-to-end API test covering chat streaming, document upload, status polling, forms download, and 404 security checks.
- **Acceptance Criteria**:
  - All endpoints from assignment specification functional and documented at `/docs`.
  - Ingestion of 60-page PDF executes asynchronously in worker without blocking Uvicorn.
- **Likely Failure Modes**:
  - Blocking HTTP request loop during embedding generation.

---

### PHASE 11: Modern React Frontend & UX (Part C)
- **Objective**: Build a responsive two-panel ChatGPT-style legal workspace with token streaming, interactive citation chips, slide-out statutory source drawer, upload progress stages, and searchable forms library.
- **Files/Modules Expected**:
  - `frontend/src/App.jsx`
  - `frontend/src/components/ChatPanel.jsx`
  - `frontend/src/components/CitationChip.jsx`
  - `frontend/src/components/SourceDrawer.jsx`
  - `frontend/src/components/DocumentUploader.jsx`
  - `frontend/src/components/FormsPanel.jsx`
  - `frontend/src/components/DisclaimerBanner.jsx`
- **Dependencies**: `react`, `react-dom`, `vite`, `tailwindcss`, `lucide-react`, `clsx`
- **Tests Required**:
  - Frontend component smoke tests & accessibility verification.
- **Acceptance Criteria**:
  - Smooth validated SSE token streaming without layout shifts.
  - Clicking citation chips slides open the source drawer showing exact statutory text and page numbers.
  - Upload shows real-time stages: `parse` $\rightarrow$ `chunk` $\rightarrow$ `embed` $\rightarrow$ `ready`.
  - Dark & Light mode toggle, full WCAG AA contrast.
- **Likely Failure Modes**:
  - Spinner-then-wall-of-text (violates Part C streaming requirement).
  - Missing source drawer.

---

### PHASE 12: Docker Containerization, CI/CD & Observability
- **Objective**: Multi-stage Docker build, docker-compose orchestration, Prometheus/Grafana monitoring, and GitHub Actions CI workflow with self-hosted runner support.
- **Files/Modules Expected**:
  - `backend/Dockerfile`
  - `frontend/Dockerfile`
  - `docker-compose.yml`
  - `monitoring/prometheus.yml`, `monitoring/grafana/`
  - `.github/workflows/ci.yml`
  - `scripts/bootstrap.sh`
- **Dependencies**: `docker`, `docker-compose`
- **Tests Required**:
  - Clean clone `docker-compose up` smoke test.
  - GitHub Actions CI workflow run with linting, unit tests, secret scan (gitleaks), and Docker build.
- **Acceptance Criteria**:
  - `docker-compose up` starts API, Frontend, Qdrant, and Prometheus with healthy status.
  - `scripts/bootstrap.sh` executes idempotent ingestion and forms extraction.
  - Zero committed credentials (verified by secret scanning in CI).
- **Likely Failure Modes**:
  - Bloated 4GB Docker image with CUDA dependencies instead of slim CPU build.
  - Hardcoded container localhost ports causing connection failures inside Docker network.

---

### PHASE 13: Final Validation, README & Demo Walkthrough
- **Objective**: Run full test suite, verify clean startup from zero, compile complete `README.md` with AI disclosures and benchmark tables, and prepare 5–8 minute Loom demonstration.
- **Files/Modules Expected**:
  - `README.md`
  - `DECISIONS.md`
  - `ARCHITECTURE.md`
- **Acceptance Criteria**:
  - Complete graded deliverable package meeting all requirements in Parts A–F.

---

## 2. High-Risk Evaluation Traps & Mitigations

| # | Evaluation Trap | Root Cause / Impact | Architectural Mitigation |
| :--- | :--- | :--- | :--- |
| **1** | **Naive Fixed-Size Chunking** | `RecursiveCharacterTextSplitter(512)` cuts across sections and splits legal definitions. | Custom hierarchy AST chunker treating **Section** as atomic unit. |
| **2** | **Wrong PDF Source / Substitution** | Downloading an external BNS PDF instead of using the exact 249-page supplied volume. | Strictly ingest `BNS bare act 2023.pdf` located in workspace. |
| **3** | **Orphaned Provisos / Exceptions / Illustrations** | Splitting a section before its proviso causes the model to state the opposite of the law. | Chunker explicitly attaches Provisos, Exceptions, Explanations, and Illustrations to parent section chunks. |
| **4** | **Incorrect / Missing Citations** | Answering legal questions without citations or wrong statutory codes. | Mandatory citation contract enforcing `[BNSS s.X(Y)]` and `[BNS s.X]` formats. |
| **5** | **Hallucinated Section Numbers** | LLM inventing section numbers (e.g. s.999) from parametric memory. | Programmatic regex citation guard comparing cited sections against retrieved context chunk IDs before client delivery. |
| **6** | **Dense-Only Retrieval** | Dense vector search misses exact statutory codes (e.g. "Section 103 BNS"). | Hybrid retrieval (Dense + BM25) with Reciprocal Rank Fusion ($k=60$). |
| **7** | **Non-Deterministic Section Lookup** | User asks "What is section 35?" and receives unrelated cosine neighbors. | Intent parser detecting section numbers and deterministically injecting exact section at Rank 1. |
| **8** | **Hardcoded Form Titles** | Hardcoding form titles instead of scraping them (automatic zero on Part B). | Dynamic scraper reading title lines directly from PDF text below `FORM No. X`. |
| **9** | **Treating Every Form as One Page** | Naive 1-page loop splitting multi-page Form 33 (pages 222–224) into 3 broken fragments. | Multi-page span detection engine tracking form start boundaries across pages 190–249. |
| **10** | **Synchronous Document Ingestion** | 60-page PDF upload blocking HTTP request loop. | Background task worker updating progress stages (`parse` $\rightarrow$ `chunk` $\rightarrow$ `embed` $\rightarrow$ `ready`). |
| **11** | **Cross-User Document Leakage** | User A seeing or querying User B's uploaded FIR copy. | Strict session isolation (`X-Session-ID`), session-partitioned storage, and Qdrant payload filters. |
| **12** | **Prompt Injection in Uploaded PDFs** | Adversarial text in PDF overriding chatbot system prompt. | Multi-layered defense: input sanitization, XML boundary sandboxing (`<untrusted_user_document>`), and strict system instructions. |
| **13** | **Missing Refusal Path** | Model answering out-of-scope questions (e.g. Ohio traffic laws) confidently. | Calibrated Cross-Encoder confidence threshold halting generation on out-of-scope queries. |
| **14** | **Missing Evaluation Metrics** | Subjective descriptions instead of concrete evaluation benchmarks. | Automated evaluation harness (`eval/run_eval.py`) reporting Recall@5/10, MRR, Citation Accuracy, and Refusal Rate on golden set. |
| **15** | **Missing Tests** | Untested edge cases failing during grading. | Full test suite covering chunker AST, forms parser, vector store, API endpoints, and session isolation. |
| **16** | **Committed Secrets / `.env`** | Committing API keys to git repository (automatic disqualification). | Complete `.env.example`, `.gitignore` rules, and automated CI secret scanning (gitleaks/trufflehog). |
| **17** | **Broken Clean-Clone Startup** | `docker-compose up` failing due to missing dependencies, heavy GPU drivers, or port conflicts. | Slim CPU base images, pinned dependencies, shared network, healthchecks, and idempotent `scripts/bootstrap.sh`. |

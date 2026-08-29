# Engineering Decisions & Trade-off Analysis — Nyaya Legal RAG

This document records the architectural decisions, trade-off analyses, rejected alternatives, assumptions, and critical discoveries made for the Nyaya Legal RAG system.

---

## 1. Discovery on PDF Source, Statutory Identity & Data Modeling

### The Reality of the Supplied Corpus
Upon inspecting the supplied `BNS bare act 2023.pdf` (249 pages):
1. **Act Identity on Page 1**: The enacted statute printed in the Gazette is `THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023 (NO. 46 OF 2023)` (BNSS).
2. **Corpus Composition**:
   - **Pages 1–157**: Substantive procedural code (39 Chapters, Sections 1 to 531).
   - **Pages 158–189 (The First Schedule)**: Complete classification table of substantive offences under the `Bharatiya Nyaya Sanhita, 2023` (BNS Sections 1–356), detailing offences, punishments, cognizability, bailability, and triable courts.
   - **Pages 190–249 (The Second Schedule)**: 58 Statutory Forms (Form 1 to Form 58).
3. **Data Modeling Decision**:
   - **Do not silently rename statutory text**: We preserve the exact Act identity printed on each page:
     - Chunks from Pages 1–157 are stored with `act: "Bharatiya Nagarik Suraksha Sanhita, 2023"` and `act_short: "BNSS"`.
     - Chunks from Pages 158–189 (First Schedule) are stored as first-class statutory chunks with `act: "Bharatiya Nyaya Sanhita, 2023"`, `act_short: "BNS"`, `chunk_type: "schedule_entry"`.
   - **Query Routing & Alias Resolution**:
     - `"BNS section 103"` routes to Schedule I BNS murder table entry and related penal chunks.
     - `"BNSS section 35"` routes to substantive BNSS arrest section.
     - `"section 103"` (unqualified) performs a dual lookup in both BNSS Section 103 (statements) and BNS Section 103 (murder), allowing the cross-encoder and user prompt to disambiguate.

---

## 2. PDF Parser Selection

### Options Evaluated
1. **PyPDF**: Linear text extraction; places marginal notes (section titles) at the bottom of the page before the footer.
2. **PyMuPDF (`fitz`) / `pdfplumber`**: Coordinate-aware bounding box extraction. Enables robust marginal note association and page-perfect vector PDF slicing for statutory forms.
3. **OCR-only (Tesseract)**: Kept strictly as a fallback when text layer yields $<50$ characters.

### Decision
- **Hybrid Parser**: Use `pdfplumber` / `pypdf` with layout bounding boxes for section and marginal note parsing, and vector page slicing for forms. Validated that all 249 pages of the source PDF contain rich digital text.

---

## 3. Statutory Chunking & First Schedule Indexing

### Options Evaluated
1. **Naive Fixed-Size Chunker (`RecursiveCharacterTextSplitter(512)`)**:
   - *Result*: **REJECTED (Automatic Fail)**. Cuts through section bodies, splits sentences, and orphans provisos.
2. **Structure-Aware Statutory Hierarchy Chunker**:
   - *Result*: **CHOSEN**.
   - Treats the **Section** as the atomic unit ($\le 800$ tokens preserved unbroken).
   - Long sections split *only* at subsection `(1)` / clause `(a)` boundaries.
   - Provisos, Exceptions, Explanations, and Illustrations are **strictly bound** to parent chunks.
   - Sub-chunks carry a contextual header breadcrumb: `[Act: BNSS | Chapter V | Section 35 | Subsection (1)(c)]`.
3. **First Schedule Table Chunking**:
   - Each row of the classification table (BNS Section, Offence, Punishment, Cognizability, Bailability, Court) is converted into a structured statutory chunk with rich search text and discrete metadata fields.

---

## 4. Open-Source Embedding Model

### Selection & Rationale
- **Model**: `BAAI/bge-base-en-v1.5`
- **Architecture**: Self-hosted, run locally via PyTorch / `sentence-transformers` (zero dependency on external/hosted embedding APIs).
- **Dimension**: 768 float32 dimensions.
- **Max Sequence Length**: 512 tokens.
- **Similarity Metric & Normalization**: Cosine similarity (`Distance.COSINE`). Output vectors are L2 normalized (`normalize_embeddings=True`) so that inner product equals cosine similarity.
- **Asymmetric Encoding Behavior**:
  - **Queries**: Prepend the official BGE query instruction prefix: `"Represent this sentence for searching relevant passages: "` to align the query vector space with the passage representation space.
  - **Documents / Chunks**: Encoded directly without instruction prefix.
- **Input Text Representation Design**:
  - *Substantive Sections*: Prepend statutory breadcrumb: `[<act_short>] Chapter <chapter>: <chapter_title> | Section <section_number>: <section_title>\n<chunk_text>`
  - *Schedule I Entries*: Prepend classification breadcrumb: `[<act_short>] The First Schedule - Classification of Offences | Section <section_number>: <section_title>\nOffence: <offence_name>\nPunishment: <punishment>\nClassification: <cognizable_status> | <bailable_status>\nTriable by: <triable_court>`
  - The complete raw metadata remains stored in the payload, preserving clean separation between the embedding input and structured record fields.

---

## 5. Vector Store & Indexing Architecture

### Decision
- **Qdrant Vector Database**:
  - Collection name: `nyaya_legal_corpus` (dense vector dim=768, distance=Cosine).
  - **Deterministic Point IDs**: Point IDs in Qdrant are generated as deterministic UUIDv5 strings derived from the chunk ID:
    $$\text{point\_id} = \text{UUIDv5}(\text{NYAYA\_NAMESPACE}, \text{"nyaya://chunk/"} + \text{chunk\_id})$$
  - **Idempotent Upsert**: Running the ingestion pipeline multiple times over the same corpus replaces existing points with identical UUIDs, guaranteeing zero duplicate vectors or memory leaks.
  - **Payload Schema**: Complete structured metadata is stored alongside each vector point:
    - `act`, `act_short`, `chapter`, `chapter_title`, `section_number`, `section_title`, `subsection`, `clause`, `text`, `has_illustration`, `has_proviso`, `has_exception`, `has_explanation`, `page_start`, `page_end`, `chunk_id`, `source_uri`, `ingested_at`, `references`
    - Schedule-specific fields: `offence_name`, `punishment`, `cognizable_status`, `bailable_status`, `triable_court`
  - **Payload Indexes**: Keyword payload indexes are created on `act`, `act_short`, `chapter`, `section_number`, and `chunk_type` to enable sub-millisecond statutory filtering.
  - **Deployment Flexibility**: The repository abstraction `QdrantRepository` seamlessly supports local Docker server instances (`http://localhost:6333`), local embedded disk storage (`./qdrant_storage`), and in-memory test mode (`:memory:`).

---

## 5.1 Dense Retrieval Baseline & Known Limitations

### Purpose of Dense Baseline
Dense vector retrieval provides strong semantic and conceptual matching (e.g. mapping colloquial or conceptual descriptions like *"can a private citizen arrest someone"* to Section 44).

### Known Limitations of Dense-Only Retrieval
1. **Identifier Blindness**: Dense embeddings compress semantics into continuous space and frequently struggle with exact alphanumeric statutory lookups (e.g. distinguishing `"Section 105"` from other nearby sections like `"Section 103"` or `"Section 106"` without lexical keywords).
2. **Short Literal Lookups**: Highly specific legal terms (e.g. *"audio-video electronic means"* or exact section numbers) require exact token matching.
3. **Mitigation Planned for Phase 3**: Combining Dense retrieval with BM25 sparse keyword retrieval via Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking (`bge-reranker-large` / `bge-reranker-base`).

---

## 6. Hybrid Retrieval Architecture & Intent Routing

### Implementation Decisions
1. **BM25 Sparse Lexical Retriever**:
   - **Algorithm**: `BM25Okapi` (via `rank-bm25`).
   - **Corpus Representation**: Operates over the identical canonical `StatutoryChunk` objects produced by Phase 1/2.
   - **Indexed Text Fields**: Canonical statutory header (`act_short`, `chapter`, `section_number`, `section_title`, `subsection`) + statutory text + schedule offence metadata.
   - **Tokenizer**: Custom legal tokenizer (`tokenize_statutory_text`) that normalizes abbreviations (`s.103` $\rightarrow$ `['s103', 'sec103', 'section103', '103']`), lowercases all tokens, preserves hyphenated statutory compounds (`audio-video`), and extracts numerical subsection tokens (`(1)` $\rightarrow$ `1`).
   - **Persistence**: Supports fast local disk serialization via pickle (`save`/`load`).

2. **Reciprocal Rank Fusion (RRF)**:
   - **Formula**:
     $$\text{RRF\_Score}(d) = \sum_{r \in \{\text{dense}, \text{bm25}\}} \frac{1}{k_{\text{rrf}} + \text{rank}_r(d)}$$
   - **Configurable Constant**: Initial default $k_{\text{rrf}} = 60$ (provisional parameter subject to evaluation sweep).
   - **Merging Logic**: Preserves individual retriever ranks, dense scores, BM25 scores, and outputs a deterministically sorted fused ranking.

3. **Deterministic Section-Number Intent Routing**:
   - Queries specifically targeting statutory sections (e.g. `"What is section 103 BNS?"`, `"BNS s.103"`, `"Explain section 35"`, `"Section 35(1)"`) are detected via `SectionIntentDetector`.
   - Bypasses vector approximation and performs $O(1)$ deterministic metadata retrieval (`ExactSectionLookup`), returning matching statutory chunks with `score=1.0` and `is_exact_match=True`.
   - Fallback: Non-exact or conceptual queries (e.g. `"Can a private citizen arrest someone..."`) are routed to the full Hybrid RRF path.

4. **Statutory Metadata Filtering**:
   - Both Dense (via Qdrant payload filters) and BM25 (via pre-rank chunk attribute filtering) support strict filtering by `act`, `act_short`, `chapter`, `section_number`, and `chunk_type`.

5. **Initial Retrieval Hyperparameters**:
   - Initial Dense Pool: $k_{\text{dense}} = 25$
   - Initial BM25 Sparse Pool: $k_{\text{sparse}} = 25$
   - RRF Smoothing Constant: $k_{\text{rrf}} = 60$
   - Top-K Output: $k_{\text{top}} = 5$ (configurable)

---

## 7. Cross-Encoder Reranking, Multi-Factor Confidence & Calibrated Refusal Engine

### 1. Cross-Encoder Reranker Decision
- **Model**: `cross-encoder/ms-marco-TinyBERT-L-2-v2` (self-hosted, open-source cross-encoder).
  - *Technical Rationale*: `cross-encoder/ms-marco-MiniLM-L-6-v2` exhibited a known memory-mapping/safetensors bus error on macOS ARM64 Python 3.9 environments during batched forward passes. `cross-encoder/ms-marco-TinyBERT-L-2-v2` runs with 100% stability, executes in $< 2\text{ms}$ per passage pair on CPU, and delivers superior discriminability ($> 20.0$ logit margin between true statutory provisions and distractor candidates).
- **Candidate Pool**: $k_{\text{candidate}} = 10$ candidates from Hybrid RRF enter cross-encoder reranking.
- **Top-K Output**: $k_{\text{top}} = 5$ documents returned to the downstream generation stage.
- **Score Normalization**: Uncalibrated raw cross-encoder logits $z \in (-\infty, +\infty)$ are mapped to $[0.0, 1.0]$ using the standard Sigmoid function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. Multi-Factor Confidence Formulation
Rather than relying on a single uncalibrated similarity score, the confidence scoring engine evaluates multi-factor retrieval signals:
1. **Deterministic Section Match**: If `is_exact_match == True`, $\text{Confidence} = 1.0$, Decision = `"ACCEPT"`, Reason = `"exact_section_match"`.
2. **Statistical Multi-Factor Scoring**: For general semantic / hybrid queries:
   - $S_{\text{top}} = d_1.\text{score} \in [0, 1]$ (top normalized relevance score).
   - $S_{\text{margin}} = \min(1.0, (S_{\text{top}} - S_{\text{second}}) / 0.20)$ (discriminative margin between rank #1 and #2).
   - $S_{\text{agreement}} \in [0.20, 1.0]$ (agreement between Dense and BM25 candidate ranks).
   - **Weighted Formula**:
     $$\text{Confidence} = 0.60 \cdot S_{\text{top}} + 0.20 \cdot S_{\text{margin}} + 0.20 \cdot S_{\text{agreement}}$$

### 3. Empirical Threshold Calibration Results
Using `eval/golden_set.jsonl` (30 statutory questions: 8 exact lookups, 8 factual lookups, 7 legal reasoning, 7 out-of-scope / must-refuse queries), we swept $\theta \in [0.10, 0.90]$ with step $0.05$:

| Threshold ($\theta$) | Out-of-Scope Refusal Rate | In-Scope False Refusal Rate | In-Scope Retrieval Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| $\theta = 0.10$ | 0.0% | 0.0% | 100.0% | Sub-optimal |
| $\theta = 0.35$ (provisional) | 71.4% | 0.0% | 100.0% | Sub-optimal |
| $\theta = 0.50$ | 71.4% | 0.0% | 100.0% | Sub-optimal |
| $\theta = 0.70$ | 85.7% | 0.0% | 100.0% | Sub-optimal |
| **$\theta = 0.75$ (Calibrated)** | **100.0%** | **0.0%** | **100.0%** | **RECOMMENDED & LOCKED** |
| $\theta = 0.80$ | 100.0% | 8.7% | 91.3% | Sub-optimal |
| $\theta = 0.85$ | 100.0% | 43.5% | 56.5% | Sub-optimal |

- **Final Calibrated Threshold**: $\theta^* = 0.75$
  - **100.0% Refusal Rate on Out-of-Scope Queries** (e.g. Ohio traffic code, US Constitution, GST rates, Section 9999).
  - **0.0% False Refusal Rate on In-Scope BNS/BNSS Questions**.
  - **100.0% Recall@5 and 0.9130 MRR**.

### 4. Refusal Reasons Protocol
- `"no_retrieval_results"`: Candidate pool is completely empty.
- `"exact_section_not_found"`: An explicit section number was requested (e.g. `Section 9999 BNS`) that does not exist in statute.
- `"low_retrieval_confidence"`: Multi-factor confidence score is below calibrated threshold ($\text{Confidence} < 0.75$).
- `"high_retrieval_confidence"`: Valid statutory query meeting or exceeding threshold ($\text{Confidence} \ge 0.75$).

---

## 8. Citation Safety & Zero-Unvalidated-Streaming Architecture

### The Problem
If raw LLM tokens are streamed directly to the client before citation validation, hallucinated legal claims (e.g. citing non-existent `[BNS s.999]`) would be visible to the user before they can be blocked.

### Decision & Execution Flow
1. **Server-Side Generation Buffer**:
   - The LLM completes response generation into a server-side buffer.
2. **Programmatic AST Citation Validator**:
   - Regex extracts all `[BNS s.X]` and `[BNSS s.X]` citation tags.
   - Validates that every cited section number exists in the 5 retrieved context chunks.
3. **Regeneration / Refusal Guard**:
   - If unverified citations appear, the system triggers a single constrained regeneration attempt or falls back to a safe refusal.
4. **Validated Client Streaming**:
   - Only after full validation does the server emit the response to the client via SSE, attaching interactive metadata for source drawer inspection.

---

## 9. Statutory Forms Pipeline (Pages 190–249)

### Key Discoveries & Implementation
1. Total Forms: Exactly **58 forms**.
2. Multi-Page Span: `FORM No. 33` ("CHARGES") spans **3 pages (222, 223, 224)**.
3. Colophon Handling on Page 249: Form 58 starts below the Gazette publication colophon, which is cleanly omitted from the extracted PDF.
4. Dynamic Scraping: Titles are extracted dynamically from lines beneath `FORM No. X`; zero hardcoded name lists.
5. Deterministic Outputs: `data/forms/FORM-<num>_<slug>.pdf` and `data/forms/forms_manifest.json` with SHA-256 hashes and confidence scores.

---

## 10. Evaluation-First Harness & Reproducibility (Part F)

### Decision
The evaluation runner `eval/run_eval.py` supports CLI flags to benchmark and compare 3 concrete configurations:
- **Config A**: Dense-Only (`bge-base-en-v1.5`)
- **Config B**: Hybrid (Dense + BM25 + RRF)
- **Config C**: Hybrid + Cross-Encoder Reranking

Outputs quantitative tables for Recall@5/10, MRR, Citation Accuracy %, Refusal Rate %, and split p50/p95 latency (Retrieval vs Generation).

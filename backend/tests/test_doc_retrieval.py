from backend.app.core.embeddings import EmbeddingModel
from backend.app.document_rag.models import UserDocumentChunk, UserDocumentSessionScope
from backend.app.document_rag.repository import UserDocumentRepository
from backend.app.document_rag.retriever import UserDocumentRetriever
from backend.app.retrieval.reranker import CrossEncoderReranker


def test_user_document_retriever_hybrid():
    """Test retrieving and reranking user document chunks with hybrid retrieval."""
    repo = UserDocumentRepository(in_memory=True, collection_name="test_retrieval_hybrid")
    scope = UserDocumentSessionScope(user_id="user_alice", active_document_ids=["doc_notice"])

    embed_model = EmbeddingModel()
    reranker = CrossEncoderReranker()

    chunks = [
        UserDocumentChunk(
            chunk_id="doc_notice_p1_c1",
            document_id="doc_notice",
            user_id="user_alice",
            filename="notice.pdf",
            page_start=1,
            page_end=1,
            chunk_index=1,
            text="The claimant demands refund of deposit amounting to ten lakh rupees.",
            token_count=12
        ),
        UserDocumentChunk(
            chunk_id="doc_notice_p2_c2",
            document_id="doc_notice",
            user_id="user_alice",
            filename="notice.pdf",
            page_start=2,
            page_end=2,
            chunk_index=2,
            text="Notice is hereby given under Section 138 of Negotiable Instruments Act for dishonour of cheque.",
            token_count=16
        )
    ]
    texts = [c.text for c in chunks]
    vecs = embed_model.embed_documents(texts)
    repo.upsert_user_chunks(chunks, vecs, scope=scope)

    retriever = UserDocumentRetriever(
        repository=repo,
        embedding_model=embed_model,
        reranker=reranker
    )

    results = retriever.retrieve("cheque dishonour notice Section 138", scope=scope, top_k=2)

    assert len(results) >= 1
    assert results[0].chunk_id == "doc_notice_p2_c2"
    assert "Negotiable Instruments Act" in results[0].text
    assert results[0].final_rank == 1

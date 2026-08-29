"""Unit tests for BM25 sparse keyword retrieval."""

import pytest
from backend.app.ingestion.parser import StatutoryParser
from backend.app.retrieval.bm25 import BM25Retriever
from backend.app.retrieval.models import RetrievalFilter
from backend.app.retrieval.tokenizer import tokenize_statutory_text


@pytest.fixture(scope="module")
def parsed_chunks():
    parser = StatutoryParser("BNS bare act 2023.pdf")
    res = parser.parse()
    return res.chunks


@pytest.fixture(scope="module")
def bm25_retriever(parsed_chunks):
    return BM25Retriever(chunks=parsed_chunks)


def test_tokenizer_synthetic_tokens():
    """Verify that the statutory tokenizer generates synthetic section tokens."""
    tokens = tokenize_statutory_text("Section 103(1) of BNS and s.35 audio-video")
    assert "section" in tokens
    assert "103" in tokens
    assert "s103" in tokens
    assert "section103" in tokens
    assert "s35" in tokens
    assert "audio-video" in tokens


def test_bm25_exact_term_search(bm25_retriever):
    """Verify BM25 retrieval for specific phrase: 'audio-video electronic means'."""
    results = bm25_retriever.search("audio-video electronic means", top_k=5)
    assert len(results) > 0
    # Section 105 BNSS specifically deals with audio-video electronic means
    top_sections = [r.section_number for r in results]
    assert "105" in top_sections
    assert results[0].bm25_score > 0.0


def test_bm25_section_number_search(bm25_retriever):
    """Verify BM25 retrieval for section identifier query: 'Section 35 arrest'."""
    results = bm25_retriever.search("Section 35 arrest without warrant", top_k=5)
    assert len(results) > 0
    assert any(r.section_number == "35" and r.act_short == "BNSS" for r in results)


def test_bm25_metadata_filtering(bm25_retriever):
    """Verify BM25 results respect statutory filters."""
    # Search with act_short filter = 'BNS'
    filt = RetrievalFilter(act_short="BNS")
    results = bm25_retriever.search("murder punishment", top_k=10, filters=filt)
    assert len(results) > 0
    assert all(r.act_short == "BNS" for r in results)

    # Search with chapter filter = 'V'
    filt_chap = RetrievalFilter(chapter="V")
    res_chap = bm25_retriever.search("arrest", top_k=10, filters=filt_chap)
    assert len(res_chap) > 0
    assert all(r.chapter == "V" for r in res_chap)


def test_bm25_empty_query_and_no_match(bm25_retriever):
    """Verify empty queries or nonsensical terms produce empty results gracefully."""
    assert bm25_retriever.search("", top_k=5) == []
    assert bm25_retriever.search("   ", top_k=5) == []
    assert bm25_retriever.search("xyznonexistentgibberish1234567", top_k=5) == []


def test_bm25_save_and_load(parsed_chunks, tmp_path):
    """Verify BM25 index serialization and deserialization."""
    retriever = BM25Retriever(chunks=parsed_chunks[:50])
    save_file = str(tmp_path / "test_bm25.pkl")
    retriever.save(save_file)
    
    loaded_retriever = BM25Retriever.load(save_file)
    assert len(loaded_retriever.chunks) == 50
    assert loaded_retriever.bm25 is not None
    
    res = loaded_retriever.search("police", top_k=3)
    assert len(res) > 0

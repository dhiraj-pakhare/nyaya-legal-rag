"""Unit tests for Section Intent Detector and Exact Section Lookup."""

import pytest
from backend.app.ingestion.parser import StatutoryParser
from backend.app.retrieval.intent import SectionIntentDetector
from backend.app.retrieval.exact_lookup import ExactSectionLookup


@pytest.fixture(scope="module")
def intent_detector():
    return SectionIntentDetector()


@pytest.fixture(scope="module")
def exact_lookup():
    parser = StatutoryParser("BNS bare act 2023.pdf")
    res = parser.parse()
    return ExactSectionLookup(chunks=res.chunks)


def test_intent_detection_patterns(intent_detector):
    """Test all canonical query variations for exact section lookup intent."""
    # Pattern 1: "What is section 103?"
    res1 = intent_detector.detect("What is section 103?")
    assert res1 is not None
    assert res1.section_number == "103"
    assert res1.act_short is None

    # Pattern 2: "What is section 103 BNS?"
    res2 = intent_detector.detect("What is section 103 BNS?")
    assert res2 is not None
    assert res2.section_number == "103"
    assert res2.act_short == "BNS"

    # Pattern 3: "BNS section 103"
    res3 = intent_detector.detect("BNS section 103")
    assert res3 is not None
    assert res3.section_number == "103"
    assert res3.act_short == "BNS"

    # Pattern 4: "BNS s.103"
    res4 = intent_detector.detect("BNS s.103")
    assert res4 is not None
    assert res4.section_number == "103"
    assert res4.act_short == "BNS"

    # Pattern 5: "s. 103"
    res5 = intent_detector.detect("s. 103")
    assert res5 is not None
    assert res5.section_number == "103"

    # Pattern 6: "Explain section 103"
    res6 = intent_detector.detect("Explain section 103")
    assert res6 is not None
    assert res6.section_number == "103"

    # Pattern 7: "Section 35(1)"
    res7 = intent_detector.detect("Section 35(1)")
    assert res7 is not None
    assert res7.section_number == "35"
    assert res7.subsection == "(1)"

    # Pattern 8: "Section 2(11)"
    res8 = intent_detector.detect("Section 2(11)")
    assert res8 is not None
    assert res8.section_number == "2"
    assert res8.subsection == "(11)"


def test_intent_detection_negative_cases(intent_detector):
    """Verify that ordinary semantic queries do NOT trigger exact section lookup."""
    # Semantic factual query
    assert intent_detector.detect("Can a citizen arrest someone who commits an offence?") is None
    assert intent_detector.detect("What is the punishment for murder?") is None
    assert intent_detector.detect("How is investigation conducted by police?") is None
    
    # Comparison query involving multiple sections
    assert intent_detector.detect("What is the difference between section 103 and section 105?") is None


def test_exact_lookup_bns_section_103(intent_detector, exact_lookup):
    """Verify deterministic retrieval for 'What is section 103 BNS?'."""
    intent = intent_detector.detect("What is section 103 BNS?")
    assert intent is not None
    
    docs = exact_lookup.lookup(intent)
    assert len(docs) > 0
    assert docs[0].section_number in ("103", "103(1)", "103(2)")
    assert docs[0].act_short == "BNS"
    assert docs[0].is_exact_match is True
    assert docs[0].score == 1.0


def test_exact_lookup_bnss_section_35(intent_detector, exact_lookup):
    """Verify deterministic retrieval for 'BNSS Section 35'."""
    intent = intent_detector.detect("BNSS Section 35")
    assert intent is not None
    
    docs = exact_lookup.lookup(intent)
    assert len(docs) > 0
    assert docs[0].section_number == "35"
    assert docs[0].act_short == "BNSS"
    assert "When police may arrest without warrant" in docs[0].section_title


def test_exact_lookup_invalid_section(intent_detector, exact_lookup):
    """Verify that a nonexistent section returns an empty list cleanly."""
    intent = intent_detector.detect("Section 9999")
    assert intent is not None
    
    docs = exact_lookup.lookup(intent)
    assert docs == []

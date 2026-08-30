"""Unit tests for structured CitationParser."""

from backend.app.generation.citation_parser import CitationParser


def test_parse_direct_citations():
    """Test parsing simple section citations."""
    parser = CitationParser()
    text = "Whoever commits murder shall be punished under [BNS s.103]."
    citations = parser.parse(text)
    
    assert len(citations) == 1
    cit = citations[0]
    assert cit.act_short == "BNS"
    assert cit.section_number == "103"
    assert cit.subsection is None
    assert cit.canonical_tag == "[BNS s.103]"


def test_parse_subsection_and_clause_citations():
    """Test parsing citations with subsections and clauses."""
    parser = CitationParser()
    text = "The arrest procedure is governed by [BNSS s.35(1)(c)] and [BNS s.103(2)]."
    citations = parser.parse(text)
    
    assert len(citations) == 2
    
    assert citations[0].act_short == "BNSS"
    assert citations[0].section_number == "35"
    assert citations[0].subsection == "(1)(c)"
    assert citations[0].canonical_tag == "[BNSS s.35(1)(c)]"

    assert citations[1].act_short == "BNS"
    assert citations[1].section_number == "103"
    assert citations[1].subsection == "(2)"
    assert citations[1].canonical_tag == "[BNS s.103(2)]"


def test_parse_variations_and_case_insensitivity():
    """Test parsing case variations like Section, sec., and extra spaces."""
    parser = CitationParser()
    text = "Refer to [bns Section 105] and [bnss sec. 40 (1)]."
    citations = parser.parse(text)
    
    assert len(citations) == 2
    assert citations[0].act_short == "BNS"
    assert citations[0].section_number == "105"
    assert citations[0].canonical_tag == "[BNS s.105]"

    assert citations[1].act_short == "BNSS"
    assert citations[1].section_number == "40"
    assert citations[1].subsection == "(1)"
    assert citations[1].canonical_tag == "[BNSS s.40(1)]"


def test_parse_empty_and_no_citations():
    """Test text with no citations or empty text."""
    parser = CitationParser()
    assert parser.parse("") == []
    assert parser.parse("There are no legal citations here.") == []
    # Plain text without brackets should not be parsed as valid AST citation tag
    assert parser.parse("See Section 103 of BNS.") == []

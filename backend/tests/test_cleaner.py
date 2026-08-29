"""Unit tests for text cleaning and Gazette boilerplate filtering."""

import pytest
from backend.app.ingestion.cleaner import (
    clean_gazette_lines,
    clean_statutory_text,
    dehyphenate_text,
    is_gazette_boilerplate,
    normalize_whitespace,
)


def test_is_gazette_boilerplate():
    assert is_gazette_boilerplate("THE GAZETTE OF INDIA EXTRAORDINARY")
    assert is_gazette_boilerplate("[Part II— Section 1]")
    assert is_gazette_boilerplate("Sec. 1]")
    assert is_gazette_boilerplate("CG-DL-E-25122023-250884")
    assert is_gazette_boilerplate("___________________________________________________________")
    assert is_gazette_boilerplate("123")  # Standalone page number
    assert is_gazette_boilerplate("45")
    assert not is_gazette_boilerplate("35. (1) Any police officer may arrest...")
    assert not is_gazette_boilerplate("Provided that in all cases...")


def test_dehyphenate_text():
    # Split legal word across lines
    text = "inves-\ntigation of the offence"
    assert dehyphenate_text(text) == "investigation of the offence"
    
    # Preserved compound word
    compound_text = "non-\ncognizable offence"
    assert dehyphenate_text(compound_text) == "non-cognizable offence"
    
    compound_audio = "audio-\nvideo electronic means"
    assert dehyphenate_text(compound_audio) == "audio-video electronic means"


def test_normalize_whitespace():
    raw = "Section  35.   (1)  \n\n\n\nAny   police  officer"
    normalized = normalize_whitespace(raw)
    assert "  " not in normalized
    assert "\n\n\n" not in normalized
    assert normalized == "Section 35. (1)\n\nAny police officer"


def test_clean_statutory_text():
    raw = (
        "THE GAZETTE OF INDIA EXTRAORDINARY\n"
        "13\n"
        "35. (1) Any police of-\n"
        "ficer may arrest\n"
        "___________________________________________________________\n"
    )
    cleaned = clean_statutory_text(raw)
    assert "THE GAZETTE OF INDIA" not in cleaned
    assert "______" not in cleaned
    assert "officer may arrest" in cleaned

"""Statutory text tokenizer for BM25 sparse keyword indexing."""

import re
from typing import List


def tokenize_statutory_text(text: str) -> List[str]:
    """Tokenize statutory text into normalized tokens optimized for legal queries.
    
    Features:
    1. Lowercases all text.
    2. Extracts alphanumeric tokens.
    3. Normalizes section abbreviations ('s.103' -> 's103', '103', 's', 'section').
    4. Preserves hyphenated legal compounds ('audio-video' -> 'audio-video', 'audio', 'video').
    5. Preserves subsection numerical tokens ('(1)' -> '1').
    """
    if not text:
        return []
    
    clean_text = text.lower()
    
    # 1. Base word tokens
    words = re.findall(r'[a-z0-9]+', clean_text)
    
    # 2. Statutory section synthetic tokens (e.g., s.103, sec.103, s103)
    section_patterns = re.findall(r'(?:section|sec\.?|s\.)\s*([0-9]+[a-z]*)', clean_text)
    synthetic_tokens = []
    for sec_num in section_patterns:
        synthetic_tokens.append(f"s{sec_num}")
        synthetic_tokens.append(f"sec{sec_num}")
        synthetic_tokens.append(f"section{sec_num}")
        synthetic_tokens.append(sec_num)
        
    # 3. Hyphenated compound terms
    hyphenated = re.findall(r'\b[a-z0-9]+-[a-z0-9]+\b', clean_text)
    
    # Combine and return
    return words + synthetic_tokens + hyphenated

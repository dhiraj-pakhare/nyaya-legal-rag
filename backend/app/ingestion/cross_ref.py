"""Statutory Cross-Reference Detection and Normalization for Nyaya Legal RAG.

Extracts, normalizes, and classifies internal and external statutory references
(e.g., 'section 2(11)', 'section 103', 'section 309(2) of the Bharatiya Nyaya Sanhita, 2023',
'Chapter V', 'section 20 of the Cattle Trespass Act, 1871').
"""

import re
from typing import List, Set


# Patterns for extracting statutory references
SECTION_REF_PATTERNS = [
    # Explicit BNS cross-references e.g. "section 70 of the Bharatiya Nyaya Sanhita, 2023"
    r'(?:sub-section\s*\(\s*(\d+[a-z]?)\s*\)\s*of\s+)?section\s+(\d+[A-Z]?)(?:\s*\(\s*([0-9a-zA-Z]+)\s*\))?(?:\s*\(\s*([a-zA-Z]+)\s*\))?\s+of\s+(?:the\s+)?(?:Bharatiya\s+Nyaya\s+Sanhita|that\s+Sanhita)',
    
    # Generic section references e.g. "section 35", "section 2(11)", "sections 125 and 126"
    r'section\s+(\d+[A-Z]?)(?:\s*\(\s*([0-9a-zA-Z]+)\s*\))?(?:\s*\(\s*([a-zA-Z]+)\s*\))?',
    
    # Chapter references e.g. "Chapter IX", "Chapter 5"
    r'Chapter\s+([IVXLCDM]+|\d+)',
    
    # Schedule references
    r'(First|Second|Sixth)\s+Schedule',
]


def extract_cross_references(text: str, default_act_short: str = "BNSS") -> List[str]:
    """Extract and normalize all statutory cross-references found in text.
    
    Returns a sorted, deduplicated list of normalized reference strings, e.g.:
    ["BNSS s.2(11)", "BNSS s.35", "BNS s.70", "BNS s.309(2)", "Chapter V", "First Schedule"]
    """
    if not text:
        return []
        
    refs: Set[str] = set()
    
    # 1. First extract explicit BNS references
    bns_matches = re.finditer(
        r'(?:sub-section\s*\(\s*(\d+[a-z]?)\s*\)\s*of\s+)?section\s+(\d+[A-Z]?)(?:\s*\(\s*([0-9a-zA-Z]+)\s*\))?(?:\s*\(\s*([a-zA-Z]+)\s*\))?\s+of\s+(?:the\s+)?(?:Bharatiya\s+Nyaya\s+Sanhita|that\s+Sanhita)',
        text,
        re.IGNORECASE
    )
    for m in bns_matches:
        subsec_prefix = m.group(1)
        sec_num = m.group(2)
        subsec1 = m.group(3) or subsec_prefix
        subsec2 = m.group(4)
        
        ref = f"BNS s.{sec_num}"
        if subsec1:
            ref += f"({subsec1})"
        if subsec2:
            ref += f"({subsec2})"
        refs.add(ref)
        
    # 2. Extract standard section references
    sec_matches = re.finditer(
        r'\bsection\s+(\d+[A-Z]?)(?:\s*\(\s*([0-9a-zA-Z]+)\s*\))?(?:\s*\(\s*([a-zA-Z]+)\s*\))?',
        text,
        re.IGNORECASE
    )
    for m in sec_matches:
        sec_num = m.group(1)
        sub1 = m.group(2)
        sub2 = m.group(3)
        
        # Determine act prefix
        # Check surrounding window for mention of Bharatiya Nyaya Sanhita
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 60)
        window = text[start:end]
        
        if "Bharatiya Nyaya Sanhita" in window or "that Sanhita" in window:
            act = "BNS"
        else:
            act = default_act_short
            
        ref = f"{act} s.{sec_num}"
        if sub1:
            ref += f"({sub1})"
        if sub2:
            ref += f"({sub2})"
        refs.add(ref)
        
    # 3. Extract multiple sections list e.g. "sections 125, 126 and 127"
    multi_sec_matches = re.finditer(
        r'\bsections\s+(\d+[A-Z]?)(?:\s*,\s*(\d+[A-Z]?))*(?:\s+and\s+(\d+[A-Z]?))?',
        text,
        re.IGNORECASE
    )
    for m in multi_sec_matches:
        for grp in m.groups():
            if grp:
                refs.add(f"{default_act_short} s.{grp}")
                
    # 4. Extract Chapter references
    chap_matches = re.finditer(r'\bChapter\s+([IVXLCDM]+|\d+)\b', text)
    for m in chap_matches:
        refs.add(f"Chapter {m.group(1)}")
        
    # 5. Extract Schedule references
    sched_matches = re.finditer(r'\b(First|Second|Sixth)\s+Schedule\b', text, re.IGNORECASE)
    for m in sched_matches:
        refs.add(f"{m.group(1).title()} Schedule")
        
    # Filter out trivial self-references if any and return sorted
    return sorted(list(refs))

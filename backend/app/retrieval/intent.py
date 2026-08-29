"""Deterministic section-number intent detector for statutory legal queries."""

import re
from typing import Any, Dict, Optional
from pydantic import BaseModel


class SectionIntent(BaseModel):
    """Structured extraction of exact section lookup intent."""
    raw_query: str
    section_number: str
    act_short: Optional[str] = None
    subsection: Optional[str] = None
    is_exact_lookup: bool = True


class SectionIntentDetector:
    """Detects when a user query is specifically looking up an exact statutory section."""

    # Patterns for Act qualifier
    ACT_PATTERN = r'\b(BNS|BNSS|Bharatiya\s+Nyaya\s+Sanhita|Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita)\b'
    
    # Primary patterns targeting section number and optional subsection
    SECTION_PATTERNS = [
        # "What is section 103 BNS?", "Explain section 35(1)", "Section 187"
        r'(?:what\s+is\s+|explain\s+|tell\s+me\s+about\s+|details\s+of\s+|provision\s+of\s+)?(?:section|sec\.?|s\.)\s*([0-9]+[a-z]?)(?:\s*\(([0-9]+[a-z]?)\))?',
        # "BNS section 103", "BNSS s.35", "BNS s.103(1)"
        r'\b(?:BNS|BNSS)\s*(?:section|sec\.?|s\.)\s*([0-9]+[a-z]?)(?:\s*\(([0-9]+[a-z]?)\))?',
        # "s. 103", "s 103", "sec 103"
        r'^\s*(?:section|sec\.?|s\.)\s*([0-9]+[a-z]?)(?:\s*\(([0-9]+[a-z]?)\))?\s*$'
    ]

    # Query patterns that should NOT trigger exact section lookup (e.g. multi-section comparisons)
    EXCLUDE_PATTERNS = [
        r'\b(?:difference\s+between|compare|distinction\s+between)\b',
        r'(?:section|sec\.?|s\.)\s*[0-9]+\s+(?:and|vs\.?|versus|or)\s+(?:section|sec\.?|s\.)\s*[0-9]+',
        r'\bhow\s+many\s+sections\b'
    ]

    def detect(self, query: str) -> Optional[SectionIntent]:
        """Analyze query string and return SectionIntent if exact lookup is targeted, else None."""
        if not query or not query.strip():
            return None

        q_clean = query.strip()
        
        # Check exclusion patterns first
        for exp in self.EXCLUDE_PATTERNS:
            if re.search(exp, q_clean, re.IGNORECASE):
                return None

        # Detect Act qualifier if explicitly present
        act_match = re.search(self.ACT_PATTERN, q_clean, re.IGNORECASE)
        act_short = None
        if act_match:
            act_text = act_match.group(1).upper()
            if "NAGARIK" in act_text or act_text == "BNSS":
                act_short = "BNSS"
            else:
                act_short = "BNS"

        # Check section extraction patterns
        for pattern in self.SECTION_PATTERNS:
            match = re.search(pattern, q_clean, re.IGNORECASE)
            if match:
                sec_num = match.group(1)
                subsec = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                subsec_str = f"({subsec})" if subsec else None
                
                # Verify that query is relatively concise (not a long paragraph that merely mentions a section)
                word_count = len(q_clean.split())
                if word_count > 15:
                    return None
                
                return SectionIntent(
                    raw_query=query,
                    section_number=sec_num,
                    act_short=act_short,
                    subsection=subsec_str,
                    is_exact_lookup=True
                )

        return None

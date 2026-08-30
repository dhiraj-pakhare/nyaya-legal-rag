"""Structured citation parser and AST extractor for statutory citations."""

import re
from typing import List, Optional

from backend.app.generation.models import ParsedCitation


class CitationParser:
    """Extracts and normalizes structured statutory citations from text."""

    # Matches bracketed citations like [BNS s.103], [BNSS s.35(1)], [BNS Section 105(2)(a)]
    BRACKETED_CITATION_REGEX = re.compile(
        r'\[\s*(?P<act>BNS|BNSS)\s+(?:s\.|sec\.|section\s*)\s*(?P<section>\d+[A-Za-z]?)(?:\s*\((?P<sub1>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub2>[0-9a-zA-Z]+)\))?(?:\s*\((?P<sub3>[0-9a-zA-Z]+)\))?\s*\]',
        re.IGNORECASE
    )

    def parse(self, text: str) -> List[ParsedCitation]:
        """Parse all statutory citations present in the text into structured ParsedCitation objects."""
        if not text:
            return []

        citations: List[ParsedCitation] = []
        for match in self.BRACKETED_CITATION_REGEX.finditer(text):
            raw_text = match.group(0)
            act = match.group("act").upper()
            section = match.group("section").strip()
            
            sub1 = match.group("sub1")
            sub2 = match.group("sub2")
            sub3 = match.group("sub3")
            
            sub_parts = []
            if sub1:
                sub_parts.append(f"({sub1})")
            if sub2:
                sub_parts.append(f"({sub2})")
            if sub3:
                sub_parts.append(f"({sub3})")
            
            subsection_str = "".join(sub_parts) if sub_parts else None
            clause_str = f"({sub2})" if (sub1 and sub2 and not sub1.isalpha() and sub2.isalpha()) else None

            canonical_tag = f"[{act} s.{section}{subsection_str or ''}]"

            citations.append(
                ParsedCitation(
                    raw_text=raw_text,
                    act_short=act,
                    section_number=section,
                    subsection=subsection_str,
                    clause=clause_str,
                    canonical_tag=canonical_tag,
                    start_char=match.start(),
                    end_char=match.end()
                )
            )

        return citations

"""First Schedule (BNS Offences Table) Parser for Nyaya Legal RAG.

Parses Pages 158–189 of the Gazette PDF containing the classification table of
offences under the Bharatiya Nyaya Sanhita, 2023 (BNS Sections 1–356), extracting
section numbers, offence titles, penalties, bailability, cognizability, and triable courts.
"""

import re
from typing import List, Optional
import pypdf
from pydantic import BaseModel

from backend.app.ingestion.models import FirstScheduleEntry
from backend.app.ingestion.cleaner import is_gazette_boilerplate


# Regex to detect row start in the classification table
ROW_START_RE = re.compile(
    r'^(?P<sec>\d+(?:\s*\(\s*[0-9a-zA-Z]+\s*\))*(?:\s*\(\s*[a-zA-Z]+\s*\))*)\s+(?P<rest>.*)$'
)

# Common court names for extraction
COURT_PATTERNS = [
    r'Court of Session',
    r'Magistrate of the first class',
    r'Magistrate of the second class',
    r'Chief Judicial Magistrate',
    r'Any Magistrate',
    r'High Court',
    r'Court by which offence abetted is triable',
    r'Court by which offence is triable'
]
COMPILED_COURTS = [re.compile(p, re.IGNORECASE) for p in COURT_PATTERNS]


class FirstScheduleParser:
    """Extracts structured BNS offence entries from The First Schedule."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.reader = pypdf.PdfReader(pdf_path)

    def parse_schedule(self, start_page: int = 158, end_page: int = 189) -> List[FirstScheduleEntry]:
        """Parse First Schedule pages into structured FirstScheduleEntry objects."""
        entries: List[FirstScheduleEntry] = []
        
        current_sec: Optional[str] = None
        current_lines: List[str] = []
        current_page = start_page
        
        for p_num in range(start_page, end_page + 1):
            if p_num > len(self.reader.pages):
                break
            page = self.reader.pages[p_num - 1]
            raw_text = page.extract_text() or ""
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            
            for line in lines:
                # Skip table headers and Gazette boilerplate
                if (is_gazette_boilerplate(line) or line.startswith("Section Offence") or 
                    line.startswith("12 3 4 5 6") or "THE FIRST SCHEDULE" in line or 
                    "CLASSIFICATION OF OFFENCES" in line or "EXPLANATORY NOTES" in line or 
                    "I.—OFFENCES UNDER THE BHARATIYA NYAYA SANHITA" in line):
                    continue
                    
                m = ROW_START_RE.match(line)
                if m:
                    sec_candidate = m.group('sec')
                    # Clean section candidate (strip internal spaces e.g. "58( a)" -> "58(a)")
                    clean_sec = re.sub(r'\s+', '', sec_candidate)
                    
                    # Sanity check: BNS sections are numbers 1..356
                    first_digits_m = re.match(r'^\d+', clean_sec)
                    if first_digits_m:
                        first_digits = int(first_digits_m.group(0))
                        if 1 <= first_digits <= 356:
                            # Flush previous entry
                            if current_sec is not None and current_lines:
                                entry = self._build_entry(current_sec, current_lines, current_page)
                                if entry:
                                    entries.append(entry)
                                    
                            current_sec = clean_sec
                            current_lines = [m.group('rest')]
                            current_page = p_num
                            continue
                            
                # Line continuation of current entry
                if current_sec is not None:
                    current_lines.append(line)
                    
        # Flush final entry
        if current_sec is not None and current_lines:
            entry = self._build_entry(current_sec, current_lines, current_page)
            if entry:
                entries.append(entry)
                
        return entries

    def _build_entry(self, sec_num: str, lines: List[str], page: int) -> Optional[FirstScheduleEntry]:
        """Parse collected lines of a table row into a structured FirstScheduleEntry."""
        full_text = " ".join(lines).strip()
        if not full_text:
            return None
            
        # Detect Cognizability
        if re.search(r'\bNon-cognizable\b', full_text, re.IGNORECASE):
            cognizable = "Non-cognizable"
        elif re.search(r'\bCognizable\b', full_text, re.IGNORECASE):
            cognizable = "Cognizable"
        else:
            cognizable = "According as offence is cognizable or non-cognizable"
            
        # Detect Bailability
        if re.search(r'\bNon-bailable\b', full_text, re.IGNORECASE):
            bailable = "Non-bailable"
        elif re.search(r'\bBailable\b', full_text, re.IGNORECASE):
            bailable = "Bailable"
        else:
            bailable = "According as offence is bailable or non-bailable"
            
        # Detect Triable Court
        triable_court = "Court of Session"
        for cp in COMPILED_COURTS:
            m_court = cp.search(full_text)
            if m_court:
                triable_court = m_court.group(0).strip(' .')
                break
                
        # Split text into offence and punishment if possible
        raw_snippet = f"BNS Section {sec_num}: {full_text}"
        
        # Approximate offence name from the beginning of the text
        # Offence name is typically the first clause before punishment
        offence_m = re.split(r'\b(?:Imprisonment|Death|Rigorous|Simple|Fine|Same\s+as)\b', full_text, maxsplit=1)
        if len(offence_m) > 1 and offence_m[0].strip():
            offence_name = offence_m[0].strip(' .—,')
            punishment_text = full_text[len(offence_m[0]):].strip(' .—,')
        else:
            offence_name = full_text[:60].strip(' .—,')
            punishment_text = full_text
            
        return FirstScheduleEntry(
            section_number=sec_num,
            offence_name=offence_name,
            punishment=punishment_text,
            cognizable_status=cognizable,
            bailable_status=bailable,
            triable_court=triable_court,
            page=page,
            raw_text=raw_snippet
        )

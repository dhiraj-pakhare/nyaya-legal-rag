"""Statutory Structure Detector for Nyaya Legal RAG.

Parses the cleaned text and coordinate layout data from the Gazette PDF into a
hierarchical AST of Chapters, Sections, Subsections, Clauses, Provisos, Exceptions,
Explanations, and Illustrations.
"""

import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.app.ingestion.models import (
    ActIdentity,
    ActShortName,
    Chapter,
    Clause,
    ExceptionModel,
    Explanation,
    Illustration,
    Proviso,
    Section,
    Subsection,
)
from backend.app.ingestion.pdf_extractor import PageLayoutData
from backend.app.ingestion.marginal_notes import associate_marginal_notes_with_sections
from backend.app.ingestion.cross_ref import extract_cross_references


class StatutoryDocument(BaseModel):
    act: str = ActIdentity.BNSS.value
    act_short: str = ActShortName.BNSS.value
    chapters: List[Chapter] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)


# Regex patterns for statutory elements
CHAPTER_HEADER_RE = re.compile(r'^\s*CHAPTER\s+([IVXLCDM]+)\s*$', re.IGNORECASE)
SECTION_START_RE = re.compile(
    r'(?:^|(?<=\.\s))(?P<sec_num>\d+)\.\s*(?P<rest>[A-Z(].*)'
)
SECTION_START_ATTACHED_RE = re.compile(
    r'^(?P<sec_num>\d+)\.(?P<rest>[A-Za-z].*)$'
)
SUBSECTION_START_RE = re.compile(r'^\((\d+)\)\s*(.*)$')
CLAUSE_START_RE = re.compile(r'^\(([a-z]+)\)\s*(.*)$')
SUBCLAUSE_START_RE = re.compile(r'^\(([ivxlcdm]+)\)\s*(.*)$', re.IGNORECASE)

PROVISO_RE = re.compile(r'^(Provided\s+(?:further\s+|also\s+)?that\b.*)', re.IGNORECASE)
EXCEPTION_RE = re.compile(r'^(Exception(?:\s+\d+)?\s*[.—].*)', re.IGNORECASE)
EXPLANATION_RE = re.compile(r'^(Explanation(?:\s+\d+)?\s*[.—].*)', re.IGNORECASE)
ILLUSTRATION_RE = re.compile(r'^(Illustration(?:\s+[a-z]|\s*\([a-z]\))?\s*[.—].*)', re.IGNORECASE)


class StructureDetector:
    """Detects and parses statutory structure across Gazette pages."""

    def __init__(self, pages_data: List[PageLayoutData]):
        self.pages_data = pages_data

    def detect_structure(self) -> StatutoryDocument:
        """Parse all substantive pages into a structured StatutoryDocument AST."""
        chapters: List[Chapter] = []
        sections: List[Section] = []
        
        # State tracking
        current_chapter_num = "I"
        current_chapter_title = "PRELIMINARY"
        current_chapter_page_start = 1
        
        # Collect raw section data across pages first
        raw_sections_data = self._extract_raw_sections()
        
        # Build Section objects with parsed child hierarchy
        for r_sec in raw_sections_data:
            sec_obj = self._parse_section_body(r_sec)
            sections.append(sec_obj)
            
        # Group sections into Chapters
        chapter_map: Dict[str, List[Section]] = {}
        chapter_info_map: Dict[str, Tuple[str, int, int]] = {}  # num -> (title, page_start, page_end)
        
        for sec in sections:
            c_num = sec.chapter_number
            if c_num not in chapter_map:
                chapter_map[c_num] = []
                chapter_info_map[c_num] = (sec.chapter_title, sec.page_start, sec.page_end)
            else:
                title, p_start, _ = chapter_info_map[c_num]
                chapter_info_map[c_num] = (title, p_start, sec.page_end)
            chapter_map[c_num].append(sec)
            
        for c_num, c_sections in chapter_map.items():
            c_title, p_start, p_end = chapter_info_map[c_num]
            chapters.append(Chapter(
                chapter_number=c_num,
                chapter_title=c_title,
                page_start=p_start,
                page_end=p_end,
                sections=c_sections
            ))
            
        return StatutoryDocument(
            act=ActIdentity.BNSS.value,
            act_short=ActShortName.BNSS.value,
            chapters=chapters,
            sections=sections
        )

    def _extract_raw_sections(self) -> List[Dict]:
        """Iterate over all pages, track chapters, and segment text into raw section blocks."""
        raw_sections: List[Dict] = []
        
        current_chap_num = "I"
        current_chap_title = "PRELIMINARY"
        
        current_sec_num: Optional[str] = None
        current_sec_title: str = ""
        current_sec_lines: List[str] = []
        current_sec_page_start = 1
        current_sec_page_end = 1
        
        for p in self.pages_data:
            page_num = p.page_number
            lines = p.cleaned_main_text.splitlines()
            
            # Step 1: Detect section starts and associate marginal note titles for this page
            sec_starts_on_page = self._find_section_starts_on_page(p)
            titles_map = associate_marginal_notes_with_sections(p, sec_starts_on_page)
            
            line_idx = 0
            while line_idx < len(lines):
                line = lines[line_idx].strip()
                if not line:
                    line_idx += 1
                    continue
                    
                # Check Chapter Header
                chap_match = CHAPTER_HEADER_RE.match(line)
                if chap_match:
                    current_chap_num = chap_match.group(1).upper()
                    # Collect chapter title lines
                    title_parts = []
                    t_idx = line_idx + 1
                    while t_idx < len(lines) and t_idx < line_idx + 4:
                        t_line = lines[t_idx].strip()
                        if not t_line or CHAPTER_HEADER_RE.match(t_line) or SECTION_START_RE.search(t_line):
                            break
                        title_parts.append(t_line)
                        t_idx += 1
                    if title_parts:
                        current_chap_title = " ".join(title_parts).strip()
                    line_idx = t_idx
                    continue
                    
                # Check Section Start
                sec_match = SECTION_START_RE.search(line) or SECTION_START_ATTACHED_RE.match(line)
                if sec_match:
                    s_num = sec_match.group('sec_num')
                    # Validate that s_num is within 1..531
                    try:
                        s_val = int(s_num)
                        is_valid_sec = (1 <= s_val <= 531)
                    except ValueError:
                        is_valid_sec = False
                        
                    if is_valid_sec:
                        # If section start occurred mid-line (e.g. after "with it."), save the prefix to the previous section
                        prefix_text = line[:sec_match.start()].strip()
                        if prefix_text and current_sec_num is not None:
                            current_sec_lines.append(prefix_text)
                            
                        # Flush previous section
                        if current_sec_num is not None:
                            raw_sections.append({
                                'section_number': current_sec_num,
                                'section_title': current_sec_title,
                                'chapter_number': current_chap_num,
                                'chapter_title': current_chap_title,
                                'page_start': current_sec_page_start,
                                'page_end': current_sec_page_end,
                                'lines': list(current_sec_lines)
                            })
                            
                        # Start new section
                        current_sec_num = s_num
                        current_sec_title = titles_map.get(s_num, f"Section {s_num}")
                        rem_text = line[sec_match.start():].strip()
                        current_sec_lines = [rem_text]
                        current_sec_page_start = page_num
                        current_sec_page_end = page_num
                        line_idx += 1
                        continue
                        
                # Continuation of current section
                if current_sec_num is not None:
                    current_sec_lines.append(line)
                    current_sec_page_end = page_num
                    
                line_idx += 1
                
        # Flush final section
        if current_sec_num is not None:
            raw_sections.append({
                'section_number': current_sec_num,
                'section_title': current_sec_title,
                'chapter_number': current_chap_num,
                'chapter_title': current_chap_title,
                'page_start': current_sec_page_start,
                'page_end': current_sec_page_end,
                'lines': list(current_sec_lines)
            })
            
        return raw_sections

    def _find_section_starts_on_page(self, page_layout: PageLayoutData) -> List[Tuple[str, float]]:
        """Find section numbers and their Y coordinates in the main column of a page."""
        starts: List[Tuple[str, float]] = []
        
        # Group elements into lines by Y coordinate
        y_lines: Dict[float, List] = {}
        for elem in page_layout.main_column_elements:
            y_key = round(elem.y / 2.0) * 2.0
            if y_key not in y_lines:
                y_lines[y_key] = []
            y_lines[y_key].append(elem)
            
        for y_k in sorted(y_lines.keys(), reverse=True):
            elems = sorted(y_lines[y_k], key=lambda e: e.x)
            line_text = " ".join(e.text for e in elems).strip()
            
            m = SECTION_START_RE.search(line_text) or SECTION_START_ATTACHED_RE.match(line_text)
            if m:
                s_num = m.group('sec_num')
                try:
                    s_val = int(s_num)
                    if 1 <= s_val <= 531:
                        starts.append((s_num, y_k))
                except ValueError:
                    pass
        return starts

    def _parse_section_body(self, raw_sec: Dict) -> Section:
        """Parse raw text lines of a section into Subsections, Clauses, Provisos, Exceptions, etc."""
        lines: List[str] = raw_sec['lines']
        full_text = "\n".join(lines)
        page_start = raw_sec['page_start']
        page_end = raw_sec['page_end']
        
        subsections: List[Subsection] = []
        provisos: List[Proviso] = []
        exceptions: List[ExceptionModel] = []
        explanations: List[Explanation] = []
        illustrations: List[Illustration] = []
        
        # Scan for provisos, exceptions, explanations, and illustrations
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 1. Provisos
            prov_m = PROVISO_RE.match(line)
            if prov_m:
                prov_text_parts = [line]
                j = i + 1
                while j < len(lines):
                    next_l = lines[j].strip()
                    if (SUBSECTION_START_RE.match(next_l) or PROVISO_RE.match(next_l) or 
                        EXCEPTION_RE.match(next_l) or EXPLANATION_RE.match(next_l) or 
                        ILLUSTRATION_RE.match(next_l)):
                        break
                    prov_text_parts.append(next_l)
                    j += 1
                provisos.append(Proviso(
                    text="\n".join(prov_text_parts),
                    page=page_start
                ))
                i = j
                continue
                
            # 2. Exceptions
            exc_m = EXCEPTION_RE.match(line)
            if exc_m:
                exc_text_parts = [line]
                j = i + 1
                while j < len(lines):
                    next_l = lines[j].strip()
                    if (SUBSECTION_START_RE.match(next_l) or PROVISO_RE.match(next_l) or 
                        EXCEPTION_RE.match(next_l) or EXPLANATION_RE.match(next_l) or 
                        ILLUSTRATION_RE.match(next_l)):
                        break
                    exc_text_parts.append(next_l)
                    j += 1
                exceptions.append(ExceptionModel(
                    text="\n".join(exc_text_parts),
                    page=page_start
                ))
                i = j
                continue
                
            # 3. Explanations
            exp_m = EXPLANATION_RE.match(line)
            if exp_m:
                exp_text_parts = [line]
                j = i + 1
                while j < len(lines):
                    next_l = lines[j].strip()
                    if (SUBSECTION_START_RE.match(next_l) or PROVISO_RE.match(next_l) or 
                        EXCEPTION_RE.match(next_l) or EXPLANATION_RE.match(next_l) or 
                        ILLUSTRATION_RE.match(next_l)):
                        break
                    exp_text_parts.append(next_l)
                    j += 1
                explanations.append(Explanation(
                    text="\n".join(exp_text_parts),
                    page=page_start
                ))
                i = j
                continue
                
            # 4. Illustrations
            ill_m = ILLUSTRATION_RE.match(line)
            if ill_m:
                ill_text_parts = [line]
                j = i + 1
                while j < len(lines):
                    next_l = lines[j].strip()
                    if (SUBSECTION_START_RE.match(next_l) or PROVISO_RE.match(next_l) or 
                        EXCEPTION_RE.match(next_l) or EXPLANATION_RE.match(next_l) or 
                        ILLUSTRATION_RE.match(next_l)):
                        break
                    ill_text_parts.append(next_l)
                    j += 1
                illustrations.append(Illustration(
                    text="\n".join(ill_text_parts),
                    page=page_start
                ))
                i = j
                continue
                
            i += 1
            
        # Parse Subsections if present
        # A section may have explicit (1), (2) subsections starting at the section line or newlines
        subsec_matches = list(re.finditer(r'(?:^|\n|(?<=\.\s))\(([0-9]+)\)\s*', full_text))
        if subsec_matches:
            for idx, sm in enumerate(subsec_matches):
                sub_id = f"({sm.group(1)})"
                start_pos = sm.start()
                end_pos = subsec_matches[idx + 1].start() if idx + 1 < len(subsec_matches) else len(full_text)
                sub_text = full_text[start_pos:end_pos].strip()
                
                # Check clauses (a), (b) within this subsection
                clauses: List[Clause] = []
                clause_matches = list(re.finditer(r'(?:^|\n)\(([a-z]+)\)\s*', sub_text))
                for c_idx, cm in enumerate(clause_matches):
                    c_id = f"({cm.group(1)})"
                    c_start = cm.start()
                    c_end = clause_matches[c_idx + 1].start() if c_idx + 1 < len(clause_matches) else len(sub_text)
                    clauses.append(Clause(
                        clause_id=c_id,
                        text=sub_text[c_start:c_end].strip(),
                        page_start=page_start,
                        page_end=page_end
                    ))
                    
                subsections.append(Subsection(
                    subsection_id=sub_id,
                    text=sub_text,
                    page_start=page_start,
                    page_end=page_end,
                    clauses=clauses
                ))
                
        # Extract normalized cross-references
        refs = extract_cross_references(full_text, default_act_short="BNSS")
        
        return Section(
            section_number=raw_sec['section_number'],
            section_title=raw_sec['section_title'],
            act=ActIdentity.BNSS.value,
            act_short=ActShortName.BNSS.value,
            chapter_number=raw_sec['chapter_number'],
            chapter_title=raw_sec['chapter_title'],
            page_start=page_start,
            page_end=page_end,
            raw_text=full_text,
            subsections=subsections,
            provisos=provisos,
            exceptions=exceptions,
            explanations=explanations,
            illustrations=illustrations,
            references=refs
        )

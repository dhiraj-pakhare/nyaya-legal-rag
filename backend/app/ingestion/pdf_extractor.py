"""Coordinate-Aware PDF Text and Layout Extractor for Nyaya Legal RAG.

Extracts text blocks with exact (x, y) coordinates from the Gazette PDF, separates
main statutory columns from marginal notes (section titles), filters Gazette boilerplates,
and handles multi-column / schedule table pages.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import pypdf
from pydantic import BaseModel, Field

from backend.app.ingestion.cleaner import is_gazette_boilerplate, clean_statutory_text


class TextElement(BaseModel):
    x: float
    y: float
    font_size: float
    text: str


class PageLayoutData(BaseModel):
    page_number: int
    raw_text: str
    cleaned_main_text: str
    main_column_elements: List[TextElement] = Field(default_factory=list)
    marginal_note_elements: List[TextElement] = Field(default_factory=list)
    raw_lines: List[str] = Field(default_factory=list)


def extract_page_elements(page: pypdf.PageObject, page_number: int) -> Tuple[List[TextElement], str]:
    """Extract all text elements with their bounding coordinates from a PDF page."""
    elements: List[TextElement] = []
    
    def visitor(text: str, cm: Any, tm: Any, font_dict: Any, font_size: float):
        clean_t = text.strip()
        if clean_t:
            elements.append(TextElement(
                x=float(tm[4]),
                y=float(tm[5]),
                font_size=float(font_size),
                text=clean_t
            ))
            
    raw_text = page.extract_text(visitor_text=visitor) or ""
    return elements, raw_text


def process_page_layout(page: pypdf.PageObject, page_number: int) -> PageLayoutData:
    """Process a single PDF page into structured layout data separating main body and marginal notes."""
    elements, raw_text = extract_page_elements(page, page_number)
    
    is_even = (page_number % 2 == 0)
    main_elements: List[TextElement] = []
    margin_elements: List[TextElement] = []
    
    # Gazette margins:
    # Even pages: marginal notes are in left margin (x < 110), main body (110 <= x <= 485)
    # Odd pages: main body (110 <= x <= 480), marginal notes are in right margin (x > 475)
    for elem in elements:
        t = elem.text
        if is_gazette_boilerplate(t):
            continue
        if elem.y < 35 or elem.y > 765:  # Running header / footer zone
            continue
            
        if is_even:
            if elem.x < 110:
                margin_elements.append(elem)
            else:
                main_elements.append(elem)
        else:
            if elem.x > 475:
                margin_elements.append(elem)
            else:
                main_elements.append(elem)
                
    main_elements.sort(key=lambda e: (-e.y, e.x))
    margin_elements.sort(key=lambda e: (-e.y, e.x))
    
    # Clean the full extracted text (which preserves complete parentheses like (1), (a), (i))
    cleaned_main_text = clean_statutory_text(raw_text)
    
    # Filter out any marginal note lines that appear at the bottom of the cleaned text stream
    # Marginal note fragments are short words at the end of the page before the footer
    margin_words = set()
    for me in margin_elements:
        for w in me.text.split():
            margin_words.add(w.strip(' .—,'))
            
    cleaned_lines = []
    for line in cleaned_main_text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        # Skip if the line is just a marginal note fragment (e.g. '1 of 1871.', '2 of 2000.')
        if re.match(r'^\d+\s+of\s+\d{4}\.?$', line_clean, re.IGNORECASE):
            continue
        cleaned_lines.append(line_clean)
        
    cleaned_main_text = "\n".join(cleaned_lines)
    raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    
    return PageLayoutData(
        page_number=page_number,
        raw_text=raw_text,
        cleaned_main_text=cleaned_main_text,
        main_column_elements=main_elements,
        marginal_note_elements=margin_elements,
        raw_lines=raw_lines
    )


class PDFExtractor:
    """Coordinate-aware extractor over the entire statutory PDF."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.reader = pypdf.PdfReader(pdf_path)
        self.total_pages = len(self.reader.pages)
        
    def get_page_layout(self, page_number: int) -> PageLayoutData:
        """Extract layout data for a specific 1-indexed page."""
        if page_number < 1 or page_number > self.total_pages:
            raise ValueError(f"Page number {page_number} out of bounds (1..{self.total_pages})")
        page = self.reader.pages[page_number - 1]
        return process_page_layout(page, page_number)
        
    def extract_all_pages(self, start_page: int = 1, end_page: Optional[int] = None) -> List[PageLayoutData]:
        """Extract layout data across a page range."""
        if end_page is None:
            end_page = self.total_pages
        end_page = min(end_page, self.total_pages)
        
        pages_data: List[PageLayoutData] = []
        for p in range(start_page, end_page + 1):
            pages_data.append(self.get_page_layout(p))
        return pages_data

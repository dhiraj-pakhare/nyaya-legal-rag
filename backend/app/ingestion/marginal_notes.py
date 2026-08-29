"""Marginal Note (Section Title) Extractor and Associator for Nyaya Legal RAG.

Associates marginal notes from Gazette pages with section headers based on
spatial vertical coordinate alignment.
"""

import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from backend.app.ingestion.pdf_extractor import PageLayoutData, TextElement


class MarginalNoteCluster(BaseModel):
    y_top: float
    y_bottom: float
    title: str
    elements: List[TextElement]


def cluster_marginal_notes(elements: List[TextElement], max_gap: float = 14.0) -> List[MarginalNoteCluster]:
    """Group individual marginal note text elements into vertical title clusters."""
    if not elements:
        return []
        
    # Sort elements top to bottom (descending Y)
    sorted_elems = sorted(elements, key=lambda e: -e.y)
    
    clusters: List[List[TextElement]] = []
    current_cluster: List[TextElement] = [sorted_elems[0]]
    
    for elem in sorted_elems[1:]:
        prev_elem = current_cluster[-1]
        gap = prev_elem.y - elem.y
        prev_ended = prev_elem.text.strip().endswith('.')
        
        # In marginal notes, line spacing is ~9-12 points. Gap > 14 or ending with a dot signifies a new title.
        if 0 <= gap <= max_gap and not prev_ended:
            current_cluster.append(elem)
        else:
            clusters.append(current_cluster)
            current_cluster = [elem]
            
    if current_cluster:
        clusters.append(current_cluster)
        
    result: List[MarginalNoteCluster] = []
    for c in clusters:
        # Join words with spaces, clean trailing dots/spaces
        raw_title = " ".join(e.text for e in c).strip()
        # Clean title: strip trailing footnote numbers or act references like "2 of 2000."
        clean_title = clean_marginal_title(raw_title)
        if clean_title:
            result.append(MarginalNoteCluster(
                y_top=c[0].y,
                y_bottom=c[-1].y,
                title=clean_title,
                elements=c
            ))
            
    return result


def clean_marginal_title(raw_title: str) -> str:
    """Clean marginal note title text by removing footnote references, trailing page numbers, etc."""
    title = raw_title.strip()
    
    # Remove act references like "1 of 1871.", "2 of 2000.", "18 of 2013."
    title = re.sub(r'\b\d+\s+of\s+\d{4}\.?\b', '', title, flags=re.IGNORECASE).strip()
    
    # Remove trailing isolated single digits (e.g., page numbers or footnote indicators)
    title = re.sub(r'\s+\d+$', '', title).strip()
    
    # Normalize internal spaces
    title = re.sub(r'\s+', ' ', title)
    
    # Strip trailing punctuation except if it's meaningful
    title = title.rstrip(' .—,-')
    
    return title.strip()


def associate_marginal_notes_with_sections(
    page_layout: PageLayoutData,
    section_starts: List[Tuple[str, float]]  # List of (section_number, y_coord)
) -> Dict[str, str]:
    """Map each section number starting on this page to its dynamically extracted marginal note title.
    
    Args:
        page_layout: PageLayoutData for the current page.
        section_starts: List of tuples (section_number_str, y_coord_in_main_column).
        
    Returns:
        Dictionary mapping section_number -> section_title.
    """
    if not section_starts:
        return {}
        
    # Handle Page 1 special case for Section 1 if needed
    if page_layout.page_number == 1:
        # Page 1 has Section 1: "Short title, extent and commencement."
        # Often placed near y ~ 0 or bottom right
        titles_on_p1 = [e.text for e in page_layout.main_column_elements if "Short title" in e.text or "extent and" in e.text]
        if not titles_on_p1:
            # Check marginal elements
            p1_notes = [e.text for e in page_layout.marginal_note_elements if "Short title" in e.text or "extent" in e.text]
            if p1_notes:
                return {"1": "Short title, extent and commencement"}
        return {"1": "Short title, extent and commencement"}
        
    clusters = cluster_marginal_notes(page_layout.marginal_note_elements)
    if not clusters:
        return {}
        
    assignments: Dict[str, str] = {}
    used_clusters = set()
    
    for sec_num, sec_y in section_starts:
        best_cluster: Optional[MarginalNoteCluster] = None
        best_dist = float('inf')
        
        for idx, cluster in enumerate(clusters):
            if idx in used_clusters:
                continue
            # Distance from section start Y to cluster top Y
            dist = abs(cluster.y_top - sec_y)
            if dist < best_dist and dist <= 55.0:  # Within vertical tolerance
                best_dist = dist
                best_cluster = cluster
                best_idx = idx
                
        if best_cluster:
            assignments[sec_num] = best_cluster.title
            used_clusters.add(best_idx)
            
    return assignments

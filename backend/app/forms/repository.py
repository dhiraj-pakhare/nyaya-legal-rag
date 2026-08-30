"""Statutory Form In-Memory Registry for Nyaya Legal RAG (Phase 7).

Manages pre-parsed statutory forms with high-performance multi-key reverse indexing
by Form Number, Form ID, Applicable Statutory Sections, and Normalized Titles.
Enforces strict tenant and corpus isolation (user documents never enter this registry).
"""

import logging
import re
from typing import Dict, List, Optional
import threading

from backend.app.forms.models import StatutoryForm
from backend.app.forms.parser import SecondScheduleParser

logger = logging.getLogger("nyaya.forms.repository")


class StatutoryFormRegistry:
    """Thread-safe in-memory registry for authoritative Second Schedule Statutory Forms."""

    def __init__(self, forms: List[StatutoryForm]):
        self._by_number: Dict[int, StatutoryForm] = {}
        self._by_id: Dict[str, StatutoryForm] = {}
        self._by_section: Dict[str, List[StatutoryForm]] = {}
        self._by_title: Dict[str, StatutoryForm] = {}
        self._forms: List[StatutoryForm] = forms

        self._build_indexes(forms)

    def _build_indexes(self, forms: List[StatutoryForm]) -> None:
        """Construct multi-key lookup indexes."""
        for form in forms:
            self._by_number[form.form_number] = form
            self._by_id[form.form_id.upper()] = form
            
            # Title index (case-normalized & punctuation-stripped)
            norm_title = self.normalize_text(form.form_title)
            self._by_title[norm_title] = form

            # Section index (reverse lookup from statutory provision to form)
            for sec in form.applicable_sections:
                clean_sec = self.normalize_section(sec)
                if clean_sec not in self._by_section:
                    self._by_section[clean_sec] = []
                self._by_section[clean_sec].append(form)

                # Also index base section number e.g. "35" from "35(3)"
                base_m = re.match(r'^\d+', clean_sec)
                if base_m:
                    base_sec = base_m.group(0)
                    if base_sec != clean_sec:
                        if base_sec not in self._by_section:
                            self._by_section[base_sec] = []
                        if form not in self._by_section[base_sec]:
                            self._by_section[base_sec].append(form)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize query or title string for robust matching."""
        t = text.lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    @staticmethod
    def normalize_section(section: str) -> str:
        """Normalize section string e.g. ' 35 ( 3 ) ' -> '35(3)'."""
        s = section.lower()
        s = re.sub(r'^section\s*', '', s)
        s = re.sub(r'^s\.\s*', '', s)
        s = re.sub(r'\s+', '', s)
        return s

    def get_by_number(self, form_number: int) -> Optional[StatutoryForm]:
        """Lookup form by exact integer form number (1..58)."""
        return self._by_number.get(form_number)

    def get_by_id(self, form_id: str) -> Optional[StatutoryForm]:
        """Lookup form by canonical ID e.g. 'BNSS_FORM_01'."""
        clean_id = form_id.strip().upper()
        return self._by_id.get(clean_id)

    def get_by_section(self, section_str: str) -> List[StatutoryForm]:
        """Lookup forms associated with a statutory section reference e.g. '35(3)', '63', '83'."""
        clean_sec = self.normalize_section(section_str)
        return self._by_section.get(clean_sec, [])

    def get_by_exact_title(self, title: str) -> Optional[StatutoryForm]:
        """Lookup form by exact normalized title."""
        norm = self.normalize_text(title)
        return self._by_title.get(norm)

    def list_all_forms(self) -> List[StatutoryForm]:
        """Return all 58 statutory forms in sequential order."""
        return list(self._forms)

    def count(self) -> int:
        """Return count of registered statutory forms."""
        return len(self._forms)


_registry_instance: Optional[StatutoryFormRegistry] = None
_registry_lock = threading.Lock()


def get_form_registry(pdf_path: str = "BNS bare act 2023.pdf") -> StatutoryFormRegistry:
    """Thread-safe singleton accessor for the StatutoryFormRegistry."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                parser = SecondScheduleParser(pdf_path=pdf_path)
                forms = parser.parse_forms()
                _registry_instance = StatutoryFormRegistry(forms)
    return _registry_instance


def reset_form_registry() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _registry_instance
    with _registry_lock:
        _registry_instance = None

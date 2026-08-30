"""Second Schedule Statutory Forms Parser for Nyaya Legal RAG (Phase 7).

Parses Pages 190–249 of the Gazette PDF containing all 58 statutory forms under
The Second Schedule of the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS Forms 1–58).
Enforces programmatic structural invariants and dynamic boundary extraction.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple
import pypdf

from backend.app.forms.models import (
    FormField,
    FormFieldType,
    FormSignature,
    FormTableHead,
    StatutoryForm,
)

logger = logging.getLogger("nyaya.forms.parser")

# Regex for detecting form headers e.g. "FORM No. 1", "FORM NO. 33", "FORM  No. 58"
FORM_HEADER_RE = re.compile(
    r'(?:^|\n)\s*FORM\s+(?:No\.?|NO\.?)\s*(\d+)',
    re.IGNORECASE
)

# Regex for extracting applicable section reference e.g. "(See section 63)", "[See section 35(3)]"
SECTION_REF_RE = re.compile(
    r'(?:\[|\()\s*See\s+sections?\s+([0-9a-zA-Z\s,\(\)andu/]+)(?:\]|\))',
    re.IGNORECASE
)

# Gazette running header pattern
GAZETTE_HEADER_RE = re.compile(
    r'SEC\.\s*1\]\s+THE\s+GAZETTE\s+OF\s+INDIA\s+EXTRAORDINAR\s*Y\s+\d+',
    re.IGNORECASE
)

# Gazette publication boilerplate at the end of the document
BOILERPLATE_LINES = [
    "DIWAKAR SINGH,",
    "Joint Secretary & Legislative Counsel to the Govt. of  India.",
    "MGIPMRND—",
    "UPLOADED BY THE MANAGER, GOVERNMENT OF INDIA PRESS",
    "AND PUBLISHED BY  THE CONTROLLER OF  PUBLICA TIONS",
    "THE SECOND SCHEDULE",
    "(See section 522)"
]


class InvariantValidationError(ValueError):
    """Raised when statutory forms fail source integrity invariant checks."""
    pass


class SecondScheduleParser:
    """Extracts and validates structured statutory forms from The Second Schedule."""

    def __init__(self, pdf_path: str = "BNS bare act 2023.pdf"):
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Statutory PDF source not found at: {pdf_path}")
        self.reader = pypdf.PdfReader(pdf_path)

    def parse_forms(self, start_page: int = 190, end_page: int = 249) -> List[StatutoryForm]:
        """Parse Second Schedule pages into structured StatutoryForm objects with invariant checks."""
        # 1. Extract raw cleaned page texts
        raw_pages: Dict[int, str] = {}
        for p_idx in range(start_page - 1, min(end_page, len(self.reader.pages))):
            page_num = p_idx + 1
            raw_text = self.reader.pages[p_idx].extract_text() or ""
            cleaned_text = self._clean_page_text(raw_text, page_num)
            raw_pages[page_num] = cleaned_text

        # 2. Dynamic multi-page form boundary identification
        form_raw_blocks = self._split_into_form_blocks(raw_pages, start_page, end_page)

        # 3. Construct structured StatutoryForm models
        forms: List[StatutoryForm] = []
        for f_num, (p_start, p_end, raw_content) in sorted(form_raw_blocks.items()):
            form_obj = self._build_statutory_form(f_num, p_start, p_end, raw_content)
            forms.append(form_obj)

        # 4. Enforce strict programmatic invariants
        self.validate_invariants(forms)
        logger.info(f"Successfully extracted and verified {len(forms)} statutory forms from The Second Schedule.")
        return forms

    def _clean_page_text(self, text: str, page_num: int) -> str:
        """Strip running Gazette headers and publication boilerplate while preserving form text."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        cleaned_lines = []
        for line in lines:
            if GAZETTE_HEADER_RE.search(line):
                continue
            # Strip preamble on page 190
            if page_num == 190 and line in ("THE SECOND SCHEDULE", "(See section 522)"):
                continue
            # Strip publication boilerplate on page 249
            if page_num == 249 and any(bp in line for bp in BOILERPLATE_LINES):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _split_into_form_blocks(
        self,
        raw_pages: Dict[int, str],
        start_page: int,
        end_page: int
    ) -> Dict[int, Tuple[int, int, str]]:
        """Dynamically slice pages into individual form text blocks."""
        # Find start locations of every FORM No. X
        # List of (form_number, page_num, line_idx_in_page)
        headers: List[Tuple[int, int, int]] = []
        page_lines: Dict[int, List[str]] = {}

        for p_num in range(start_page, end_page + 1):
            if p_num not in raw_pages:
                continue
            lines = raw_pages[p_num].splitlines()
            page_lines[p_num] = lines
            for l_idx, line in enumerate(lines):
                m = FORM_HEADER_RE.match(line)
                if m:
                    f_num = int(m.group(1))
                    headers.append((f_num, p_num, l_idx))

        # Sort headers by form_number
        headers.sort(key=lambda x: x[0])

        form_blocks: Dict[int, Tuple[int, int, str]] = {}

        for i, (f_num, p_start, l_start) in enumerate(headers):
            # Determine boundary where next form begins
            if i + 1 < len(headers):
                _, next_p_start, next_l_start = headers[i + 1]
            else:
                next_p_start = end_page
                next_l_start = len(page_lines.get(end_page, []))

            # Assemble lines spanning from (p_start, l_start) to (next_p_start, next_l_start)
            collected_lines: List[str] = []
            for p in range(p_start, next_p_start + 1):
                cur_p_lines = page_lines.get(p, [])
                start_l = l_start if p == p_start else 0
                end_l = next_l_start if p == next_p_start else len(cur_p_lines)
                collected_lines.extend(cur_p_lines[start_l:end_l])

            # Determine true page_end: the last page from which lines were collected
            p_end = p_start
            if next_p_start > p_start:
                # If lines were taken from next_p_start (prior to next_l_start), p_end is next_p_start
                if next_l_start > 0:
                    p_end = next_p_start
                else:
                    p_end = next_p_start - 1

            raw_block = "\n".join(collected_lines).strip()
            form_blocks[f_num] = (p_start, p_end, raw_block)

        return form_blocks

    def _build_statutory_form(
        self,
        form_number: int,
        page_start: int,
        page_end: int,
        raw_text: str
    ) -> StatutoryForm:
        """Parse raw form text into typed StatutoryForm model with fields, signatures, and sections."""
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        
        # 1. Extract Form Title and Section References
        form_title, applicable_sections, body_lines = self._extract_title_and_sections(lines, form_number)

        # 2. Extract Placeholders and Typed Fields
        fields = self._extract_fields(body_lines, form_number)

        # 3. Extract Signatures / Seals
        signatures = self._extract_signatures(body_lines)

        # 4. Extract Tables / Charge Heads for multi-head charge forms (Form 33)
        tables = self._extract_tables(raw_text) if form_number == 33 else []

        form_id = f"BNSS_FORM_{form_number:02d}"
        provenance = f"[BNSS Second Schedule, Form {form_number}]"

        return StatutoryForm(
            form_id=form_id,
            form_number=form_number,
            form_title=form_title,
            applicable_sections=applicable_sections,
            page_start=page_start,
            page_end=page_end,
            raw_text=raw_text,
            fields=fields,
            signatures=signatures,
            tables=tables,
            provenance_citation=provenance
        )

    def _extract_title_and_sections(
        self,
        lines: List[str],
        form_number: int
    ) -> Tuple[str, List[str], List[str]]:
        """Extract title lines and statutory provision references."""
        title_parts: List[str] = []
        applicable_sections: List[str] = []
        body_start_idx = 1  # Line 0 is 'FORM No. X'

        for idx in range(1, min(6, len(lines))):
            line = lines[idx]
            sec_m = SECTION_REF_RE.search(line)
            if sec_m:
                raw_secs = sec_m.group(1).strip()
                # Parse multiple sections e.g. "234, 235 and 236" or "35(3)" or "84, 90 and 93"
                sec_nums = re.findall(r'\d+(?:\s*\(\s*[0-9a-zA-Z]+\s*\))*', raw_secs)
                clean_secs = [re.sub(r'\s+', '', s) for s in sec_nums if s.strip()]
                applicable_sections = clean_secs if clean_secs else [raw_secs]
                body_start_idx = idx + 1
                break
            else:
                title_parts.append(line)

        title = " ".join(title_parts).strip()
        if not title and len(lines) > 1:
            title = lines[1]
            body_start_idx = 2

        # Clean title internal whitespace
        title = re.sub(r'\s+', ' ', title)
        body_lines = lines[body_start_idx:]
        return title, applicable_sections, body_lines

    def _extract_fields(self, lines: List[str], form_number: int) -> List[FormField]:
        """Detect and structure placeholder lines, addressees, and variable fields."""
        fields: List[FormField] = []
        field_idx = 1

        for i, line in enumerate(lines):
            # Check for dotted underline placeholder
            if "..." in line or "…" in line or "___" in line:
                # Look for bracketed guide on current line or next line
                guide_m = re.search(r'[\[\(]([a-zA-Z0-9\s,\./\-\(\)]+)[\]\)]', line)
                if not guide_m and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.startswith("[") or next_line.startswith("("):
                        guide_m = re.search(r'[\[\(]([a-zA-Z0-9\s,\./\-\(\)]+)[\]\)]', next_line)

                label = guide_m.group(1).strip() if guide_m else f"Field {field_idx}"
                
                # Filter out pure court seal/signature indicators as fields
                if any(k in label.lower() for k in ["signature", "seal", "magistrate", "judge"]):
                    continue

                f_type = FormFieldType.TEXT_PLACEHOLDER
                if any(k in label.lower() for k in ["date", "day of"]):
                    f_type = FormFieldType.DATE_PLACEHOLDER
                elif any(k in label.lower() for k in ["time", "am/pm", "hours"]):
                    f_type = FormFieldType.TIME_PLACEHOLDER
                elif any(k in label.lower() for k in ["address", "police station", "district", "place"]):
                    f_type = FormFieldType.LOCATION_PLACEHOLDER
                elif "section" in label.lower() or "u/s" in label.lower():
                    f_type = FormFieldType.SECTION_REFERENCE

                fields.append(
                    FormField(
                        field_id=f"bnss_f{form_number:02d}_f{field_idx:02d}",
                        label=label,
                        field_type=f_type,
                        raw_text=line,
                        placeholder=re.sub(r'[a-zA-Z0-9\[\]\(\)]', '', line).strip() or "............",
                        is_required=True
                    )
                )
                field_idx += 1

        return fields

    def _extract_signatures(self, lines: List[str]) -> List[FormSignature]:
        """Extract signature, seal, and attestation blocks."""
        signatures: List[FormSignature] = []
        for line in lines:
            if any(k in line for k in ["(Signature", "(Seal of the Court)", "Signature and seal", "Police Officer"]):
                signatures.append(
                    FormSignature(
                        signatory_title=line,
                        seal_required="seal" in line.lower()
                    )
                )
        return signatures

    def _extract_tables(self, raw_text: str) -> List[FormTableHead]:
        """Extract multi-head charge components for Form 33."""
        heads: List[FormTableHead] = []
        head_re = re.compile(r'(?:^|\n)([I|V|X]+|\([0-9]+\))\.\s+([^\n]+)', re.IGNORECASE)
        matches = list(head_re.finditer(raw_text))
        for idx, m in enumerate(matches):
            h_num = m.group(1).strip()
            h_title = m.group(2).strip()
            start_pos = m.end()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            c_text = raw_text[start_pos:end_pos].strip()
            heads.append(
                FormTableHead(
                    head_number=h_num,
                    head_title=h_title,
                    charge_text=c_text[:300]
                )
            )
        return heads

    @classmethod
    def validate_invariants(cls, forms: List[StatutoryForm]) -> None:
        """Enforce strict programmatic source integrity invariants."""
        # 1. Total form count must be exactly 58
        if len(forms) != 58:
            raise InvariantValidationError(f"Invariant Violated: Expected exactly 58 statutory forms, found {len(forms)}")

        # 2. Numbering must be strictly contiguous 1..58
        numbers = [f.form_number for f in forms]
        expected_numbers = list(range(1, 59))
        if numbers != expected_numbers:
            missing = set(expected_numbers) - set(numbers)
            duplicates = [num for num in numbers if numbers.count(num) > 1]
            raise InvariantValidationError(
                f"Invariant Violated: Form numbers are not contiguous 1..58. Missing: {missing}, Duplicates: {duplicates}"
            )

        # 3. Form 1 must start on page 190, Form 58 on page 249
        if forms[0].page_start != 190:
            raise InvariantValidationError(f"Invariant Violated: Form 1 must start on Page 190, got {forms[0].page_start}")
        if forms[-1].page_start != 249:
            raise InvariantValidationError(f"Invariant Violated: Form 58 must start on Page 249, got {forms[-1].page_start}")

        # 4. Form 33 must span pages 222 to 224
        form_33 = next((f for f in forms if f.form_number == 33), None)
        if not form_33:
            raise InvariantValidationError("Invariant Violated: Form 33 (Charges) not found.")
        if form_33.page_start != 222 or form_33.page_end != 224:
            raise InvariantValidationError(
                f"Invariant Violated: Form 33 expected to span Pages 222-224, got {form_33.page_start}-{form_33.page_end}"
            )

        # 5. Non-empty text and title on every form
        for f in forms:
            if not f.form_title or len(f.form_title.strip()) < 3:
                raise InvariantValidationError(f"Invariant Violated: Form {f.form_number} has empty or invalid title: '{f.form_title}'")
            if not f.raw_text or len(f.raw_text.strip()) < 50:
                raise InvariantValidationError(f"Invariant Violated: Form {f.form_number} has suspiciously short raw text: {len(f.raw_text)} chars")
            if f.page_start > f.page_end:
                raise InvariantValidationError(f"Invariant Violated: Form {f.form_number} page_start ({f.page_start}) > page_end ({f.page_end})")

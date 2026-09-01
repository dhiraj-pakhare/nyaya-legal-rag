"""Statutory Forms Application Service (Part B & Phase 8).

Coordinates deterministic statutory form retrieval, disambiguation, PDF export, and bulk downloads.
"""

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import zipfile
import pypdf

from backend.app.api.errors import NotFoundError
from backend.app.api.schemas.forms import (
    FormFieldDTO,
    FormLookupResponseDTO,
    FormSignatureDTO,
    FormTableHeadDTO,
    StatutoryFormDTO,
    StatutoryFormListItemDTO,
    StatutoryFormListResponseDTO,
)
from backend.app.forms.exporter import (
    calculate_extraction_confidence,
    get_form_filename,
    slugify_form_title,
)
from backend.app.forms.lookup import DeterministicFormIdentifier
from backend.app.forms.models import StatutoryForm
from backend.app.forms.renderer import DeterministicFormRenderer
from backend.app.forms.repository import StatutoryFormRegistry, get_form_registry

logger = logging.getLogger("nyaya.services.forms")


class StatutoryFormsService:
    """Application service for Second Schedule statutory forms."""

    def __init__(
        self,
        registry: Optional[StatutoryFormRegistry] = None,
        forms_dir: str = "data/forms",
        source_pdf_path: Optional[str] = None
    ):
        self.registry = registry or get_form_registry()
        self.identifier = DeterministicFormIdentifier(registry=self.registry)
        self.renderer = DeterministicFormRenderer()
        self.forms_dir = forms_dir
        self.source_pdf_path = self.resolve_source_pdf(source_pdf_path)

    @classmethod
    def resolve_source_pdf(cls, explicit_path: Optional[str] = None) -> str:
        """Deterministically resolve the statutory source PDF location.

        Resolution order:
        1. If explicit_path is provided by the caller:
           - Return its absolute path if it exists on disk
           - Otherwise return explicit_path directly (so caller's explicit path is preserved)
        2. Configured settings.pdf_path (if exists on disk)
        3. Environment variable PDF_PATH (if set and exists on disk)
        4. Standard container & local workspace candidates:
           - /app/BNS bare act 2023.pdf (Railway/Docker container root)
           - BNS bare act 2023.pdf (Local CWD)
           - <repo_root>/BNS bare act 2023.pdf
           - /app/data/raw/BNS bare act 2023.pdf
           - data/raw/BNS bare act 2023.pdf
        5. Fallback default: settings.pdf_path or "BNS bare act 2023.pdf"
        """
        if explicit_path:
            if os.path.isfile(explicit_path):
                return os.path.abspath(explicit_path)
            return explicit_path

        try:
            from backend.app.core.config import settings
            if getattr(settings, "pdf_path", None) and os.path.isfile(settings.pdf_path):
                return os.path.abspath(settings.pdf_path)
        except Exception:
            pass

        env_path = os.getenv("PDF_PATH")
        if env_path and os.path.isfile(env_path):
            return os.path.abspath(env_path)

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        candidates = [
            "/app/BNS bare act 2023.pdf",
            "BNS bare act 2023.pdf",
            os.path.join(repo_root, "BNS bare act 2023.pdf"),
            "/app/data/raw/BNS bare act 2023.pdf",
            os.path.join(repo_root, "data", "raw", "BNS bare act 2023.pdf"),
            "data/raw/BNS bare act 2023.pdf",
        ]

        for path in candidates:
            if path and os.path.isfile(path):
                return os.path.abspath(path)

        try:
            from backend.app.core.config import settings
            if getattr(settings, "pdf_path", None):
                return settings.pdf_path
        except Exception:
            pass

        return "BNS bare act 2023.pdf"

    def list_forms(self, api_prefix: str = "/api/v1") -> StatutoryFormListResponseDTO:
        """List all 58 statutory forms with metadata, byte sizes, hashes, and download links."""
        forms = self.registry.get_all_forms()
        manifest_data: Dict[int, Dict[str, Any]] = {}

        # Load cached manifest if available on disk
        manifest_path = os.path.join(self.forms_dir, "forms_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    for item in m.get("forms", []):
                        manifest_data[item["form_number"]] = item
            except Exception as e:
                logger.warning(f"Failed to read forms_manifest.json: {e}")

        items: List[StatutoryFormListItemDTO] = []
        for form in sorted(forms, key=lambda f: f.form_number):
            cached = manifest_data.get(form.form_number)
            filename = cached.get("filename") if cached else get_form_filename(form.form_number, form.form_title)
            slug = cached.get("slug") if cached else slugify_form_title(form.form_title)
            byte_size = cached.get("byte_size") if cached else None
            sha256 = cached.get("sha256") if cached else None

            if cached:
                conf_score = cached.get("extraction_confidence", 1.0)
                needs_review = cached.get("needs_review", False)
            else:
                conf_score, needs_review = calculate_extraction_confidence(form)

            download_url = f"{api_prefix}/forms/{form.form_number}/download"
            page_count = form.page_end - form.page_start + 1

            items.append(
                StatutoryFormListItemDTO(
                    form_number=form.form_number,
                    form_id=form.form_id,
                    title=form.form_title,
                    slug=slug,
                    filename=filename,
                    applicable_sections=form.applicable_sections,
                    page_start=form.page_start,
                    page_end=form.page_end,
                    page_count=page_count,
                    byte_size=byte_size,
                    sha256=sha256,
                    extraction_confidence=conf_score,
                    needs_review=needs_review,
                    download_url=download_url,
                    provenance=form.provenance_citation
                )
            )

        return StatutoryFormListResponseDTO(
            total_forms=len(items),
            schedule="The Second Schedule (Bharatiya Nagarik Suraksha Sanhita, 2023)",
            forms=items
        )

    def lookup_form(self, query: str) -> FormLookupResponseDTO:
        """Execute deterministic form identification and return typed DTO."""
        raw_res = self.identifier.identify(query)
        form_dto: Optional[StatutoryFormDTO] = None
        rendered_md: Optional[str] = None

        if raw_res.status == "SUCCESS" and raw_res.form:
            form_dto = self.map_form_to_dto(raw_res.form)
            rendered_md = self.renderer.render_markdown(raw_res.form)

        return FormLookupResponseDTO(
            status=raw_res.status,
            query=raw_res.query,
            form=form_dto,
            candidate_forms=raw_res.candidate_forms,
            provenance=raw_res.provenance,
            rendered_markdown=rendered_md,
            is_refused=raw_res.is_refused,
            refusal_reason=raw_res.refusal_reason,
            latency_ms=raw_res.latency_ms
        )

    def search_forms(self, query: str) -> FormLookupResponseDTO:
        """Deterministic search endpoint (alias for lookup_form)."""
        return self.lookup_form(query)

    def get_form_by_id_or_number(self, id_or_number: str) -> StatutoryFormDTO:
        """Retrieve a statutory form directly by ID ('BNSS_FORM_01') or number ('1')."""
        clean = id_or_number.strip()
        form: Optional[StatutoryForm] = None

        if clean.isdigit():
            form = self.registry.get_by_number(int(clean))
        else:
            form = self.registry.get_by_id(clean)

        if not form:
            raise NotFoundError(
                message=f"Statutory form '{id_or_number}' not found in The Second Schedule (available: Forms 1–58).",
                details={"form_identifier": id_or_number}
            )

        return self.map_form_to_dto(form)

    def get_form_pdf_bytes(self, id_or_number: str) -> Tuple[str, bytes]:
        """Retrieve the binary PDF bytes and filename for a specific statutory form.

        Checks disk storage first; generates dynamically from source if not cached.
        """
        clean = id_or_number.strip()
        form: Optional[StatutoryForm] = None

        if clean.isdigit():
            form = self.registry.get_by_number(int(clean))
        else:
            form = self.registry.get_by_id(clean)

        if not form:
            raise NotFoundError(
                message=f"Statutory form '{id_or_number}' not found for download (available: Forms 1–58).",
                details={"form_identifier": id_or_number}
            )

        filename = get_form_filename(form.form_number, form.form_title)
        disk_path = os.path.join(self.forms_dir, filename)

        if os.path.exists(disk_path):
            with open(disk_path, "rb") as f:
                return filename, f.read()

        # Dynamic on-demand page extraction fallback
        source_path = self.resolve_source_pdf(self.source_pdf_path)
        if not os.path.exists(source_path):
            raise NotFoundError(
                message="Source statutory PDF unavailable for export.",
                details={"source_pdf": self.source_pdf_path}
            )

        reader = pypdf.PdfReader(source_path)
        writer = pypdf.PdfWriter()
        for p_num in range(form.page_start, form.page_end + 1):
            p_idx = p_num - 1
            if 0 <= p_idx < len(reader.pages):
                writer.add_page(reader.pages[p_idx])

        bio = io.BytesIO()
        writer.write(bio)
        pdf_bytes = bio.getvalue()

        # Cache dynamically extracted PDF to disk if directory is writable
        try:
            os.makedirs(self.forms_dir, exist_ok=True)
            with open(disk_path, "wb") as f:
                f.write(pdf_bytes)
        except Exception as exc:
            logger.debug(f"Could not cache extracted form to disk: {exc}")

        return filename, pdf_bytes

    def get_bulk_forms_zip(self) -> Tuple[str, bytes]:
        """Generate a single ZIP archive containing all 58 statutory form PDFs."""
        forms = sorted(self.registry.get_all_forms(), key=lambda f: f.form_number)
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for form in forms:
                fname, pdf_bytes = self.get_form_pdf_bytes(str(form.form_number))
                zf.writestr(fname, pdf_bytes)

        zip_filename = "bnss_second_schedule_forms_1_to_58.zip"
        return zip_filename, zip_buffer.getvalue()

    @staticmethod
    def map_form_to_dto(form: StatutoryForm) -> StatutoryFormDTO:
        """Map internal domain StatutoryForm to API StatutoryFormDTO."""
        return StatutoryFormDTO(
            form_id=form.form_id,
            form_number=form.form_number,
            form_title=form.form_title,
            act=form.act,
            act_short=form.act_short,
            schedule=form.schedule,
            applicable_sections=form.applicable_sections,
            page_start=form.page_start,
            page_end=form.page_end,
            raw_text=form.raw_text,
            fields=[
                FormFieldDTO(
                    field_id=f.field_id,
                    label=f.label,
                    field_type=f.field_type.value,
                    placeholder=f.placeholder,
                    is_required=f.is_required
                )
                for f in form.fields
            ],
            signatures=[
                FormSignatureDTO(
                    signatory_title=s.signatory_title,
                    seal_required=s.seal_required
                )
                for s in form.signatures
            ],
            tables=[
                FormTableHeadDTO(
                    head_number=t.head_number,
                    head_title=t.head_title,
                    charge_text=t.charge_text
                )
                for t in form.tables
            ],
            provenance_citation=form.provenance_citation
        )


_forms_service_instance: Optional[StatutoryFormsService] = None


def get_forms_service() -> StatutoryFormsService:
    """Singleton provider for StatutoryFormsService."""
    global _forms_service_instance
    if _forms_service_instance is None:
        _forms_service_instance = StatutoryFormsService(
            source_pdf_path=None
        )
    return _forms_service_instance

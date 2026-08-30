"""Statutory Forms Application Service (Phase 8).

Coordinates deterministic statutory form retrieval, disambiguation, and DTO mapping.
"""

import logging
from typing import Optional

from backend.app.api.errors import NotFoundError
from backend.app.api.schemas.forms import (
    FormFieldDTO,
    FormLookupResponseDTO,
    FormSignatureDTO,
    FormTableHeadDTO,
    StatutoryFormDTO,
)
from backend.app.forms.lookup import DeterministicFormIdentifier
from backend.app.forms.models import StatutoryForm
from backend.app.forms.renderer import DeterministicFormRenderer
from backend.app.forms.repository import StatutoryFormRegistry, get_form_registry

logger = logging.getLogger("nyaya.services.forms")


class StatutoryFormsService:
    """Application service for Second Schedule statutory forms."""

    def __init__(self, registry: Optional[StatutoryFormRegistry] = None):
        self.registry = registry or get_form_registry()
        self.identifier = DeterministicFormIdentifier(registry=self.registry)
        self.renderer = DeterministicFormRenderer()

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
        _forms_service_instance = StatutoryFormsService()
    return _forms_service_instance

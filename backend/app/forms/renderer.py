"""Deterministic Statutory Form Renderer for Nyaya Legal RAG (Phase 7).

Generates formatted Markdown and plain text representations of statutory forms
with ZERO LLM dependency, grounded entirely in the validated StatutoryForm model.
"""

from typing import Any, Dict
from backend.app.forms.models import StatutoryForm, FormFieldType


class DeterministicFormRenderer:
    """Renders structured StatutoryForm instances into presentation formats."""

    @staticmethod
    def render_markdown(form: StatutoryForm) -> str:
        """Render a statutory form into clean GitHub-flavored Markdown."""
        lines = [
            f"# FORM No. {form.form_number}",
            f"## {form.form_title}",
            ""
        ]

        if form.applicable_sections:
            secs_str = ", ".join(f"Section {s}" for s in form.applicable_sections)
            lines.append(f"*(See {secs_str} of the Bharatiya Nagarik Suraksha Sanhita, 2023)*")
            lines.append("")

        lines.append(f"**Canonical Citation**: `{form.provenance_citation}`  ")
        lines.append(f"**Source Pages**: Gazette PDF Pages {form.page_start}–{form.page_end}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Render structured fields if present
        if form.fields:
            lines.append("### Form Fields & Placeholders")
            for f in form.fields:
                lines.append(f"- **{f.label}** (`{f.field_type.value}`): `{f.placeholder or '............'}`")
            lines.append("")

        # Render multi-head charge tables if present (Form 33)
        if form.tables:
            lines.append("### Charge Heads")
            for t in form.tables:
                lines.append(f"#### Head {t.head_number}: {t.head_title}")
                lines.append(f"> {t.charge_text}")
                lines.append("")

        # Render verbatim statutory body text
        lines.append("### Statutory Text")
        lines.append("```text")
        lines.append(form.raw_text.strip())
        lines.append("```")
        lines.append("")

        if form.signatures:
            lines.append("### Execution & Seal Requirements")
            for sig in form.signatures:
                lines.append(f"- {sig.signatory_title}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def render_text(form: StatutoryForm) -> str:
        """Render a concise plain-text representation."""
        secs = ", ".join(form.applicable_sections) if form.applicable_sections else "N/A"
        return (
            f"FORM No. {form.form_number}: {form.form_title}\n"
            f"Applicable Sections: {secs}\n"
            f"Provenance: {form.provenance_citation} (Pages {form.page_start}-{form.page_end})\n\n"
            f"{form.raw_text}"
        )

    @staticmethod
    def render_summary(form: StatutoryForm) -> Dict[str, Any]:
        """Render a JSON-serializable structured summary."""
        return {
            "form_id": form.form_id,
            "form_number": form.form_number,
            "form_title": form.form_title,
            "applicable_sections": form.applicable_sections,
            "page_range": [form.page_start, form.page_end],
            "provenance_citation": form.provenance_citation,
            "fields_count": len(form.fields),
            "signatures_count": len(form.signatures),
            "has_tables": len(form.tables) > 0
        }

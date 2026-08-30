"""Unit and Integration Tests for Statutory Forms PDF Export & Manifest (Part B)."""

import hashlib
import json
import os
import tempfile
import zipfile
import pypdf
import pytest

from backend.app.forms.exporter import (
    StatutoryFormExporter,
    calculate_extraction_confidence,
    get_form_filename,
    slugify_form_title,
)
from backend.app.forms.parser import SecondScheduleParser


def test_slugify_form_title():
    """Verify deterministic, collision-free, filesystem-safe slugification."""
    assert slugify_form_title("NOTICE FOR APPEARANCE BY THE POLICE") == "Notice-for-Appearance-by-the-Police"
    assert slugify_form_title("BOND AND BAIL-BOND AFTER ARREST UNDER A WARRANT") == "Bond-and-Bail-Bond-after-Arrest-under-a-Warrant"
    assert slugify_form_title("CHARGES") == "Charges"
    assert slugify_form_title("58: WARRANT OF IMPRISONMENT") == "Warrant-of-Imprisonment"
    # Traversal characters removed
    assert slugify_form_title("../../etc/passwd") == "Etcpasswd"


def test_get_form_filename():
    """Verify filename format matches FORM-<number>_<slugified-title>.pdf."""
    fname = get_form_filename(12, "Bond and Bail-Bond for Attendance before Court")
    assert fname == "FORM-12_Bond-and-Bail-Bond-for-Attendance-before-Court.pdf"
    assert not any(c in fname for c in [" ", "/", "\\", ".."])


def test_calculate_extraction_confidence():
    """Verify deterministic confidence calculation and needs_review flag."""
    parser = SecondScheduleParser(pdf_path="BNS bare act 2023.pdf")
    forms = parser.parse_forms()
    form_1 = [f for f in forms if f.form_number == 1][0]
    conf_1, needs_rev_1 = calculate_extraction_confidence(form_1)
    assert 0.85 <= conf_1 <= 1.0
    assert needs_rev_1 is False

    form_33 = [f for f in forms if f.form_number == 33][0]
    conf_33, needs_rev_33 = calculate_extraction_confidence(form_33)
    assert 0.85 <= conf_33 <= 1.0
    assert needs_rev_33 is False


def test_statutory_forms_export_pipeline():
    """Test full export pipeline creating 58 PDF files and forms_manifest.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = StatutoryFormExporter(pdf_path="BNS bare act 2023.pdf", output_dir=tmpdir)
        manifest = exporter.export_all()

        assert manifest["total_forms"] == 58
        assert len(manifest["forms"]) == 58

        # 1. Verify manifest file exists on disk
        manifest_file = os.path.join(tmpdir, "forms_manifest.json")
        assert os.path.exists(manifest_file)
        with open(manifest_file, "r") as f:
            disk_manifest = json.load(f)
        assert disk_manifest["total_forms"] == 58

        # 2. Verify all 58 PDF files exist on disk
        for entry in disk_manifest["forms"]:
            assert 1 <= entry["form_number"] <= 58
            assert len(entry["title"]) >= 3
            assert entry["filename"].startswith(f"FORM-{entry['form_number']}_")
            assert entry["filename"].endswith(".pdf")
            assert entry["byte_size"] > 0
            assert len(entry["sha256"]) == 64
            assert 0.0 <= entry["extraction_confidence"] <= 1.0
            assert isinstance(entry["needs_review"], bool)

            pdf_path = os.path.join(tmpdir, entry["filename"])
            assert os.path.exists(pdf_path)

            # Check SHA-256 matches actual file
            with open(pdf_path, "rb") as pf:
                actual_hash = hashlib.sha256(pf.read()).hexdigest()
            assert actual_hash == entry["sha256"]

        # 3. Specifically verify Form 33 has 3 pages
        f33_entry = [e for e in disk_manifest["forms"] if e["form_number"] == 33][0]
        assert f33_entry["page_start"] == 222
        assert f33_entry["page_end"] == 224
        assert f33_entry["page_count"] == 3

        f33_reader = pypdf.PdfReader(os.path.join(tmpdir, f33_entry["filename"]))
        assert len(f33_reader.pages) == 3


def test_statutory_forms_export_idempotency():
    """Verify that running export twice produces byte-identical files and hashes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = StatutoryFormExporter(pdf_path="BNS bare act 2023.pdf", output_dir=tmpdir)
        manifest_1 = exporter.export_all()
        manifest_2 = exporter.export_all()

        assert manifest_1["total_forms"] == manifest_2["total_forms"]
        for f1, f2 in zip(manifest_1["forms"], manifest_2["forms"]):
            assert f1["form_number"] == f2["form_number"]
            assert f1["sha256"] == f2["sha256"]
            assert f1["byte_size"] == f2["byte_size"]
            assert f1["filename"] == f2["filename"]

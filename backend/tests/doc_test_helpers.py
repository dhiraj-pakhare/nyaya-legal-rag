"""Test helper utilities for generating sample PDF files for Phase 6 tests."""

import io
from typing import List, Optional
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter


def create_test_pdf_bytes(page_texts: List[str], password: Optional[str] = None) -> bytes:
    """Generate in-memory PDF bytes with the given text on each page, optionally encrypted."""
    packet = io.BytesIO()
    can = canvas.Canvas(packet)

    for text in page_texts:
        # Draw lines of text
        lines = text.split("\n")
        y = 750
        for line in lines:
            can.drawString(50, y, line)
            y -= 20
        can.showPage()

    can.save()
    packet.seek(0)
    pdf_bytes = packet.getvalue()

    if password:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt(password)
        encrypted_stream = io.BytesIO()
        writer.write(encrypted_stream)
        return encrypted_stream.getvalue()

    return pdf_bytes

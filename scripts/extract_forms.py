#!/usr/bin/env python3
"""CLI Script for Extracting Second Schedule Statutory Forms (Part B).

Usage:
    python scripts/extract_forms.py [--pdf-path "BNS bare act 2023.pdf"] [--output-dir data/forms]
"""

import argparse
import logging
import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.forms.exporter import StatutoryFormExporter
from backend.app.forms.parser import SecondScheduleParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("extract_forms")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Second Schedule Statutory Forms into individual PDFs and manifest."
    )
    parser.add_argument(
        "--pdf-path",
        default="BNS bare act 2023.pdf",
        help="Path to source BNSS Gazette PDF"
    )
    parser.add_argument(
        "--output-dir",
        default="data/forms",
        help="Output directory for form PDFs and manifest"
    )
    args = parser.parse_args()

    logger.info(f"Starting Statutory Forms extraction from '{args.pdf_path}' into '{args.output_dir}'...")
    forms_parser = SecondScheduleParser(pdf_path=args.pdf_path)
    forms = forms_parser.parse_forms()
    logger.info(f"Parsed {len(forms)} statutory forms with invariant checks passed.")

    exporter = StatutoryFormExporter(pdf_path=args.pdf_path, output_dir=args.output_dir)
    manifest = exporter.export_all(forms=forms)
    logger.info(f"Extraction complete! Manifest written with {manifest['total_forms']} forms at {args.output_dir}/forms_manifest.json.")


if __name__ == "__main__":
    main()

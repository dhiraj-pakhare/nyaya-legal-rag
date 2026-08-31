#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# Nyaya Legal RAG — System Bootstrap & Initialization Script
#
# Idempotently verifies and performs:
# 1. Statutory Gazette PDF verification.
# 2. Statutory corpus ingestion & Qdrant vector indexing (1,027 canonical points).
# 3. Second Schedule statutory forms extraction (58 forms) into data/forms/.
#
# Safe to run repeatedly: skips steps that are already completed.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PDF_FILE="${PDF_PATH:-BNS bare act 2023.pdf}"
export QDRANT_TARGET_URL="${QDRANT_URL:-http://localhost:6333}"
FORMS_OUTPUT_DIR="${FORMS_DIR:-data/forms}"

echo "========================================================"
echo "      Nyaya Legal RAG — System Initialization          "
echo "========================================================"
echo "Repository Root: ${ROOT_DIR}"
echo "Source PDF:      ${PDF_FILE}"
echo "Qdrant URL:      ${QDRANT_TARGET_URL}"
echo "Forms Directory: ${FORMS_OUTPUT_DIR}"
echo "--------------------------------------------------------"

# 1. Verify Source PDF
if [ ! -f "${PDF_FILE}" ]; then
  echo "[ERROR] Source Gazette PDF not found at '${PDF_FILE}'."
  echo "Please place 'BNS bare act 2023.pdf' in the repository root or mount it into the container."
  exit 1
fi
echo "[OK] Source Gazette PDF found."

# 2. Check Qdrant Statutory Collection Status
echo "[INFO] Inspecting Qdrant statutory collection..."

set +e
python3 - << 'EOF'
import os
import sys

from backend.app.core.config import settings
from backend.app.core.qdrant_repo import QdrantRepository

target_url = os.getenv("QDRANT_TARGET_URL") or os.getenv("QDRANT_URL", "http://localhost:6333")
collection_name = settings.qdrant_collection

try:
    repo = QdrantRepository(url=target_url, collection_name=collection_name)
    count = repo.count()
    print(f"[INFO] Current collection point count: {count}")
    if count >= 1027:
        print("[OK] Statutory corpus already initialized (1,027 points present). Skipping ingestion.")
        sys.exit(0)
    else:
        print(f"[INFO] Collection has {count} points (< 1,027). Ingestion required.")
        sys.exit(10)
except Exception as e:
    print(f"[WARN] Unable to read existing collection ({e}). Proceeding to ingest...")
    sys.exit(10)
EOF
INGEST_CHECK_CODE=$?
set -e

if [ ${INGEST_CHECK_CODE} -eq 10 ]; then
  echo "[INFO] Executing statutory ingestion pipeline..."
  python3 scripts/ingest.py --pdf-path "${PDF_FILE}" --qdrant-url "${QDRANT_TARGET_URL}"
  echo "[OK] Statutory ingestion completed successfully."
elif [ ${INGEST_CHECK_CODE} -eq 0 ]; then
  echo "[OK] Statutory collection verified."
else
  echo "[ERROR] Diagnostic check against Qdrant failed with code ${INGEST_CHECK_CODE}."
  exit 1
fi

# 3. Statutory Forms Extraction Status
echo "[INFO] Inspecting Second Schedule statutory forms status..."
MANIFEST_FILE="${FORMS_OUTPUT_DIR}/forms_manifest.json"

EXTRACT_REQUIRED=1
if [ -f "${MANIFEST_FILE}" ]; then
  if grep -q '"total_forms": 58' "${MANIFEST_FILE}" 2>/dev/null; then
    EXTRACT_REQUIRED=0
    echo "[OK] Statutory forms already extracted (${MANIFEST_FILE} contains 58 forms). Skipping extraction."
  fi
fi

if [ ${EXTRACT_REQUIRED} -eq 1 ]; then
  echo "[INFO] Extracting statutory forms to '${FORMS_OUTPUT_DIR}'..."
  mkdir -p "${FORMS_OUTPUT_DIR}"
  python3 scripts/extract_forms.py --pdf-path "${PDF_FILE}" --output-dir "${FORMS_OUTPUT_DIR}"
  echo "[OK] Statutory forms extraction completed successfully."
fi

echo "--------------------------------------------------------"
echo "[SUCCESS] Nyaya Legal RAG bootstrap completed cleanly."
echo "========================================================"
exit 0

"""Text Cleaning and Gazette Artifact Filtering for Nyaya Legal RAG.

Provides utilities to strip Gazette running headers, footers, barcode colophons,
underscore separators, and clean hyphenated line breaks without losing legal context.
"""

import re
from typing import List

# Patterns of Gazette running headers and footers to filter out
GAZETTE_HEADER_FOOTER_PATTERNS = [
    r"^THE\s+GAZETTE\s+OF\s+INDIA\s+EXTRAORDINARY.*$",
    r"^\[Part\s+II—.*$",
    r"^Sec\.\s*1\].*$",
    r"^PART\s+II\s*—\s*Section\s*1.*$",
    r"^vlk/kkj\.k.*$",
    r"^EXTRAORDINARY.*$",
    r"^Hkkx\s+II.*$",
    r"^izkf/kdkj\s+ls\s+izdkf'kr.*$",
    r"^PUBLISHED\s+BY\s+AUTHORITY.*$",
    r"^lañ\s+54\].*$",
    r"^No\.\s*54\].*$",
    r"^bl\s+Hkkx\s+esa.*$",
    r"^Separate\s+paging\s+is\s+given.*$",
    r"^xxxGID[A-Z]+xxx.*$",
    r"^jftLVªh\s+lañ.*$",
    r"^REGISTERED\s+N\s*O\.\s*DL—.*$",
    r"^सी\.जी\.-डी\.एल\.-.*$",
    r"^CG-DL-E-.*$",
    r"^_{10,}.*$",  # 10 or more underscores
    r"^—{3,}.*$",   # 3 or more em-dashes
    r"^MGIPMRND—.*$",
    r"^UPLOADED\s+BY\s+THE\s+MANAGER.*$",
    r"^AND\s+PUBLISHED\s+BY\s+THE\s+CONTROLLER.*$",
]

COMPILED_GAZETTE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in GAZETTE_HEADER_FOOTER_PATTERNS]

# Known legal compound words that MUST retain their hyphen
PRESERVED_HYPHEN_WORDS = {
    "audio-video", "non-cognizable", "non-bailable", "bail-bond", "sub-section",
    "sub-clause", "cross-examination", "re-examination", "quasi-judicial",
    "ex-parte", "suo-motu", "first-class", "second-class", "court-fee",
    "time-limit", "part-heard", "well-founded", "bona-fide"
}


def is_gazette_boilerplate(line: str) -> bool:
    """Check if a line is Gazette boilerplate header, footer, or colophon."""
    stripped = line.strip()
    if not stripped:
        return False
    
    # Check standalone page numbers (digits only, length <= 3)
    if stripped.isdigit() and len(stripped) <= 3:
        return True
        
    for pat in COMPILED_GAZETTE_PATTERNS:
        if pat.match(stripped):
            return True
            
    return False


def clean_gazette_lines(lines: List[str]) -> List[str]:
    """Filter out Gazette boilerplate lines from raw extracted lines."""
    cleaned = []
    for line in lines:
        if not is_gazette_boilerplate(line):
            cleaned.append(line.rstrip())
    return cleaned


def dehyphenate_text(text: str) -> str:
    """Join hyphenated line breaks cleanly while preserving genuine compound words.
    
    Example:
        'inves-\ntigation' -> 'investigation'
        'audio-\nvideo' -> 'audio-video'
        'non-\ncognizable' -> 'non-cognizable'
    """
    lines = text.splitlines()
    if not lines:
        return ""
        
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if line ends with a trailing hyphen
        if line.rstrip().endswith("-") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Extract word before hyphen
            last_word_match = re.search(r'([A-Za-z0-9]+)-$', line.rstrip())
            first_word_match = re.match(r'^([A-Za-z0-9]+)', next_line)
            
            if last_word_match and first_word_match:
                w1 = last_word_match.group(1)
                w2 = first_word_match.group(1)
                combined = f"{w1}-{w2}".lower()
                
                # Check if it should stay hyphenated or be joined
                if combined in PRESERVED_HYPHEN_WORDS or w1.isupper():
                    joined_word = f"{w1}-{w2}"
                else:
                    joined_word = f"{w1}{w2}"
                    
                # Replace the split in the lines
                line_prefix = line.rstrip()[:last_word_match.start(1)]
                next_line_suffix = next_line[first_word_match.end(1):]
                
                merged_line = f"{line_prefix}{joined_word}{next_line_suffix}".strip()
                result_lines.append(merged_line)
                i += 2
                continue
                
        result_lines.append(line)
        i += 1
        
    return "\n".join(result_lines)


def normalize_whitespace(text: str) -> str:
    """Standardize spaces, tabs, and multiple blank lines."""
    # Replace weird unicode spaces with regular space
    text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F]', ' ', text)
    # Strip spaces before newlines
    text = re.sub(r'[ \t]+(?=\n)', '', text)
    # Normalize multiple horizontal spaces to single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize 3+ newlines to 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_statutory_text(raw_text: str) -> str:
    """Run full cleaning pipeline on raw extracted statutory text."""
    lines = raw_text.splitlines()
    cleaned_lines = clean_gazette_lines(lines)
    filtered_text = "\n".join(cleaned_lines)
    dehyphenated = dehyphenate_text(filtered_text)
    return normalize_whitespace(dehyphenated)

"""Statutory Embedding Representation Formatter for Nyaya Legal RAG.

Formats StatutoryChunk objects into canonical text representations for dense vector embedding,
preserving statutory identity, chapter context, and section headers without excessive duplication.
"""

from backend.app.ingestion.models import ChunkType, StatutoryChunk


def format_chunk_for_embedding(chunk: StatutoryChunk) -> str:
    """Construct the canonical text representation for dense embedding.
    
    Substantive Sections:
        [<act_short>] Chapter <chapter>: <chapter_title> | Section <section_number>: <section_title>
        <chunk_text>
        
    First Schedule Entries:
        [<act_short>] The First Schedule - Classification of Offences | Section <section_number>: <section_title>
        Offence: <offence_name>
        Punishment: <punishment>
        Classification: <cognizable_status> | <bailable_status>
        Triable by: <triable_court>
    """
    if chunk.chunk_type == ChunkType.SCHEDULE_ENTRY.value:
        return (
            f"[{chunk.act_short}] The First Schedule - Classification of Offences | "
            f"Section {chunk.section_number}: {chunk.section_title}\n"
            f"Offence: {chunk.offence_name or chunk.section_title}\n"
            f"Punishment: {chunk.punishment or 'As provided in Sanhita'}\n"
            f"Classification: {chunk.cognizable_status or 'Cognizable'} | {chunk.bailable_status or 'Non-bailable'}\n"
            f"Triable by: {chunk.triable_court or 'Court of Session'}"
        )
        
    # Substantive Section:
    header = f"[{chunk.act_short}] Chapter {chunk.chapter or 'I'}: {chunk.chapter_title or 'PRELIMINARY'} | Section {chunk.section_number}: {chunk.section_title}"
    if chunk.text.startswith(f"[{chunk.act_short} s.{chunk.section_number}:"):
        # Text already contains breadcrumb from chunker, just prepend act name if needed
        return chunk.text
    else:
        return f"{header}\n{chunk.text}"

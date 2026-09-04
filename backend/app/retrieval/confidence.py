"""Multi-factor confidence scoring and refusal decision engine."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.retrieval.models import RetrievedDocument

STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "this", "that", "with", "from",
    "under", "into", "over", "after", "than", "about", "other", "some", "such", "only",
    "same", "have", "been", "made", "each", "more", "then", "them", "these", "their",
    "there", "does", "tell", "explain", "give", "details", "provision", "provisions",
    "part", "court", "section", "order", "test", "title"
}

_COMMON_ENGLISH_WORDS = {
    # Pronouns, determiners, prepositions, conjunctions, verbs, question words
    "a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along", "already",
    "also", "although", "always", "am", "among", "an", "and", "another", "any", "anybody", "anyone", "anything",
    "anywhere", "are", "area", "around", "as", "ask", "at", "away", "back", "be", "became", "because", "become",
    "been", "before", "began", "behind", "being", "below", "best", "better", "between", "beyond", "big", "both",
    "but", "by", "came", "can", "cannot", "case", "cases", "certain", "clear", "come", "could", "did", "do",
    "does", "done", "down", "during", "each", "early", "either", "else", "end", "even", "ever", "every", "everyone",
    "everything", "everywhere", "face", "fact", "far", "felt", "few", "find", "first", "for", "form", "from",
    "further", "gave", "general", "get", "give", "given", "go", "good", "got", "great", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "however", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "keep", "kept", "knew", "know", "known", "large", "last",
    "later", "least", "less", "let", "like", "likely", "little", "long", "look", "made", "make", "making", "many",
    "may", "me", "might", "more", "most", "much", "must", "my", "myself", "near", "need", "never", "new", "next",
    "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "number", "of", "off", "often",
    "old", "on", "once", "one", "only", "or", "order", "other", "others", "our", "ours", "ourselves", "out",
    "over", "own", "part", "per", "place", "point", "possible", "present", "public", "put", "quite", "rather",
    "really", "right", "said", "same", "saw", "say", "saying", "see", "seem", "seemed", "seeming", "seems", "seen",
    "several", "shall", "she", "should", "show", "side", "since", "small", "so", "some", "somebody", "someone",
    "something", "somewhere", "state", "states", "still", "such", "sure", "take", "taken", "taking", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "these", "they", "thing", "things",
    "think", "this", "those", "though", "thought", "three", "through", "thus", "time", "to", "together", "too",
    "took", "toward", "towards", "two", "under", "until", "up", "upon", "us", "use", "used", "using", "very", "want",
    "was", "way", "we", "well", "went", "were", "what", "whatever", "when", "whenever", "where", "wherever", "whether",
    "which", "while", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would",
    "year", "years", "yet", "you", "your", "yours", "yourself", "yourselves",
    # Legal, statutory, procedural, constitutional, and governance vocabulary
    "act", "acts", "action", "accused", "alleged", "allegation", "amendment", "applicable", "apply", "applies",
    "appeal", "appearance", "apprehend", "apprehended", "arrest", "arrested", "arresting", "article", "assistance",
    "attorney", "audio", "authority", "authorized", "available", "bail", "bailable", "non-bailable", "bare",
    "belonging", "bench", "bill", "bind", "bns", "bnss", "bsa", "body", "breach", "bribe", "bribery", "burden",
    "california", "calling", "cause", "causing", "charge", "charges", "cheating", "chief", "citizen", "claim",
    "clause", "clerk", "closed", "code", "codes", "cognizable", "non-cognizable", "commit", "committed", "committing",
    "common", "compel", "compensation", "competent", "complaint", "comply", "compliance", "condition", "conduct",
    "confession", "confinement", "consent", "consequence", "constitution", "constitutional", "contempt", "continue",
    "contract", "convict", "convicted", "conviction", "copyright", "corporate", "corporation", "counsel", "court",
    "crime", "criminal", "crpc", "culpable", "custody", "customs", "damage", "damages", "date", "death", "deceased",
    "declaration", "decree", "deemed", "defendant", "defense", "defined", "definition", "definitions", "delay",
    "deliver", "demand", "depute", "deputy", "detain", "detained", "detention", "device", "direction", "discharge",
    "disclose", "discovery", "dismiss", "disobey", "dispute", "district", "document", "documents", "duty", "electronic",
    "enact", "enacted", "enactment", "enforce", "enforcement", "entered", "entering", "entry", "error", "escape",
    "evidence", "examination", "examine", "execution", "executive", "exemption", "explanation", "export", "extend",
    "failure", "false", "fee", "female", "fifa", "file", "filed", "filing", "final", "fine", "fined", "first",
    "force", "foreign", "fourth", "fraud", "fraudulent", "freedom", "fund", "further", "general", "good", "goods",
    "government", "governor", "grant", "granted", "grievous", "ground", "grounds", "group", "guilt", "guilty",
    "hearing", "heir", "high", "homicide", "house", "hurt", "identification", "illegal", "imprisonment", "imported",
    "income", "india", "indian", "indictment", "inform", "information", "informed", "infirm", "ingress", "initiation",
    "injunction", "injury", "inquiry", "inspection", "inspector", "instance", "instrument", "insufficient", "intent",
    "intention", "intentional", "internal", "international", "interpretation", "investigate", "investigation", "ipc",
    "issue", "issued", "jail", "jaywalking", "judge", "judgment", "judicial", "judiciary", "jurisdiction", "justice",
    "juvenile", "kidnap", "kidnapping", "killing", "knowledge", "law", "lawful", "laws", "lawyer", "legal",
    "legislation", "legislative", "liability", "liable", "life", "limitation", "list", "local", "lynching",
    "magistrate", "maintenance", "mandatory", "manner", "marriage", "match", "material", "matter", "means",
    "medical", "memorandum", "mens", "minor", "mischief", "misconduct", "mob", "mobile", "mode", "money", "month",
    "months", "motion", "movable", "municipal", "murder", "nagarik", "name", "nature", "negligence", "negligent",
    "negotiable", "notice", "notification", "null", "nyaya", "oath", "object", "objection", "obtain", "obtained",
    "occupant", "offence", "offences", "offender", "offense", "offenses", "officer", "official", "ohio", "omission",
    "open", "opinion", "order", "orders", "ordinance", "outside", "owner", "ownership", "paper", "pardon", "parent",
    "party", "parties", "pass", "passed", "peace", "penalty", "penalties", "pending", "period", "permission",
    "person", "persons", "petition", "petitioner", "phone", "place", "plea", "plead", "police", "possession",
    "power", "powers", "practice", "precedent", "preliminary", "premise", "premises", "presence", "president",
    "presumption", "prevention", "prima", "primary", "principal", "prison", "prisoner", "private", "privilege",
    "procedure", "proceeding", "proceedings", "process", "produce", "production", "prohibited", "prohibition",
    "proof", "property", "prosecution", "protect", "protection", "provisions", "punish", "punishable", "punished",
    "punishment", "purpose", "pursuant", "rate", "rates", "rea", "reasonable", "reason", "reasons", "receipt",
    "receive", "record", "recording", "recovery", "refusal", "refuse", "register", "registered", "registration",
    "regulation", "regulations", "release", "released", "relevant", "relief", "remedy", "remand", "removal",
    "repeal", "report", "reported", "require", "required", "requirement", "requirements", "requisite", "residence",
    "resident", "resolution", "respect", "respondent", "responsibility", "restitution", "restoration", "restriction",
    "revenue", "review", "revocation", "right", "rights", "rule", "rules", "ruling", "sanhita", "satisfaction",
    "schedule", "scheme", "scored", "search", "searched", "searches", "second", "section", "sections", "secure",
    "securing", "security", "seize", "seized", "seizure", "sentence", "service", "session", "sessions", "settlement",
    "shall", "sheriff", "smartphones", "special", "specific", "state", "statement", "station", "statute", "statutes",
    "statutory", "stay", "stipulation", "subordinate", "sub-section", "subsection", "suit", "summary", "summons",
    "superintendent", "supreme", "suraksha", "surety", "surrender", "suspect", "suspected", "suspicion", "taking",
    "tax", "taxation", "taxes", "tenant", "term", "terms", "territory", "testimony", "theft", "third", "threat",
    "time", "title", "tort", "traffic", "transfer", "transmission", "treaties", "treaty", "trial", "tribunal",
    "unauthorized", "unconstitutional", "undertaking", "unlawful", "unnecessary", "us", "valid", "validity", "vehicle",
    "venue", "verdict", "vessel", "victim", "video", "violation", "void", "voluntary", "waiver", "warrant", "warrantless",
    "weapons", "willful", "witness", "witnesses", "won", "world", "writ", "wrongful"
}


def _is_multilingual_or_transliterated(query: str) -> bool:
    """Detect if a search query is in non-English / Indic script or contains transliterated terms."""
    if not query or not query.strip():
        return False
    # Check for non-ASCII Unicode (e.g. Devanagari, Gurmukhi, Tamil, etc.)
    if any(ord(c) >= 128 for c in query):
        return True
    # Tokenize into Latin alphabetic words
    tokens = re.findall(r"[a-zA-Z]+", query.lower())
    valid_tokens = [t for t in tokens if len(t) > 1]
    if not valid_tokens:
        return False
    non_english = [t for t in valid_tokens if t not in _COMMON_ENGLISH_WORDS]
    return (len(non_english) / len(valid_tokens)) >= 0.30


def _get_base_section(sec_str: str) -> str:
    m = re.match(r"^([0-9]+[a-zA-Z]?)", (sec_str or "").strip())
    return m.group(1).lower() if m else (sec_str or "").strip().lower()


def _are_in_same_statutory_cluster(d1: RetrievedDocument, d2: RetrievedDocument) -> bool:
    """Determine if two retrieved documents belong to the same statutory cluster or topic."""
    # Same Act and base section (e.g. BNS s.103(1) and BNS s.103(2))
    if d1.act_short and d2.act_short and d1.act_short == d2.act_short:
        b1 = _get_base_section(d1.section_number)
        b2 = _get_base_section(d2.section_number)
        if b1 and b2 and b1 == b2:
            return True

    # Shared substantive legal roots in section titles across acts (e.g. BNS offence + BNSS charge illustration)
    t1 = set(re.findall(r"\b[a-zA-Z]{4,}\b", (d1.section_title or "").lower())) - STOP_WORDS
    t2 = set(re.findall(r"\b[a-zA-Z]{4,}\b", (d2.section_title or "").lower())) - STOP_WORDS
    return len(t1.intersection(t2)) > 0


class ConfidenceResult(BaseModel):
    """Structured confidence evaluation and refusal decision."""
    confidence_score: float = Field(..., description="Calibrated confidence score in [0.0, 1.0]")
    decision: str = Field(..., description="'ACCEPT' or 'REFUSE'")
    threshold: float = Field(..., description="Active threshold evaluated against")
    reason: str = Field(..., description="Detailed categorical reason for decision")
    top_result_score: float = Field(default=0.0)
    score_margin: float = Field(default=0.0)
    retrieval_evidence: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceScorer:
    """Evaluates multi-factor retrieval signals to determine confidence and refusal."""

    def __init__(self, threshold: float = settings.confidence_threshold):
        self.threshold = threshold

    def evaluate(
        self,
        query: str,
        documents: List[RetrievedDocument],
        mode: str = "hybrid_rrf",
        detected_intent: Optional[Dict[str, Any]] = None,
        override_threshold: Optional[float] = None
    ) -> ConfidenceResult:
        """Compute multi-factor confidence and produce an ACCEPT/REFUSE decision.
        
        Factors evaluated:
        1. Exact deterministic section match (score = 1.0)
        2. Top candidate relevance score (calibrated for legal domain & candidate pool)
        3. Tie-aware discriminative margin between distinct statutory clusters
        4. Cross-retriever agreement (Dense + BM25 alignment on target cluster)
        5. Candidate pool statutory coherence
        """
        active_threshold = override_threshold if override_threshold is not None else self.threshold

        # Case 1: Empty retrieval
        if not documents:
            if detected_intent and detected_intent.get("is_exact_lookup"):
                sec = detected_intent.get("section_number")
                act = detected_intent.get("act_short") or "Statute"
                return ConfidenceResult(
                    confidence_score=0.0,
                    decision="REFUSE",
                    threshold=active_threshold,
                    reason="exact_section_not_found",
                    top_result_score=0.0,
                    score_margin=0.0,
                    retrieval_evidence={
                        "total_documents": 0,
                        "query": query,
                        "missing_section": f"{act} Section {sec}"
                    }
                )
            return ConfidenceResult(
                confidence_score=0.0,
                decision="REFUSE",
                threshold=active_threshold,
                reason="no_retrieval_results",
                top_result_score=0.0,
                score_margin=0.0,
                retrieval_evidence={"total_documents": 0, "query": query}
            )

        top_doc = documents[0]

        # Case 2: Exact deterministic section lookup
        if top_doc.is_exact_match:
            return ConfidenceResult(
                confidence_score=1.0,
                decision="ACCEPT",
                threshold=active_threshold,
                reason="exact_section_match",
                top_result_score=1.0,
                score_margin=1.0,
                retrieval_evidence={
                    "mode": "exact_lookup",
                    "matched_section": f"{top_doc.act_short} s.{top_doc.section_number}",
                    "chunk_id": top_doc.chunk_id
                }
            )

        # Case 3: Statistical multi-factor confidence for normal search
        raw_logits = [
            d.metadata.get("reranker_raw_score")
            for d in documents
            if d.metadata.get("reranker_raw_score") is not None
        ]
        has_logits = len(raw_logits) == len(documents) and len(raw_logits) > 0
        top_raw = raw_logits[0] if has_logits else None

        # Factor A: Calibrated Cross-Encoder / Relevance Score
        if has_logits and top_raw is not None:
            # Legal cross-encoder noise floor is ~ -10.5, solid relevance is >= -5.5
            s_ce = max(0.0, min(1.0, (top_raw - (-10.5)) / 6.0))
        else:
            s_ce = max(0.0, min(1.0, float(top_doc.score)))

        # Factor B: Tie-Aware Statutory Cluster Margin
        competing_doc = None
        for d in documents[1:]:
            if not _are_in_same_statutory_cluster(top_doc, d):
                competing_doc = d
                break

        if competing_doc is None:
            s_margin = 1.0
            margin_val = 1.0
        elif has_logits and top_raw is not None:
            comp_raw = competing_doc.metadata.get("reranker_raw_score", top_raw)
            margin_val = max(0.0, top_raw - comp_raw)
            s_margin = min(1.0, margin_val / 1.5)
        else:
            margin_val = max(0.0, float(top_doc.score) - float(competing_doc.score))
            s_margin = min(1.0, margin_val / 0.15)

        # Factor C: Dual-Retriever Agreement (Cluster-Aware)
        dense_ranks = [
            d.dense_rank
            for d in documents
            if d.dense_rank is not None and _are_in_same_statutory_cluster(d, top_doc)
        ]
        bm25_ranks = [
            d.bm25_rank
            for d in documents
            if d.bm25_rank is not None and _are_in_same_statutory_cluster(d, top_doc)
        ]
        best_dense = min(dense_ranks) if dense_ranks else top_doc.dense_rank
        best_bm25 = min(bm25_ranks) if bm25_ranks else top_doc.bm25_rank

        if best_dense is not None and best_bm25 is not None:
            if best_dense <= 5 and best_bm25 <= 5:
                s_agree = 1.0
            elif best_dense <= 10 and best_bm25 <= 10:
                s_agree = 0.90
            elif best_dense <= 20 and best_bm25 <= 20:
                s_agree = 0.80
            else:
                s_agree = 0.60
        elif best_dense is not None or best_bm25 is not None:
            single_best = best_dense if best_dense is not None else best_bm25
            if single_best <= 3:
                s_agree = 0.80
            elif single_best <= 10:
                s_agree = 0.60
            else:
                s_agree = 0.35
        else:
            s_agree = 0.15

        # Factor D: Candidate Pool Statutory Coherence
        coherent_with_top = sum(1 for d in documents[1:] if _are_in_same_statutory_cluster(top_doc, d))
        s_cohere = 1.0 if coherent_with_top >= 1 else 0.0

        is_multilingual = _is_multilingual_or_transliterated(query)

        # Composite Confidence Scoring
        if s_ce >= 0.70:
            # Authoritative cross-encoder signal
            raw_confidence = (0.45 * s_ce) + (0.25 * s_agree) + (0.15 * s_margin) + (0.15 * s_cohere)
        elif is_multilingual and s_cohere >= 0.80 and s_agree >= 0.75:
            # Multilingual / cross-lingual query: BM25/dense consensus & topical coherence ground the query
            raw_confidence = (0.45 * s_cohere) + (0.40 * s_agree) + (0.15 * s_margin)
        else:
            # Off-topic / ungrounded retrieval: bounded by weak relevance
            raw_confidence = (0.45 * s_ce) + (0.25 * s_agree) + (0.15 * s_margin) + (0.15 * s_cohere)
            if s_ce <= 0.20:
                raw_confidence = min(raw_confidence, s_ce * 2.0)

        confidence_score = round(max(0.0, min(1.0, raw_confidence)), 4)
        decision = "ACCEPT" if confidence_score >= active_threshold else "REFUSE"
        reason = "high_retrieval_confidence" if decision == "ACCEPT" else "low_retrieval_confidence"

        return ConfidenceResult(
            confidence_score=confidence_score,
            decision=decision,
            threshold=active_threshold,
            reason=reason,
            top_result_score=round(float(top_doc.score), 4),
            score_margin=round(margin_val, 4),
            retrieval_evidence={
                "top_chunk_id": top_doc.chunk_id,
                "top_section": f"{top_doc.act_short} s.{top_doc.section_number}",
                "dense_rank": top_doc.dense_rank,
                "bm25_rank": top_doc.bm25_rank,
                "best_cluster_dense_rank": best_dense,
                "best_cluster_bm25_rank": best_bm25,
                "s_ce": round(s_ce, 4),
                "agreement_score": round(s_agree, 4),
                "margin_score": round(s_margin, 4),
                "coherence_score": round(s_cohere, 4),
                "coherent_candidates": coherent_with_top,
                "top_raw_score": top_raw
            }
        )

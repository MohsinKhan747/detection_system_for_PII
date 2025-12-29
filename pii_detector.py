from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import spacy
except (ImportError, Exception):
    spacy = None


@dataclass
class PIIResult:
    """Container for PII detection results."""

    entities: Dict[str, List[Dict[str, object]]]
    highlighted_markdown: str


# Expanded stopwords to filter out common words that aren't names
_NAME_STOPWORDS = {
    "i", "we", "you", "he", "she", "it", "they", "me", "him", "her", "us", "them",
    "my", "our", "your", "his", "her", "its", "their","there",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must",
    "this", "that", "these", "those",
    "can", "from", "with", "about", "as", "by", "of",
    "not", "no", "yes", "ok", "okay",
}

# Pakistani cities and common location names to avoid detecting as personal names
_LOCATION_NAMES = {
    "islamabad", "rawalpindi", "lahore", "karachi", "peshawar", "quetta", 
    "multan", "faisalabad", "sialkot", "gujranwala", "hyderabad", "sukkur",
    "bahawalpur", "sargodha", "shekhupura", "jhang", "rahim yar khan",
    "pakistan", "punjab", "sindh", "balochistan", "kpk",
}

# Common titles and honorifics that often precede names
_NAME_TITLES = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "sir", "madam", "lord", "lady"
}


def _regex_patterns() -> Dict[str, re.Pattern]:
    """
    Structured PII patterns using regex for well-defined formats.
    """
    patterns = {
        # Username detection: looks for username keyword followed by value
        "USERNAME": re.compile(
            r"\b(?:username|user|userid|user[-_]?id|login)\s*[:=]\s*([a-zA-Z0-9_.-]{3,})",
            re.IGNORECASE,
        ),
        
        # Password detection: looks for password keyword followed by value
        "PASSWORD": re.compile(
            r"\b(?:password|passwd|pwd|pass)\s*[:=]\s*([^\s,;]{6,})",
            re.IGNORECASE,
        ),
        
        # Email detection
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        
        # Phone detection - more specific patterns
        "PHONE": re.compile(
            r"(?:\+92|0)?[-.\s]?3\d{2}[-.\s]?\d{7}|"  # Pakistani mobile
            r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b"  # General
        ),
        
        # IP Address
        "IP_ADDRESS": re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        
        # Pakistan CNIC pattern: 12345-1234567-1
        "NATIONAL_ID": re.compile(r"\b\d{5}-\d{7}-\d\b"),
        
        # Credit card - more specific (must be 13-16 digits with optional separators)
        "CREDIT_CARD": re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}|\b\d{13,16}\b"
        ),
        
        # Bank account patterns
        "BANK_ACCOUNT": re.compile(
            r"\b(?:account|acc|acct)[\s#:]*(\d{10,18})\b",
            re.IGNORECASE
        ),
        
        # Address patterns
        # Pakistani sectors: I10, F-11, G-9/4, etc. with city names
        "ADDRESS": re.compile(
            r"\b[A-Z]-?\d{1,2}(?:/\d{1,2})?\s*,?\s*(?:Islamabad|Rawalpindi|Lahore|Karachi|Peshawar|Quetta|Multan|Faisalabad|Sialkot|Gujranwala|Hyderabad)\b|"
            # Street addresses with numbers
            r"\b(?:house|flat|plot|street|st|road|rd|avenue|ave|block)[\s#:]*[A-Z0-9-/]+(?:\s*,?\s*[A-Z][a-z]+){1,3}\b|"
            # Explicit address context - captures the full address after the keyword
            r"\b(?:address|location|residence)\s+(?:is|:)\s+([A-Z0-9][^,.\n]{3,50}(?:,\s*[A-Z][a-z]+)?)|"
            r"\blive\s+(?:in|at)\s+([A-Z0-9][^,.\n]{3,50})",
            re.IGNORECASE
        ),
    }
    return patterns


def _is_valid_name(name: str) -> bool:
    """
    Check if a potential name is actually a valid personal name.
    Returns False for common words, pronouns, city names, etc.
    """
    # Convert to lowercase for checking
    name_lower = name.lower().strip()
    
    # Check against stopwords
    if name_lower in _NAME_STOPWORDS:
        return False
    
    # Check against location names
    if name_lower in _LOCATION_NAMES:
        return False
    
    # Check if all words are stopwords
    words = name_lower.split()
    if all(word in _NAME_STOPWORDS for word in words):
        return False
    
    # Check if it's a location name
    if all(word in _LOCATION_NAMES for word in words):
        return False
    
    # Reject common verbs and action words that aren't names
    common_verbs = {
        "reset", "set", "get", "make", "take", "give", "send", "help",
        "start", "stop", "run", "walk", "talk", "work", "play", "stay"
    }
    if name_lower in common_verbs or all(word in common_verbs for word in words):
        return False
    
    # Single letter is not a valid name
    if len(name.strip()) <= 1:
        return False
    
    # Must contain at least one letter
    if not any(c.isalpha() for c in name):
        return False
    
    # Reject if it's all uppercase and only one word (likely an acronym)
    if name.isupper() and len(words) == 1 and len(name) <= 4:
        return False
    
    return True


def _apply_regex(
    text: str, enabled_labels: List[str] | None = None
) -> List[Tuple[str, int, int]]:
    """
    Find all regex-based PII matches in the text.
    Returns a list of tuples: (label, start, end).
    """
    patterns = _regex_patterns()
    if enabled_labels:
        patterns = {k: v for k, v in patterns.items() if k in enabled_labels}

    matches: List[Tuple[str, int, int]] = []
    
    for label, pattern in patterns.items():
        for m in pattern.finditer(text):
            # For passwords, usernames, and bank accounts, only highlight the value
            if label in ("PASSWORD", "USERNAME", "BANK_ACCOUNT") and m.lastindex:
                matches.append((label, m.start(1), m.end(1)))
            elif label == "ADDRESS" and m.lastindex:
                # For explicit "address is/: value" format, highlight the captured value
                # Find the first non-None group (the captured address)
                for group_idx in range(1, m.lastindex + 1):
                    if m.group(group_idx):
                        matches.append((label, m.start(group_idx), m.end(group_idx)))
                        break
                else:
                    # If no group matched, use the whole match
                    matches.append((label, m.start(), m.end()))
            else:
                # Additional validation for credit cards
                if label == "CREDIT_CARD":
                    card_text = text[m.start():m.end()].replace(" ", "").replace("-", "")
                    # Must be valid length and pass basic checks
                    if not (13 <= len(card_text) <= 16 and card_text.isdigit()):
                        continue
                
                matches.append((label, m.start(), m.end()))
    
    return matches


def _apply_name_detection(text: str) -> List[Tuple[int, int]]:
    """
    Detect names using multiple heuristics:
    1. Names following "my name is" or "name is"
    2. Names following titles (Mr., Dr., etc.)
    3. Names following pronouns (me, him, her, etc.)
    4. Title case names that appear to be actual names
    
    Returns a list of (start, end) character indices for detected names.
    """
    spans: List[Tuple[int, int]] = []
    
    # Pattern 1: "my name is X" or "name is X" - case insensitive matching
    ctx_pattern = re.compile(
        r"\b(?:my\s+name\s+is|name\s+is|i\s+am|this\s+is|call\s+me|called)\s+"
        r"([A-Za-z][a-z]{1,}(?:\s+[A-Za-z][a-z]{1,}){0,3})",
        re.IGNORECASE,
    )
    
    for m in ctx_pattern.finditer(text):
        name = m.group(1)
        if _is_valid_name(name):
            spans.append((m.start(1), m.end(1)))
    
    # Pattern 2: Pronouns followed by names (me/him/her + name)
    pronoun_pattern = re.compile(
        r"\b(?:me|him|her|us)\s+"
        r"([A-Za-z][a-z]{1,}(?:\s+[A-Za-z][a-z]{1,}){0,3})\b",
        re.IGNORECASE
    )
    
    for m in pronoun_pattern.finditer(text):
        name = m.group(1)
        # More lenient validation for pronoun + name pattern
        if len(name.strip()) > 2 and name.lower() not in _NAME_STOPWORDS:
            spans.append((m.start(1), m.end(1)))
    
    # Pattern 3: Titles followed by names
    title_pattern = re.compile(
        r"\b(" + "|".join(_NAME_TITLES) + r")\.?\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
        re.IGNORECASE
    )
    
    for m in title_pattern.finditer(text):
        name = m.group(2)
        if _is_valid_name(name):
            spans.append((m.start(2), m.end(2)))
    
    # Pattern 4: Standalone capitalized names (more conservative)
    # Only match 2-4 word names to avoid false positives
    standalone_pattern = re.compile(
        r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})\b"
    )
    
    for m in standalone_pattern.finditer(text):
        name = m.group(1)
        # Additional checks for standalone names
        if _is_valid_name(name):
            # Check if it's at the start of a sentence or after common name contexts
            start = m.start()
            prefix = text[max(0, start-20):start].lower()
            
            # More likely to be a name if after certain words
            name_contexts = ["called", "named", "name", "signed", "from", "by", "dear", "me", "him", "her"]
            if any(ctx in prefix for ctx in name_contexts) or start == 0 or text[start-1] in '.!?\n':
                spans.append((m.start(1), m.end(1)))
    
    return spans


def analyze_text(
    text: str,
    min_confidence: float = 0.0,
    enabled_labels: List[str] | None = None,
) -> PIIResult:

    spans: List[Tuple[str, int, int, str]] = []

    # 1. Regex detection for structured data (email, phone, etc.)
    for label, start, end in _apply_regex(text, enabled_labels=enabled_labels):
        spans.append((label, start, end, "REGEX"))

    # 2. Name detection (only if NAME is enabled)
    if enabled_labels is None or "NAME" in enabled_labels:
        for start, end in _apply_name_detection(text):
            spans.append(("NAME", start, end, "PATTERN"))

    # 3. NLP-based detection (spaCy) - only for PERSON entities
    if enabled_labels is None or "NAME" in enabled_labels:
        nlp_spans = _apply_nlp_ner(text, enabled_labels=enabled_labels)
        spans.extend(nlp_spans)

    # Sort + merge overlaps
    spans.sort(key=lambda x: (x[1], -(x[2]-x[1])))
    merged = []
    for label, start, end, source in spans:
        if not merged:
            merged.append((label, start, end, source))
            continue
        last_label, last_start, last_end, last_source = merged[-1]
        if start <= last_end:
            # Keep the longer span
            if (end - start) > (last_end - last_start):
                merged[-1] = (label, start, end, source)
        else:
            merged.append((label, start, end, source))

    # Group entities
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for label, start, end, source in merged:
        grouped.setdefault(label, []).append(
            {"text": text[start:end], "start": start, "end": end, "source": source}
        )

    # Highlighted HTML
    highlighted = _build_highlighted_html(text, merged)

    return PIIResult(entities=grouped, highlighted_markdown=highlighted)


def _build_highlighted_html(
    text: str, spans: List[Tuple[str, int, int, str]]
) -> str:
    """
    Build HTML string with colored spans for Streamlit markdown.
    """
    color_map = {
        "NAME": "#ffcccc",
        "USERNAME": "#e1bee7",
        "EMAIL": "#b3e5fc",
        "PHONE": "#b2dfdb",
        "IP_ADDRESS": "#f8bbd0",
        "NATIONAL_ID": "#c5cae9",
        "CREDIT_CARD": "#ffccbc",
        "PASSWORD": "#ffb3e6",
        "BANK_ACCOUNT": "#c8e6c9",
        "ADDRESS": "#fff59d",
    }

    # Sort spans to build text in order
    spans = sorted(spans, key=lambda x: x[1])
    pos = 0
    parts: List[str] = []

    for label, start, end, source in spans:
        # Add plain text before the span
        if start > pos:
            parts.append(_escape_html(text[pos:start]))

        bg = color_map.get(label, "#e0e0e0")
        token = _escape_html(text[start:end])
        tooltip = f"{label} ({source})"

        parts.append(
            f'<span style="background-color:{bg}; padding:2px 3px; '
            f'border-radius:3px;" title="{tooltip}">{token}</span>'
        )
        pos = end

    # Remaining text
    if pos < len(text):
        parts.append(_escape_html(text[pos:]))

    return "<p style='white-space: pre-wrap; font-family:monospace;'>" + "".join(
        parts
    ) + "</p>"


# Load spaCy model once
_nlp = None
if spacy is not None:
    try:
        _nlp = spacy.load("en_core_web_sm")
    except (OSError, Exception):
        print("Warning: spaCy model 'en_core_web_sm' not found or spaCy is not compatible. NLP-based name detection will be disabled.")
        _nlp = None


def _apply_nlp_ner(text: str, enabled_labels=None):
    """
    Use spaCy NER to detect PERSON entities.
    Returns list of (label, start, end, source).
    """
    if _nlp is None:
        return []
    
    doc = _nlp(text)
    results = []

    for ent in doc.ents:
        # Only detect PERSON entities for names
        if ent.label_ == "PERSON":
            # Additional validation
            name_text = text[ent.start_char:ent.end_char]
            if _is_valid_name(name_text):
                # Respect enabled_labels
                if enabled_labels and "NAME" not in enabled_labels:
                    continue
                results.append(("NAME", ent.start_char, ent.end_char, "NLP"))
    
    return results


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )
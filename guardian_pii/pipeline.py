import re
import unicodedata
from functools import lru_cache

# Verhoeff multiplication and permutation tables.
_D = [
 [0,1,2,3,4,5,6,7,8,9],
 [1,2,3,4,0,6,7,8,9,5],
 [2,3,4,0,1,7,8,9,5,6],
 [3,4,0,1,2,8,9,5,6,7],
 [4,0,1,2,3,9,5,6,7,8],
 [5,9,8,7,6,0,4,3,2,1],
 [6,5,9,8,7,1,0,4,3,2],
 [7,6,5,9,8,2,1,0,4,3],
 [8,7,6,5,9,3,2,1,0,4],
 [9,8,7,6,5,4,3,2,1,0],
]
_P = [
 [0,1,2,3,4,5,6,7,8,9],
 [1,5,7,6,2,8,3,0,9,4],
 [5,8,0,3,7,9,6,1,4,2],
 [8,9,1,6,0,4,3,5,2,7],
 [9,4,5,3,1,2,6,8,7,0],
 [4,2,8,6,5,7,3,9,0,1],
 [2,7,9,3,8,0,6,4,1,5],
 [7,0,4,6,9,1,3,2,5,8],
]
_INV = [0,4,3,2,1,5,6,7,8,9]

def verhoeff_valid(number: str) -> bool:
    digits = [int(x) for x in reversed(re.sub(r"\D", "", number))]
    c = 0
    for i, digit in enumerate(digits):
        c = _D[c][_P[i % 8][digit]]
    return c == 0

def luhn_valid(number: str) -> bool:
    digits = [int(x) for x in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    total = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
    digit_map = {}
    for start in (0x0966, 0x0C66, 0x0BE6):
        for i in range(10):
            digit_map[chr(start + i)] = str(i)
    text = text.translate(str.maketrans(digit_map))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def detect_language(text: str) -> str:
    counts = {"HI": 0, "TE": 0, "TA": 0, "EN": 0}
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            counts["HI"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["TE"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["TA"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["EN"] += 1
    nonzero = [k for k, v in counts.items() if v]
    if not nonzero:
        return "Unknown"
    if len(nonzero) > 1:
        return "Mixed"
    return nonzero[0]

def language_segments(text: str):
    langs = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            langs.append("HI")
        elif 0x0C00 <= cp <= 0x0C7F:
            langs.append("TE")
        elif 0x0B80 <= cp <= 0x0BFF:
            langs.append("TA")
        elif ch.isascii() and ch.isalpha():
            langs.append("EN")
    return sorted(set(langs))

KEYWORDS = {
    "AADHAAR": ["aadhaar", "aadhar", "uid", "आधार", "ఆధార్", "ஆதார்"],
    "MOBILE": ["mobile", "phone", "contact", "tel", "फोन", "मोबाइल", "ఫోన్", "మొబైల్", "தொலைபேசி", "மொபைல்"],
    "PAN": ["pan", "permanent account", "पैन", "పాన్", "பான்"],
    "CARD": ["card", "credit", "debit", "visa", "mastercard", "कार्ड", "కార్డు", "அட்டை"],
}

def _near_keyword(text, start, end, category, window=55):
    snippet = text[max(0, start-window):min(len(text), end+window)].lower()
    return any(k.lower() in snippet for k in KEYWORDS.get(category, []))

def _regex_entities(text):
    found = []
    def add(pattern, category, validator=None, context=None, flags=re.I):
        for m in re.finditer(pattern, text, flags):
            value = m.group(0)
            if validator and not validator(value):
                continue
            if context and not context(text, m.start(), m.end(), category):
                continue
            found.append({
                "start": m.start(), "end": m.end(), "category": category,
                "value": value, "detector": "rules",
                "language": detect_language(text[max(0,m.start()-30):m.end()+30])
            })

    add(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", "EMAIL")
    add(r"(?<!\d)(?:(?:\+91|0)[ -]?)?[6-9]\d{9}(?!\d)", "MOBILE",
        context=lambda t,s,e,c: _near_keyword(t,s,e,c) or t[s:e].lstrip().startswith(("+91","0")))
    add(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)", "AADHAAR",
        validator=verhoeff_valid,
        context=lambda t,s,e,c: _near_keyword(t,s,e,c))
    add(r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])", "PAN",
        context=lambda t,s,e,c: _near_keyword(t,s,e,c))
    add(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", "CARD",
        validator=luhn_valid,
        context=lambda t,s,e,c: _near_keyword(t,s,e,c))
    return found

@lru_cache(maxsize=2)
def _load_ner(model_name):
    from transformers import pipeline
    return pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
    )

def _ner_entities(text, model_name):
    ner = _load_ner(model_name)
    raw = ner(text)
    output = []
    mapping = {
        "PER": "PERSON", "PERSON": "PERSON",
        "LOC": "LOCATION", "LOCATION": "LOCATION",
        "ORG": "ORGANISATION", "ORGANIZATION": "ORGANISATION",
    }
    for item in raw:
        label = str(item.get("entity_group", item.get("entity", ""))).upper()
        # Accommodate labels like B-PER / I-ORG.
        label = label.split("-")[-1]
        category = mapping.get(label)
        start, end = int(item["start"]), int(item["end"])
        if category and end > start:
            output.append({
                "start": start, "end": end, "category": category,
                "value": text[start:end], "detector": "XLM-RoBERTa NER",
                "language": detect_language(text[max(0,start-30):end+30])
            })
    return output

def _resolve_overlaps(entities):
    # Validated structured identifiers take priority over NER; longest span wins next.
    priority = {
        "AADHAAR": 0, "MOBILE": 0, "EMAIL": 0, "PAN": 0, "CARD": 0,
        "PERSON": 1, "LOCATION": 1, "ORGANISATION": 1
    }
    ordered = sorted(entities, key=lambda e: (priority.get(e["category"], 9),
                                               -(e["end"]-e["start"]), e["start"]))
    accepted = []
    for candidate in ordered:
        if any(candidate["start"] < old["end"] and candidate["end"] > old["start"]
               for old in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda e: e["start"])

def detect_and_mask(text, mode="hybrid", enable_ner=False,
                    model_name="Davlan/xlm-roberta-base-ner-hrl"):
    clean = normalize_text(text)
    entities = []
    warnings = []
    if mode in ("hybrid", "rules"):
        entities.extend(_regex_entities(clean))
    if mode in ("hybrid", "ner") and enable_ner:
        try:
            entities.extend(_ner_entities(clean, model_name))
        except Exception as exc:
            warnings.append(f"NER model unavailable; rule detections still shown. Details: {exc}")
    elif mode == "ner" and not enable_ner:
        warnings.append("NER-only mode selected but NER is disabled. Enable the model in the sidebar.")

    entities = _resolve_overlaps(entities)
    masked = clean
    for entity in reversed(entities):
        masked = masked[:entity["start"]] + f'[{entity["category"]}]' + masked[entity["end"]:]
    return {
        "masked_text": masked,
        "entities": entities,
        "language_summary": "+".join(language_segments(clean)),
        "warnings": warnings,
    }

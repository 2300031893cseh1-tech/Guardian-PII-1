import json
from pathlib import Path
from collections import defaultdict
from .pipeline import detect_and_mask

def _f2(p, r):
    return 0.0 if (p == 0 and r == 0) else (5 * p * r / (4 * p + r))

def _metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f2": _f2(p, r)}

def _overlap(a, b):
    return a["start"] < b["end"] and a["end"] > b["start"]

def _match(gold, predicted, exact=False):
    used = set()
    tp = 0
    for g in gold:
        for i, p in enumerate(predicted):
            if i in used or g["category"] != p["category"]:
                continue
            matches = (g["start"] == p["start"] and g["end"] == p["end"]) if exact else _overlap(g, p)
            if matches:
                tp += 1
                used.add(i)
                break
    return tp, len(predicted)-tp, len(gold)-tp

def evaluate_records(records, mode="hybrid", enable_ner=False):
    totals = {"partial": [0,0,0], "exact": [0,0,0]}
    by_lang = defaultdict(lambda: {"partial":[0,0,0], "exact":[0,0,0]})
    by_cat = defaultdict(lambda: {"partial":[0,0,0], "exact":[0,0,0]})
    char_correct = char_total = 0

    for record in records:
        text = record["text"]
        gold = record.get("entities", [])
        result = detect_and_mask(text, mode=mode, enable_ner=enable_ner)
        pred = result["entities"]
        lang = record.get("language", result["language_summary"])
        for key, exact in (("partial", False), ("exact", True)):
            tp, fp, fn = _match(gold, pred, exact=exact)
            for i, v in enumerate((tp,fp,fn)):
                totals[key][i] += v
                by_lang[lang][key][i] += v
            cats = set([x["category"] for x in gold] + [x["category"] for x in pred])
            for cat in cats:
                gcat = [x for x in gold if x["category"] == cat]
                pcat = [x for x in pred if x["category"] == cat]
                ctp, cfp, cfn = _match(gcat, pcat, exact=exact)
                for i, v in enumerate((ctp,cfp,cfn)):
                    by_cat[cat][key][i] += v

        # Character-level binary PII/non-PII accuracy using partial overlap spans.
        gold_mask = [0] * len(text)
        pred_mask = [0] * len(text)
        for e in gold:
            for pos in range(max(0,e["start"]), min(len(text),e["end"])):
                gold_mask[pos] = 1
        for e in pred:
            for pos in range(max(0,e["start"]), min(len(text),e["end"])):
                pred_mask[pos] = 1
        char_correct += sum(a == b for a,b in zip(gold_mask,pred_mask))
        char_total += len(text)

    def pack(d):
        return _metrics(*d)
    per_language = []
    for lang, values in sorted(by_lang.items()):
        per_language.append({"language":lang, **{k:pack(v) for k,v in values.items()}})
    per_category = []
    for cat, values in sorted(by_cat.items()):
        per_category.append({"category":cat, **{k:pack(v) for k,v in values.items()}})
    return {
        "partial": pack(totals["partial"]),
        "exact": pack(totals["exact"]),
        "character_accuracy": char_correct / char_total if char_total else 0.0,
        "per_language": per_language,
        "per_category": per_category,
        "records_evaluated": len(records),
        "note": "Metrics are computed only against the supplied annotations. Character accuracy is supplementary; span precision/recall/F2 are the primary project metrics."
    }

def evaluate_jsonl(path_or_file, mode="hybrid", enable_ner=False):
    if hasattr(path_or_file, "read"):
        raw = path_or_file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        with open(path_or_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    return evaluate_records(records, mode=mode, enable_ner=enable_ner)

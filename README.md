# Guardian PII — VS Code Project

A local Streamlit prototype based on the uploaded Guardian PII capstone report.

## Features

- Upload TXT, CSV, DOCX or text-based PDF files, or paste text.
- Normalise Unicode and map Devanagari, Telugu and Tamil digits to ASCII.
- Detect email, Indian mobile, Aadhaar (Verhoeff), PAN and payment-card (Luhn) candidates.
- Optional Hugging Face XLM-RoBERTa NER for PERSON, LOCATION and ORGANISATION.
- Resolve overlapping spans and replace them with category labels.
- Display detected entities and download masked text / entity CSV.
- Evaluate an annotated JSONL dataset with partial and exact span precision, recall and F2.
- Supplementary character-level accuracy is calculated from annotated ground truth.

## Run in VS Code (Windows PowerShell)

Open this folder in VS Code, then open Terminal → New Terminal.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

The browser should open at `http://localhost:8501`.

### Smaller install (without transformer model)

For a rules-only demonstration, install:
```powershell
pip install streamlit pandas pypdf python-docx
```
Then run the app and leave **Enable XLM-RoBERTa NER** unchecked. Rule detectors still work.
The optional NER checkbox requires `transformers`, `torch`, internet access for the first model download, and additional RAM.

## How to use

1. Open **Scan text / file**.
2. Upload a sample file or use the example text.
3. Click **Detect and mask PII**.
4. Review the masked text and entity table.
5. Download the masked text or entity CSV.
6. Open **Evaluate annotated dataset** and click **Run sample evaluation** to see metrics for the included labelled sample.

## Evaluation JSONL format

Each line is one JSON object:
```json
{
  "text": "Email: demo@example.com",
  "language": "EN",
  "entities": [
    {"start": 7, "end": 23, "category": "EMAIL"}
  ]
}
```
Offsets are zero-based character offsets into the exact `text` string. Categories must match the detector labels (EMAIL, MOBILE, AADHAAR, PAN, CARD, PERSON, LOCATION, ORGANISATION).

## Metrics: important distinction

The report's Chapter 7 defines **precision**, **recall**, and **F2 (β=2)** with partial and exact span matching. Table 7 contains language-wise hybrid precision/recall/F2 values. The report does not report an overall Guardian PII accuracy value. This app computes character-level PII/non-PII accuracy only when ground-truth annotations are supplied; it must not be described as a score from the report.

The included `data/sample_eval.jsonl` is a small demonstration dataset created for this runnable project. It is **not** the report's 400-record manually annotated research dataset and must not be used to claim the report's measured performance.

## Detection notes and limitations

- Structured identifiers use format and checksum validation, and Aadhaar/PAN/card/mobile typing uses nearby multilingual keywords where appropriate.
- Mobile, Aadhaar, PAN and card detections are context-aware to reduce confusion with unrelated digit strings.
- NER is optional because downloading/running a transformer is resource intensive. If the model is unavailable, the app reports the issue and keeps rule detections.
- This is a prototype, not a production anonymisation or compliance system. It does not guarantee detection of all PII, and it does not evaluate transliterated Indic text, adversarial evasion, concurrent-user scalability, latency, tunnel security or legal compliance.
- Use synthetic data only. Do not upload real personal or institutional data.

## Report alignment

The implementation follows the report's Appendix A/B and Chapters 4–7 pipeline:
normalisation → script/language labelling → NER + regex candidates → validation/context
filtering → overlap resolution → category-label masking → annotated evaluation.

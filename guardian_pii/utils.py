import io
from pathlib import Path

def extract_text_from_upload(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return data.decode("utf-8-sig", errors="replace")
    if name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(io.BytesIO(data), dtype=str).fillna("")
        return "\n".join(" | ".join(row) for row in df.astype(str).values.tolist())
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("Supported file types are TXT, CSV, DOCX and PDF.")

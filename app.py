import io
import json
import pandas as pd
import streamlit as st

from guardian_pii.pipeline import detect_and_mask
from guardian_pii.evaluation import evaluate_jsonl
from guardian_pii.utils import extract_text_from_upload

st.set_page_config(page_title="Guardian PII", page_icon="🛡️", layout="wide")
st.title("🛡️ Guardian PII")
st.caption("Multilingual personal-data detection and category-label masking")
st.warning(
    "Prototype only. Do not upload real personal or institutional data. "
    "Masking is not a guarantee of anonymisation."
)

with st.sidebar:
    st.header("Configuration")
    mode_label = st.selectbox(
        "Detection configuration",
        ["Guardian PII hybrid", "Rules-only", "NER-only"],
        index=0,
    )
    use_ner = st.checkbox(
        "Enable XLM-RoBERTa NER (downloads model on first use)",
        value=False,
        help="Requires internet access and additional memory. Rules work without this option.",
    )
    model_name = st.text_input(
        "Hugging Face NER model",
        value="Davlan/xlm-roberta-base-ner-hrl",
        disabled=not use_ner or mode_label == "Rules-only",
    )
    st.caption("Supported text: English, Hindi, Telugu and Tamil.")

tab_scan, tab_eval, tab_info = st.tabs(
    ["🔎 Scan text / file", "📊 Evaluate annotated dataset", "ℹ️ About & limitations"]
)

with tab_scan:
    st.subheader("Submit text")
    uploaded = st.file_uploader(
        "Or upload a TXT / CSV / DOCX / PDF file",
        type=["txt", "csv", "docx", "pdf"],
    )
    default_text = (
        "Name: Divya Chand. Contact: divya.demo@example.com. "
        "Mobile: 9876543210. PAN: ABCDE1234F. "
        "Aadhaar: 234567890122. Card: 4111111111111111."
    )
    text_input = st.text_area(
        "Enter English, Hindi, Telugu or Tamil text",
        value=default_text,
        height=180,
    )

    if uploaded is not None:
        try:
            extracted = extract_text_from_upload(uploaded)
            st.info(f"Loaded text from: {uploaded.name}")
            text_input = extracted
            st.text_area("Extracted file text (editable)", value=extracted, height=180, key="file_text")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")

    run_scan = st.button("Detect and mask PII", type="primary", use_container_width=True)

    if run_scan:
        source_text = st.session_state.get("file_text", text_input) if uploaded else text_input
        if not source_text.strip():
            st.error("Please enter text or upload a file.")
        else:
            effective_mode = {"Guardian PII hybrid": "hybrid", "Rules-only": "rules", "NER-only": "ner"}[mode_label]
            with st.spinner("Analysing text..."):
                result = detect_and_mask(
                    source_text,
                    mode=effective_mode,
                    enable_ner=use_ner and effective_mode != "rules",
                    model_name=model_name,
                )
            st.subheader("Masked output")
            st.code(result["masked_text"], language="text")
            st.download_button(
                "Download masked text",
                data=result["masked_text"].encode("utf-8"),
                file_name="guardian_pii_masked.txt",
                mime="text/plain",
            )

            st.subheader("Detection summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Entities detected", len(result["entities"]))
            c2.metric("Detected categories", len(set(x["category"] for x in result["entities"])))
            c3.metric("Detected language/script", result["language_summary"] or "Mixed / unknown")

            if result["entities"]:
                df = pd.DataFrame(result["entities"])
                df["value"] = df["value"].map(lambda x: x[:3] + "…" if len(x) > 3 else "•••")
                st.dataframe(df[["category", "value", "language", "detector", "start", "end"]],
                             use_container_width=True, hide_index=True)
                st.download_button(
                    "Download entity report CSV",
                    data=pd.DataFrame(result["entities"]).to_csv(index=False).encode("utf-8"),
                    file_name="guardian_pii_entities.csv",
                    mime="text/csv",
                )
            else:
                st.info("No entities matched the enabled detectors and validation rules.")

            if result.get("warnings"):
                for warning in result["warnings"]:
                    st.warning(warning)

with tab_eval:
    st.subheader("Evaluate against annotated ground truth")
    st.write(
        "Choose a JSONL file where each line contains `text`, `language`, and an "
        "`entities` array with `start`, `end`, and `category`. The included sample dataset "
        "is a small demonstration set, not the report's 400-record research dataset."
    )
    eval_file = st.file_uploader("Upload annotated JSONL (optional)", type=["jsonl"], key="eval_file")
    if st.button("Run sample evaluation", type="primary"):
        path = eval_file if eval_file is not None else str(
            __import__("pathlib").Path(__file__).parent / "data" / "sample_eval.jsonl"
        )
        with st.spinner("Computing span and character-level metrics..."):
            try:
                report = evaluate_jsonl(
                    path,
                    mode="hybrid",
                    enable_ner=False,
                )
                st.markdown("#### Overall metrics")
                a, b, c, d = st.columns(4)
                a.metric("Precision (partial)", f'{report["partial"]["precision"]:.3f}')
                b.metric("Recall (partial)", f'{report["partial"]["recall"]:.3f}')
                c.metric("F2 (partial)", f'{report["partial"]["f2"]:.3f}')
                d.metric("Character accuracy", f'{report["character_accuracy"]:.3f}')
                st.caption(
                    "Character accuracy is a supplementary character-level measure. "
                    "The report's primary measures are precision, recall and recall-weighted F2."
                )
                st.markdown("#### Exact-span metrics")
                x, y, z = st.columns(3)
                x.metric("Precision (exact)", f'{report["exact"]["precision"]:.3f}')
                y.metric("Recall (exact)", f'{report["exact"]["recall"]:.3f}')
                z.metric("F2 (exact)", f'{report["exact"]["f2"]:.3f}')
                st.markdown("#### Per-language results")
                st.dataframe(pd.DataFrame(report["per_language"]), use_container_width=True, hide_index=True)
                st.markdown("#### Per-category results")
                st.dataframe(pd.DataFrame(report["per_category"]), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download evaluation report JSON",
                    data=json.dumps(report, indent=2).encode("utf-8"),
                    file_name="guardian_pii_evaluation.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

with tab_info:
    st.subheader("Pipeline implemented from the capstone report")
    st.markdown("""
    1. Unicode NFC normalisation, removal of zero-width / non-breaking characters,
       whitespace cleanup and native-digit mapping.
    2. Script/language labelling for English (EN), Hindi (HI), Telugu (TE) and Tamil (TA).
    3. Transformer NER for PERSON, LOCATION and ORGANISATION (optional model download).
    4. Rule-based candidates for Indian mobile, email, Aadhaar (Verhoeff), PAN and
       payment card (Luhn).
    5. Multilingual keyword context, filtering and priority-based overlap resolution.
    6. Replace accepted spans with labels such as `[AADHAAR]`, `[MOBILE]` and `[PERSON]`.
    """)
    st.markdown("#### Metrics and honesty about results")
    st.write(
        "The report defines precision, recall and F2 (β=2), and calls for partial and exact "
        "span matching. Its Table 7 provides hybrid precision/recall/F2 values by language. "
        "It does not provide an overall accuracy figure for Guardian PII. This app therefore "
        "computes accuracy only from the uploaded/sample annotated ground truth and does "
        "not present it as a result from the report."
    )
    st.markdown("#### Reported hybrid language metrics (reference only)")
    reference = pd.DataFrame([
        {"Language": "English", "Precision (partial)": 0.94, "Recall (partial)": 0.96, "F2 (partial)": 0.96},
        {"Language": "Hindi", "Precision (partial)": 0.91, "Recall (partial)": 0.93, "F2 (partial)": 0.93},
        {"Language": "Telugu", "Precision (partial)": 0.89, "Recall (partial)": 0.91, "F2 (partial)": 0.91},
        {"Language": "Tamil", "Precision (partial)": 0.88, "Recall (partial)": 0.90, "F2 (partial)": 0.90},
    ])
    st.dataframe(reference, use_container_width=True, hide_index=True)
    st.caption("Reference values transcribed from Table 7, page 39 of the uploaded capstone report.")
    st.error(
        "Privacy note: this prototype processes uploaded content in the local Python process, "
        "but file contents may still be retained by the browser/session or environment. "
        "Use synthetic data for demonstrations."
    )

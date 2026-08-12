import json
import io
import pandas as pd
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from rapidfuzz import process, utils
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Handwriting Reference Extractor", layout="wide")

COMPANY_MAPPING = {
    "مصري": "coral",
    "مع رضا": "alliance",
    "sun": "sun",
    "masters": "masters",
    "cairo express": "cairo express",
    "blue sky": "BLUE SKY",
    "jois": "JOIS",
}

KNOWN_COMPANIES = ["alliance", "BLUE SKY", "cairo express", "coral", "JOIS", "masters", "sun"]


def normalize_company(raw_text: str) -> str:
    if not raw_text:
        return ""
    clean_text = raw_text.strip().lower()
    if clean_text in COMPANY_MAPPING:
        return COMPANY_MAPPING[clean_text]
    match = process.extractOne(
        clean_text, KNOWN_COMPANIES, processor=utils.default_process, score_cutoff=65
    )
    if match:
        return match[0]
    return raw_text


def process_image(image: Image.Image, api_key: str, model: str = "gemini-2.5-flash"):
    client = genai.Client(api_key=api_key)
    prompt = """
    Analyze this handwritten note carefully. It contains list entries organized under dates (e.g., 29-7, 30-7).
    Extract each line item into a structured list with these exact keys:
    1. "date": The section header date (e.g., "29-7", "30-7").
    2. "reference": The code or alphanumeric reference string on the left side (e.g., "2605NLRRJ6H3", "369701", "BP4Yj").
    3. "company": The text on the right side if present (e.g., "مع رضا", "مصري", "sun", "Masters"). If blank, use null.

    Respond STRICTLY with a valid JSON array of objects.
    """
    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
    )
    raw_data = json.loads(response.text)
    processed_records = []
    for item in raw_data:
        raw_company = item.get("company")
        standardized_company = normalize_company(raw_company) if raw_company else ""
        processed_records.append(
            {
                "Date": item.get("date", ""),
                "Reference": item.get("reference", ""),
                "Raw Extracted Text": raw_company if raw_company else "",
                "Company": standardized_company,
            }
        )
    return pd.DataFrame(processed_records)


def to_excel_buffer(df: pd.DataFrame, sheet_name: str = "References") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


st.title("Handwritten Reference & Company Extractor")
st.write("Upload one or more images of handwritten reference numbers. Extract, edit, and download as Excel/CSV.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)

uploaded_files = st.file_uploader(
    "Upload Handwritten Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if "results" not in st.session_state:
    st.session_state.results = {}

if uploaded_files:
    if st.button("Extract All Images"):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            st.session_state.results.clear()
            progress = st.progress(0, text="Starting extraction...")
            for i, uploaded in enumerate(uploaded_files):
                progress.progress(
                    i / len(uploaded_files),
                    text=f"Processing {uploaded.name} ({i + 1}/{len(uploaded_files)})...",
                )
                image_bytes = uploaded.getvalue()
                image = Image.open(uploaded)
                try:
                    df = process_image(image, api_key, model)
                    st.session_state.results[uploaded.name] = {"df": df, "image": image_bytes}
                except Exception as e:
                    st.session_state.results[uploaded.name] = {"df": f"ERROR: {e}", "image": image_bytes}
            progress.progress(1.0, text="Done!")
            st.rerun()

    if st.session_state.results:
        st.divider()

        has_any = any(isinstance(v["df"], pd.DataFrame) for v in st.session_state.results.values())
        if has_any:
            merge_col1, merge_col2 = st.columns([1, 3])
            with merge_col1:
                selected = st.multiselect(
                    "Select images to merge",
                    options=[k for k, v in st.session_state.results.items() if isinstance(v["df"], pd.DataFrame)],
                    default=[k for k, v in st.session_state.results.items() if isinstance(v["df"], pd.DataFrame)],
                )
            with merge_col2:
                if selected:
                    merged = pd.concat(
                        [st.session_state.results[name]["df"] for name in selected], ignore_index=True
                    )
                    st.download_button(
                        label=f"Download Merged Excel ({len(merged)} rows)",
                        data=to_excel_buffer(merged),
                        file_name="merged_references.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                    st.download_button(
                        label=f"Download Merged CSV ({len(merged)} rows)",
                        data=merged.to_csv(index=False).encode("utf-8-sig"),
                        file_name="merged_references.csv",
                        mime="text/csv",
                    )

        st.subheader("Per-Image Results")

        for name, result in st.session_state.results.items():
            with st.expander(name, expanded=True):
                if isinstance(result["df"], str):
                    st.error(result["df"])
                else:
                    st.subheader("Preview")
                    st.image(result["image"], width=300)
                    edited = st.data_editor(result["df"], num_rows="dynamic", key=f"editor_{name}")
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            label=f"Download {name} Excel",
                            data=to_excel_buffer(edited),
                            file_name=f"{os.path.splitext(name)[0]}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_excel_{name}",
                        )
                    with dl_col2:
                        st.download_button(
                            label=f"Download {name} CSV",
                            data=edited.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"{os.path.splitext(name)[0]}.csv",
                            mime="text/csv",
                            key=f"dl_csv_{name}",
                        )

        if st.button("Clear All Results"):
            st.session_state.results.clear()
            st.rerun()

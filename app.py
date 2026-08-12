import json
import io
import time
import logging
import base64
import pandas as pd
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from rapidfuzz import process, utils
from dotenv import load_dotenv
import ollama
import requests
import os

load_dotenv()

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

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

EXTRACTION_PROMPT = """
Analyze this handwritten note carefully. It contains list entries organized under dates (e.g., 29-7, 30-7).
Extract each line item into a structured list with these exact keys:
1. "date": The section header date (e.g., "29-7", "30-7").
2. "reference": The code or alphanumeric reference string on the left side (e.g., "2605NLRRJ6H3", "369701", "BP4Yj").
3. "company": The text on the right side if present (e.g., "مع رضا", "المصري", "sun", "Masters"). If blank, use null.

Respond STRICTLY with a valid JSON array of objects. No markdown, no explanation.
"""


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


def parse_to_dataframe(raw_data: list) -> pd.DataFrame:
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


def process_image_gemini(image: Image.Image, api_key: str, model: str = "gemini-2.5-flash"):
    logger.info("Starting Gemini extraction | model=%s | image_size=%sx%s", model, image.width, image.height)
    start = time.time()
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[image, EXTRACTION_PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
        )
        elapsed = time.time() - start
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", "?") if usage else "?"
        completion_tokens = getattr(usage, "candidates_token_count", "?") if usage else "?"
        logger.info(
            "Gemini response OK | model=%s | time=%.2fs | prompt_tokens=%s | completion_tokens=%s",
            model, elapsed, prompt_tokens, completion_tokens,
        )
        raw_data = json.loads(response.text)
        logger.info("Parsed %d records from Gemini response", len(raw_data))
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Gemini call failed | model=%s | time=%.2fs | error=%s", model, elapsed, str(e))
        raise
    return parse_to_dataframe(raw_data)


def process_image_ollama(image: Image.Image, model: str, base_url: str = "http://localhost:11434",
                         api_key: str | None = None):
    logger.info("Starting Ollama extraction | model=%s | base_url=%s | image_size=%sx%s",
                model, base_url, image.width, image.height)
    start = time.time()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        client = ollama.Client(host=base_url, headers=headers)
        response = client.chat(
            model=model,
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT, "images": [image_b64]},
            ],
            options={"temperature": 0.1},
        )
        elapsed = time.time() - start
        content = response["message"]["content"]
        logger.info("Ollama response OK | model=%s | time=%.2fs | response_length=%d",
                    model, elapsed, len(content))
        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]
        raw_data = json.loads(json_match.strip())
        logger.info("Parsed %d records from Ollama response", len(raw_data))
    except Exception as e:
        elapsed = time.time() - start
        logger.error("Ollama call failed | model=%s | time=%.2fs | error=%s", model, elapsed, str(e))
        raise
    return parse_to_dataframe(raw_data)


def get_ollama_models(base_url: str = "http://localhost:11434", api_key: str | None = None) -> list[str]:
    try:
        if "ollama.com" in base_url:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            resp = requests.get(f"{base_url}/api/tags", headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        else:
            client = ollama.Client(host=base_url)
            models = client.list()
            return [m.model for m in models.models]
    except Exception as e:
        logger.error("Failed to fetch Ollama models | base_url=%s | error=%s", base_url, str(e))
        return []


def to_excel_buffer(df: pd.DataFrame, sheet_name: str = "References") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


st.title("Handwritten Reference & Company Extractor")
st.write("Upload one or more images of handwritten reference numbers. Extract, edit, and download as Excel/CSV.")

with st.sidebar:
    st.header("Configuration")
    provider = st.radio("Provider", ["Google Gemini", "Ollama Cloud"], horizontal=True)

    if provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        model = st.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
        ollama_base_url = None
        ollama_api_key = None
    else:
        ollama_base_url = "https://ollama.com"
        ollama_api_key = st.text_input("Ollama API Key", type="password",
                                       value=os.getenv("OLLAMA_API_KEY", "204e06089d064ab7b57ea61ca220731f.aBlDqS6bE-kVeMlUen98QQa1"))
        ollama_models = get_ollama_models(ollama_base_url, ollama_api_key)
        if ollama_models:
            saved_idx = 0
            if "ollama_cloud_model" in st.session_state and st.session_state.ollama_cloud_model in ollama_models:
                saved_idx = ollama_models.index(st.session_state.ollama_cloud_model)
            model = st.selectbox("Ollama Cloud Model", ollama_models, index=saved_idx, key="ollama_cloud_model")
        else:
            model = ""
            st.warning("No cloud models found. Check your API key.")
        api_key = None

uploaded_files = st.file_uploader(
    "Upload Handwritten Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if "results" not in st.session_state:
    st.session_state.results = {}
if "confirmed" not in st.session_state:
    st.session_state.confirmed = {}

if uploaded_files:
    if st.button("Extract All Images"):
        if provider == "Google Gemini" and not api_key:
            logger.warning("Extraction blocked: no API key provided")
            st.error("Please enter your Gemini API Key in the sidebar.")
        elif provider == "Ollama Cloud" and not model:
            st.error("No Ollama models available.")
        else:
            st.session_state.results.clear()
            st.session_state.confirmed.clear()
            logger.info("Batch extraction started | provider=%s | files=%d | model=%s",
                        provider, len(uploaded_files), model)
            progress = st.progress(0, text="Starting extraction...")
            for i, uploaded in enumerate(uploaded_files):
                progress.progress(
                    i / len(uploaded_files),
                    text=f"Processing {uploaded.name} ({i + 1}/{len(uploaded_files)})...",
                )
                logger.info("Processing file %d/%d: %s", i + 1, len(uploaded_files), uploaded.name)
                image_bytes = uploaded.getvalue()
                image = Image.open(uploaded)
                try:
                    if provider == "Google Gemini":
                        df = process_image_gemini(image, api_key, model)
                    else:
                        df = process_image_ollama(image, model, ollama_base_url, ollama_api_key)
                    st.session_state.results[uploaded.name] = {"df": df, "image": image_bytes}
                    logger.info("Extraction OK | file=%s | rows=%d", uploaded.name, len(df))
                except Exception as e:
                    st.session_state.results[uploaded.name] = {"df": f"ERROR: {e}", "image": image_bytes}
                    logger.error("Extraction FAILED | file=%s | error=%s", uploaded.name, str(e))
            progress.progress(1.0, text="Done!")
            logger.info("Batch extraction complete | total=%d | success=%d",
                        len(uploaded_files),
                        sum(1 for v in st.session_state.results.values() if isinstance(v["df"], pd.DataFrame)))
            st.rerun()

    if st.session_state.results:
        st.divider()

        confirmed_names = [n for n, c in st.session_state.confirmed.items() if c]
        total = len(st.session_state.results)
        confirmed_count = len(confirmed_names)

        if confirmed_count > 0:
            st.subheader(f"Confirmed: {confirmed_count}/{total}")
            merged = pd.concat(
                [st.session_state.results[n]["df"] for n in confirmed_names], ignore_index=True
            )
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(
                    label=f"Download All Confirmed Excel ({len(merged)} rows)",
                    data=to_excel_buffer(merged),
                    file_name="confirmed_references.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_all_confirmed_excel",
                )
            with dl_col2:
                st.download_button(
                    label=f"Download All Confirmed CSV ({len(merged)} rows)",
                    data=merged.to_csv(index=False).encode("utf-8-sig"),
                    file_name="confirmed_references.csv",
                    mime="text/csv",
                    key="dl_all_confirmed_csv",
                )
            with dl_col3:
                if st.button("Clear All Results", key="clear_all"):
                    logger.info("Clearing all results | count=%d", total)
                    st.session_state.results.clear()
                    st.session_state.confirmed.clear()
                    st.rerun()
        else:
            if st.button("Clear All Results", key="clear_all"):
                logger.info("Clearing all results | count=%d", total)
                st.session_state.results.clear()
                st.session_state.confirmed.clear()
                st.rerun()

        st.subheader("Review Each Image")

        tabs = st.tabs(list(st.session_state.results.keys()))

        for tab, (name, result) in zip(tabs, st.session_state.results.items()):
            with tab:
                col_img, col_data = st.columns([1, 1])

                with col_img:
                    st.subheader("Original Image")
                    st.image(result["image"], use_container_width=True)

                with col_data:
                    st.subheader("Extracted Data")
                    if isinstance(result["df"], str):
                        st.error(result["df"])
                    else:
                        edited = st.data_editor(
                            result["df"], num_rows="dynamic", key=f"editor_{name}"
                        )

                        btn_col1, btn_col2, btn_col3 = st.columns(3)

                        with btn_col1:
                            if st.button("Save Edits", key=f"save_{name}"):
                                st.session_state.results[name]["df"] = edited
                                st.session_state.confirmed[name] = True
                                logger.info("Edits saved & confirmed | file=%s | rows=%d", name, len(edited))
                                st.rerun()

                        with btn_col2:
                            if st.button("Re-extract", key=f"reextract_{name}"):
                                image_bytes = st.session_state.results[name]["image"]
                                image = Image.open(io.BytesIO(image_bytes))
                                try:
                                    if provider == "Google Gemini":
                                        new_df = process_image_gemini(image, api_key, model)
                                    else:
                                        new_df = process_image_ollama(image, model, ollama_base_url, ollama_api_key)
                                    st.session_state.results[name]["df"] = new_df
                                    st.session_state.confirmed[name] = False
                                    logger.info("Re-extraction OK | file=%s | rows=%d", name, len(new_df))
                                except Exception as e:
                                    st.error(f"Re-extraction failed: {e}")
                                    logger.error("Re-extraction FAILED | file=%s | error=%s", name, str(e))
                                st.rerun()

                        with btn_col3:
                            is_confirmed = st.session_state.confirmed.get(name, False)
                            if is_confirmed:
                                st.success("Confirmed")
                            else:
                                st.warning("Pending")

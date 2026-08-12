Key Technical Strategy
Multimodal Extraction: Send the image to the Vision model with a JSON Schema or structured prompt forcing it to group entries under their corresponding dates (e.g., 29-7, 30-7) and extract two main fields per line: reference and company.

Translation & Mapping: Normalize Arabic company names (e.g., mapping "مع رضا" or "مصري" into consistent English text if needed, or keeping them as-is).

Python Implementation
1. Install Required Packages
Bash
pip install google-genai pandas openpyxl pillow
2. Extraction Script (extract_to_excel.py)
Python
import json
import pandas as pd
from google import genai
from google.genai import types
from PIL import Image

# Initialize the Gemini API client
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

def process_handwritten_note(image_path: str, output_excel_path: str = "output.xlsx"):
    # Load the image
    image = Image.open(image_path)

    # Prompt forcing a strict JSON output matching your target table structure
    prompt = """
    Analyze this handwritten note carefully. It contains list entries organized under dates (e.g., 29-7, 30-7).
    Extract each line item into a structured list containing:
    1. "date": The section header date (e.g., "29-7", "30-7").
    2. "reference": The code or alphanumeric reference number on the left (e.g., "2605NLRRJ6H3", "369701").
    3. "company": The company/notes text on the right side if present (e.g., "مع رضا", "مصري", "sun", "Masters"). If blank, use null.

    Respond ONLY with a valid JSON array of objects. Do not include markdown code block syntax if possible.
    Example schema:
    [
      {"date": "29-7", "reference": "2605NLRRJ6H3", "company": "مع رضا"},
      {"date": "29-7", "reference": "3670149", "company": None}
    ]
    """

    # Request structured output using JSON response schema
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1 # Low temperature for accurate transcription
        )
    )

    # Parse JSON
    data = json.loads(response.text)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # Reorder columns to match standard reference/company layout
    df = df[["date", "reference", "company"]]

    # Export to Excel and CSV
    df.to_excel(output_excel_path, index=False)
    df.to_csv("output.csv", index=False, encoding='utf-8-sig')

    print(f"Data successfully saved to {output_excel_path} and output.csv")
    return df

# Run the extraction
if __name__ == "__main__":
    process_handwritten_note("handwritten_note.jpg")
App Architecture Options
Depending on your goal, you can package this backend into:

Web App (Streamlit / Gradio):

Simple UI where users upload an image and download the generated Excel file immediately.

Mobile App (Flutter / React Native) + FastAPI Backend:

Mobile app captures the paper document with the camera.

Sends the image to your FastAPI endpoint running the Python script above.

Returns the processed table for inline user review/editing before downloading the Excel file.
Here is a complete, production-ready solution featuring a Streamlit web application alongside a fuzzy matching module to automatically standardize Arabic handwriting (e.g., mapping "مصري" or "مع رضا") into your preferred English company names (e.g., "coral", "alliance", "masters").

Step 1: Requirements Installation
Make sure you have the required libraries installed:

Bash
pip install google-genai pandas openpyxl pillow streamlit rapidfuzz
Step 2: The Streamlit Web App (app.py)
Create a file named app.py and paste the code below:

Python
import json
import pandas as pd
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from rapidfuzz import process, utils

# Set page layout
st.set_page_config(page_title="Handwriting Reference Extractor", layout="wide")

# Known company list for fuzzy matching / mapping dictionary
COMPANY_MAPPING = {
    "مصري": "coral",
    "مع رضا": "alliance",
    "sun": "sun",
    "masters": "masters",
    "cairo express": "cairo express",
    "blue sky": "BLUE SKY",
    "jois": "JOIS"
}

KNOWN_COMPANIES = ["alliance", "BLUE SKY", "cairo express", "coral", "JOIS", "masters", "sun"]

def normalize_company(raw_text: str) -> str:
    """Normalizes Arabic or handwritten company names into standard English names."""
    if not raw_text:
        return ""
    
    clean_text = raw_text.strip().lower()

    # 1. Check direct mapping table first
    if clean_text in COMPANY_MAPPING:
        return COMPANY_MAPPING[clean_text]

    # 2. Perform fuzzy string matching for English variations
    match = process.extractOne(
        clean_text, 
        KNOWN_COMPANIES, 
        processor=utils.default_process, 
        score_cutoff=65
    )
    
    if match:
        return match[0]

    # Return original text if no clean match is found
    return raw_text

def process_image(image: Image.Image, api_key: str):
    """Sends image to Gemini API and processes structured extraction."""
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
        model='gemini-2.5-flash',
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )

    raw_data = json.loads(response.text)
    
    # Process and map extracted fields
    processed_records = []
    for item in raw_data:
        raw_company = item.get("company")
        standardized_company = normalize_company(raw_company) if raw_company else ""
        
        processed_records.append({
            "Date": item.get("date", ""),
            "Reference": item.get("reference", ""),
            "Raw Extracted Text": raw_company if raw_company else "",
            "Company": standardized_company
        })

    return pd.DataFrame(processed_records)


# --- Streamlit UI ---
st.title("📄 Handwritten Reference & Company Extractor")
st.write("Upload an image of handwritten reference numbers to generate an Excel/CSV file.")

# Sidebar API Key Input
api_key = st.sidebar.text_input("Gemini API Key", type="password")

uploaded_file = st.file_uploader("Upload Handwritten Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])

    image = Image.open(uploaded_file)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Extraction Controls")
        if st.button("Extract Data"):
            if not api_key:
                st.error("Please enter your Gemini API Key in the sidebar.")
            else:
                with st.spinner("Processing image with Gemini Vision AI..."):
                    try:
                        df = process_image(image, api_key)
                        
                        st.success("Extraction Complete!")
                        
                        # Interactive Table Editor so users can verify/edit data before export
                        edited_df = st.data_editor(df, num_rows="dynamic")

                        # Export Options
                        st.subheader("Download Options")
                        csv_data = edited_df.to_csv(index=False).encode("utf-8-sig")
                        
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="extracted_references.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
Step 3: Run the Application
Run the application locally using Streamlit:

Bash
streamlit run app.py
Key Features Included:
Interactive Data Editor: Allows users to manually adjust misread numbers or company names directly on the screen prior to downloading.

Arabic to English Normalization: Automatically converts handwriting like "مصري" directly into "coral" using standard mapping tables and fuzzy matching (RapidFuzz).

Structured Export: Generates clean .csv output matching the format of reference lists.
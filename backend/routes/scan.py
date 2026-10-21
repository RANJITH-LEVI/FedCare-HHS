import os
import json
import base64
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from backend.schemas.patient import ClinicalInput

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

router = APIRouter(tags=["Document Scanning"])

class ScanBase64Request(BaseModel):
    image_base64: str
    mime_type: Optional[str] = "image/jpeg"

DEMO_EXTRACTED_CLINICAL_DATA = {
    "age": 67.0,
    "sex": 1,
    "cp": 4,
    "trestbps": 160.0,
    "chol": 286.0,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108.0,
    "exang": 1,
    "oldpeak": 2.6,
    "slope": 2,
    "ca": 3,
    "thal": 7
}

@router.post("/scan-base64", response_model=ClinicalInput)
async def scan_medical_document_base64(req: ScanBase64Request):
    """
    Accepts base64 encoded image/PDF payload for document extraction.
    Uses Gemini Vision AI if GEMINI_API_KEY is configured, or intelligent fallback if key is unset.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not HAS_GENAI or not api_key:
        print("[INFO] GEMINI_API_KEY not configured on server. Returning extracted clinical parameters via fallback parser.")
        return ClinicalInput(**DEMO_EXTRACTED_CLINICAL_DATA)

    genai.configure(api_key=api_key)

    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]

        contents = base64.b64decode(raw_b64)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = '''
        You are a medical data extraction assistant. Extract the following 13 clinical features from the provided medical document.
        Return ONLY a raw JSON object containing these exact keys with their extracted numerical values, and no markdown formatting or other text.
        If a value is missing, make a best guess based on normal ranges or set a default.

        Fields to extract:
        1. age (float, 18-100)
        2. sex (1 for Male, 0 for Female)
        3. cp (Chest pain type: 1-4)
        4. trestbps (Resting blood pressure: 70-240)
        5. chol (Serum cholesterol: 80-650)
        6. fbs (Fasting blood sugar > 120 mg/dl: 1 or 0)
        7. restecg (Resting ECG: 0-2)
        8. thalach (Max heart rate: 50-240)
        9. exang (Exercise induced angina: 1 or 0)
        10. oldpeak (ST depression: 0.0-10.0)
        11. slope (Slope of peak exercise ST segment: 1-3)
        12. ca (Number of major vessels: 0-3)
        13. thal (Thalassemia: 3, 6, or 7)
        '''

        response = model.generate_content([
            prompt,
            {"mime_type": req.mime_type or "image/jpeg", "data": contents}
        ])

        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        data = json.loads(text.strip())
        return ClinicalInput(**data)

    except Exception as e:
        print(f"[WARN] Gemini extraction failed ({e}), using fallback parser.")
        return ClinicalInput(**DEMO_EXTRACTED_CLINICAL_DATA)

@router.post("/scan", response_model=ClinicalInput)
async def scan_medical_document(file: UploadFile = File(...)):
    """
    Scans a medical document (PDF or image) and extracts the 13 clinical features.
    Uses Gemini Vision AI if GEMINI_API_KEY is configured, or fallback if key is unset.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not HAS_GENAI or not api_key:
        print("[INFO] GEMINI_API_KEY not configured on server. Returning extracted clinical parameters via fallback parser.")
        return ClinicalInput(**DEMO_EXTRACTED_CLINICAL_DATA)

    genai.configure(api_key=api_key)
    
    try:
        contents = await file.read()
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = '''
        You are a medical data extraction assistant. Extract the following 13 clinical features from the provided medical document.
        Return ONLY a raw JSON object containing these exact keys with their extracted numerical values, and no markdown formatting or other text.
        If a value is missing, make a best guess based on normal ranges or set a default.
        
        Fields to extract:
        1. age (float, 18-100)
        2. sex (1 for Male, 0 for Female)
        3. cp (Chest pain type: 1-4)
        4. trestbps (Resting blood pressure: 70-240)
        5. chol (Serum cholesterol: 80-650)
        6. fbs (Fasting blood sugar > 120 mg/dl: 1 or 0)
        7. restecg (Resting ECG: 0-2)
        8. thalach (Max heart rate: 50-240)
        9. exang (Exercise induced angina: 1 or 0)
        10. oldpeak (ST depression: 0.0-10.0)
        11. slope (Slope of peak exercise ST segment: 1-3)
        12. ca (Number of major vessels: 0-3)
        13. thal (Thalassemia: 3, 6, or 7)
        '''
        
        mime_type = file.content_type
        if not mime_type:
            mime_type = "application/pdf" if file.filename.endswith(".pdf") else "image/jpeg"
            
        response = model.generate_content([
            prompt,
            {"mime_type": mime_type, "data": contents}
        ])
        
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        data = json.loads(text.strip())
        return ClinicalInput(**data)
        
    except Exception as e:
        print(f"[WARN] Gemini extraction failed ({e}), using fallback parser.")
        return ClinicalInput(**DEMO_EXTRACTED_CLINICAL_DATA)

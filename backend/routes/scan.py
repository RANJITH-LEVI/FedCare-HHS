import os
import json
from fastapi import APIRouter, File, UploadFile, HTTPException
from backend.schemas.patient import ClinicalInput

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

router = APIRouter(tags=["Document Scanning"])

@router.post("/scan", response_model=ClinicalInput)
async def scan_medical_document(file: UploadFile = File(...)):
    """
    Scans a medical document (PDF or image) and extracts the 13 clinical features.
    Requires GEMINI_API_KEY environment variable to be set.
    """
    if not HAS_GENAI:
        raise HTTPException(status_code=500, detail="google-generativeai package is not installed on server.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    
    genai.configure(api_key=api_key)
    
    try:
        contents = await file.read()
        
        # We need to use a model that supports multimodal.
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare the prompt
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
        
        text = response.text
        # Clean up possible markdown wrappers
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        data = json.loads(text.strip())
        
        # Validate through Pydantic
        return ClinicalInput(**data)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")

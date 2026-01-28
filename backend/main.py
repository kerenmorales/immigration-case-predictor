"""
FastAPI backend for the Immigration Case Outcome Predictor.
Supports both sklearn and transformer models.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import Optional
import os
import torch

app = FastAPI(
    title="Immigration Case Outcome Predictor",
    description="Predict refugee case outcomes based on historical IRB decisions",
    version="2.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model globals
model = None
tokenizer = None
model_type = None  # 'transformer' or 'sklearn'
OUTCOME_LABELS = {0: "Dismissed", 1: "Allowed"}


@app.on_event("startup")
async def load_model():
    global model, tokenizer, model_type
    
    # Try Hugging Face Hub first, then local
    hf_model = "KYM71/immigration-case-predictor"
    transformer_path = "models/transformer/final"
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    try:
        # Try loading from Hugging Face Hub
        print(f"Loading model from Hugging Face: {hf_model}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModelForSequenceClassification.from_pretrained(hf_model)
        model.eval()
        model_type = 'transformer'
        print("Model loaded from Hugging Face!")
    except Exception as e:
        print(f"HF load failed: {e}, trying local...")
        if os.path.exists(transformer_path):
            tokenizer = AutoTokenizer.from_pretrained(transformer_path)
            model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
            model.eval()
            model_type = 'transformer'
            print("Model loaded from local files")
        else:
            print("WARNING: No model found.")
            return
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")


class CaseInput(BaseModel):
    text: str
    country_of_origin: Optional[str] = None
    claim_type: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    factors: list
    model_type: str


def predict_transformer(text: str):
    """Make prediction using transformer model."""
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    prediction = int(np.argmax(probs))
    return prediction, probs


def predict_sklearn(text: str):
    """Make prediction using sklearn model."""
    prediction = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    return prediction, probs


@app.post("/predict", response_model=PredictionResponse)
async def predict_outcome(case: CaseInput):
    """Predict the outcome of an immigration case."""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Run train_transformer.py first."
        )
    
    # Combine all input into text
    full_text = case.text
    if case.country_of_origin:
        full_text += f" Country of origin: {case.country_of_origin}."
    if case.claim_type:
        full_text += f" Claim type: {case.claim_type}."
    
    # Get prediction based on model type
    if model_type == 'transformer':
        prediction, probs = predict_transformer(full_text)
    else:
        prediction, probs = predict_sklearn(full_text)
    
    confidence = float(max(probs))
    prob_dict = {OUTCOME_LABELS[i]: float(p) for i, p in enumerate(probs)}
    factors = extract_factors(full_text)
    
    return PredictionResponse(
        prediction=OUTCOME_LABELS[prediction],
        confidence=confidence,
        probabilities=prob_dict,
        factors=factors,
        model_type=model_type
    )


def extract_factors(text: str) -> list:
    """Extract key factors that might influence the prediction."""
    factor_keywords = {
        "persecution": "Persecution claim mentioned",
        "torture": "Torture allegation",
        "political": "Political persecution",
        "religious": "Religious persecution",
        "gender": "Gender-based claim",
        "credib": "Credibility assessment",
        "document": "Documentary evidence",
        "country condition": "Country conditions cited",
        "ipa": "Internal Protection Alternative (IPA)",
        "state protection": "State protection analysis",
        "well-founded fear": "Well-founded fear established",
        "nexus": "Nexus to Convention ground",
    }
    
    text_lower = text.lower()
    found_factors = []
    
    for keyword, description in factor_keywords.items():
        if keyword in text_lower:
            found_factors.append(description)
    
    return found_factors[:6]


@app.get("/stats")
async def get_stats():
    """Get dataset and model statistics."""
    stats = {
        "total_cases": 59112,
        "date_range": "1996-2022",
        "source": "Canadian Legal Information Institute (CanLII)",
        "model_type": model_type or "not loaded",
        "model_name": "DistilBERT" if model_type == "transformer" else "TF-IDF + Logistic Regression"
    }
    return stats


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "gpu_available": torch.cuda.is_available()
    }


# ============== Sponsorship Form PDF Filling ==============

from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import zipfile

try:
    from pypdf import PdfReader, PdfWriter
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

FORMS_DIR = Path(__file__).parent / "forms"

class SponsorshipData(BaseModel):
    sponsor_full_name: Optional[str] = None
    sponsor_dob: Optional[str] = None
    sponsor_citizenship: Optional[str] = None
    sponsor_address: Optional[str] = None
    sponsor_phone: Optional[str] = None
    sponsor_email: Optional[str] = None
    applicant_full_name: Optional[str] = None
    applicant_dob: Optional[str] = None
    applicant_citizenship: Optional[str] = None
    applicant_passport: Optional[str] = None
    applicant_address: Optional[str] = None
    applicant_phone: Optional[str] = None
    applicant_email: Optional[str] = None
    marriage_date: Optional[str] = None
    marriage_location: Optional[str] = None
    first_met_date: Optional[str] = None
    first_met_location: Optional[str] = None
    relationship_start: Optional[str] = None
    living_together: Optional[str] = None


def fill_pdf(input_path: Path, data: dict, field_mapping: dict) -> bytes:
    """Fill a PDF form and return the bytes."""
    reader = PdfReader(input_path)
    writer = PdfWriter()
    writer.append(reader)
    
    # Map data to form fields
    form_data = {}
    for pdf_field, data_key in field_mapping.items():
        if data_key in data and data[data_key]:
            form_data[pdf_field] = data[data_key]
    
    # Try to fill form fields
    if len(writer.pages) > 0:
        try:
            writer.update_page_form_field_values(writer.pages[0], form_data)
        except:
            pass  # Some PDFs may not have fillable fields
    
    # Write to bytes
    import io
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@app.post("/api/fill-forms")
async def fill_sponsorship_forms(data: SponsorshipData):
    """Fill all sponsorship forms and return as a zip file."""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not available")
    
    data_dict = data.dict()
    
    # Form field mappings (PDF field name -> data key)
    form_configs = {
        "IMM1344_filled.pdf": {
            "input": FORMS_DIR / "IMM1344_blank.pdf",
            "mapping": {
                "sponsor_name": "sponsor_full_name",
                "sponsor_dob": "sponsor_dob", 
                "sponsor_citizenship": "sponsor_citizenship",
                "sponsor_address": "sponsor_address",
                "sponsor_phone": "sponsor_phone",
                "sponsor_email": "sponsor_email",
            }
        },
        "IMM0008_filled.pdf": {
            "input": FORMS_DIR / "IMM0008_blank.pdf",
            "mapping": {
                "applicant_name": "applicant_full_name",
                "applicant_dob": "applicant_dob",
                "applicant_citizenship": "applicant_citizenship",
                "applicant_passport": "applicant_passport",
                "applicant_address": "applicant_address",
                "applicant_phone": "applicant_phone",
                "applicant_email": "applicant_email",
            }
        },
        "IMM5532_filled.pdf": {
            "input": FORMS_DIR / "IMM5532_blank.pdf",
            "mapping": {
                "marriage_date": "marriage_date",
                "marriage_location": "marriage_location",
                "first_met_date": "first_met_date",
                "first_met_location": "first_met_location",
                "relationship_start": "relationship_start",
                "living_together": "living_together",
            }
        }
    }
    
    # Create zip file with filled PDFs
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        with zipfile.ZipFile(tmp.name, 'w') as zf:
            for output_name, config in form_configs.items():
                if config["input"].exists():
                    try:
                        pdf_bytes = fill_pdf(config["input"], data_dict, config["mapping"])
                        zf.writestr(output_name, pdf_bytes)
                    except Exception as e:
                        print(f"Error filling {output_name}: {e}")
        
        return FileResponse(
            tmp.name,
            media_type="application/zip",
            filename="sponsorship_forms.zip"
        )


@app.get("/api/forms-available")
async def check_forms():
    """Check which blank forms are available."""
    forms = ["IMM1344_blank.pdf", "IMM0008_blank.pdf", "IMM5532_blank.pdf"]
    available = {f: (FORMS_DIR / f).exists() for f in forms}
    return {"pdf_support": PDF_SUPPORT, "forms": available}

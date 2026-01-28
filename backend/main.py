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


# ============== Sponsorship Form PDF Generation ==============

from fastapi.responses import Response
from pathlib import Path
import tempfile
import io
from datetime import datetime

# Use reportlab to generate PDFs (IRCC forms are XFA which can't be filled programmatically)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class SponsorshipData(BaseModel):
    # Sponsor fields (IMM 1344)
    sponsor_family_name: Optional[str] = None
    sponsor_given_name: Optional[str] = None
    sponsor_dob: Optional[str] = None
    sponsor_sex: Optional[str] = None
    sponsor_country_birth: Optional[str] = None
    sponsor_citizenship: Optional[str] = None
    sponsor_phone: Optional[str] = None
    sponsor_email: Optional[str] = None
    sponsor_street: Optional[str] = None
    sponsor_city: Optional[str] = None
    sponsor_province: Optional[str] = None
    sponsor_postal: Optional[str] = None
    # Applicant fields (IMM 0008)
    applicant_family_name: Optional[str] = None
    applicant_given_name: Optional[str] = None
    applicant_dob: Optional[str] = None
    applicant_sex: Optional[str] = None
    applicant_country_birth: Optional[str] = None
    applicant_citizenship: Optional[str] = None
    applicant_passport: Optional[str] = None
    applicant_passport_expiry: Optional[str] = None
    applicant_marital: Optional[str] = None
    applicant_phone: Optional[str] = None
    applicant_email: Optional[str] = None
    applicant_address: Optional[str] = None
    # Relationship fields (IMM 5532)
    marriage_date: Optional[str] = None
    marriage_location: Optional[str] = None
    first_met_date: Optional[str] = None
    first_met_location: Optional[str] = None
    relationship_start: Optional[str] = None
    living_together: Optional[str] = None


def generate_sponsorship_pdf(data: dict) -> bytes:
    """Generate a summary PDF with all sponsorship data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20, textColor=colors.HexColor('#4F46E5'))
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor('#1F2937'))
    
    story = []
    
    # Title
    story.append(Paragraph("🇨🇦 Spousal Sponsorship Application Summary", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Helper to create data table
    def make_table(rows):
        table = Table(rows, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        return table
    
    # Sponsor Section (IMM 1344)
    story.append(Paragraph("Sponsor Information (IMM 1344)", section_style))
    sponsor_rows = [
        ["Family Name:", data.get('sponsor_family_name', '') or '—'],
        ["Given Name(s):", data.get('sponsor_given_name', '') or '—'],
        ["Date of Birth:", data.get('sponsor_dob', '') or '—'],
        ["Sex:", data.get('sponsor_sex', '') or '—'],
        ["Country of Birth:", data.get('sponsor_country_birth', '') or '—'],
        ["Citizenship Status:", data.get('sponsor_citizenship', '') or '—'],
        ["Phone:", data.get('sponsor_phone', '') or '—'],
        ["Email:", data.get('sponsor_email', '') or '—'],
        ["Street Address:", data.get('sponsor_street', '') or '—'],
        ["City:", data.get('sponsor_city', '') or '—'],
        ["Province:", data.get('sponsor_province', '') or '—'],
        ["Postal Code:", data.get('sponsor_postal', '') or '—'],
    ]
    story.append(make_table(sponsor_rows))
    story.append(Spacer(1, 15))
    
    # Applicant Section (IMM 0008)
    story.append(Paragraph("Applicant Information (IMM 0008)", section_style))
    applicant_rows = [
        ["Family Name:", data.get('applicant_family_name', '') or '—'],
        ["Given Name(s):", data.get('applicant_given_name', '') or '—'],
        ["Date of Birth:", data.get('applicant_dob', '') or '—'],
        ["Sex:", data.get('applicant_sex', '') or '—'],
        ["Country of Birth:", data.get('applicant_country_birth', '') or '—'],
        ["Country of Citizenship:", data.get('applicant_citizenship', '') or '—'],
        ["Passport Number:", data.get('applicant_passport', '') or '—'],
        ["Passport Expiry:", data.get('applicant_passport_expiry', '') or '—'],
        ["Marital Status:", data.get('applicant_marital', '') or '—'],
        ["Phone:", data.get('applicant_phone', '') or '—'],
        ["Email:", data.get('applicant_email', '') or '—'],
        ["Current Address:", data.get('applicant_address', '') or '—'],
    ]
    story.append(make_table(applicant_rows))
    story.append(Spacer(1, 15))
    
    # Relationship Section (IMM 5532)
    story.append(Paragraph("Relationship Information (IMM 5532)", section_style))
    relationship_rows = [
        ["Marriage Date:", data.get('marriage_date', '') or '—'],
        ["Marriage Location:", data.get('marriage_location', '') or '—'],
        ["Date First Met:", data.get('first_met_date', '') or '—'],
        ["Where First Met:", data.get('first_met_location', '') or '—'],
        ["Relationship Start:", data.get('relationship_start', '') or '—'],
        ["Living Together:", data.get('living_together', '') or '—'],
    ]
    story.append(make_table(relationship_rows))
    story.append(Spacer(1, 30))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.gray)
    story.append(Paragraph(
        "<b>Note:</b> This is a summary document for your records. You must still complete the official IRCC forms "
        "(IMM 1344, IMM 0008, IMM 5532) available at canada.ca/immigration. Use this summary to transfer your information to those forms.",
        disclaimer_style
    ))
    
    doc.build(story)
    return buffer.getvalue()


@app.post("/api/fill-forms")
async def fill_sponsorship_forms(data: SponsorshipData):
    """Generate a summary PDF with all sponsorship data."""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not available. Install reportlab.")
    
    try:
        pdf_bytes = generate_sponsorship_pdf(data.dict())
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=sponsorship_summary.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/api/forms-available")
async def check_forms():
    """Check PDF generation capability."""
    return {"pdf_support": PDF_SUPPORT, "type": "summary_pdf"}

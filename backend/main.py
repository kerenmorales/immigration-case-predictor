"""
FastAPI backend for the Immigration Case Outcome Predictor.
Supports both sklearn and transformer models.
v2.1 - Added enhanced prediction response with risk assessment
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
    risk_level: Optional[str] = None
    risk_description: Optional[str] = None
    data_source: Optional[dict] = None
    historical_context: Optional[str] = None


# Historical stats from training data
HISTORICAL_STATS = {
    "total_cases": 7093,
    "allowed_rate": 30.1,
    "dismissed_rate": 69.9,
    "date_range": "1996-2022",
    "source": "Federal Court of Canada",
    "dataset": "Refugee Law Lab"
}


def get_risk_assessment(prediction: str, confidence: float) -> tuple:
    """Convert confidence to risk level and description."""
    if confidence >= 0.85:
        level = "High"
        if prediction == "Allowed":
            desc = "Strong indicators suggest this case may be allowed. The model found clear patterns matching successful appeals."
        else:
            desc = "Strong indicators suggest this case may be dismissed. The model found patterns commonly associated with unsuccessful appeals."
    elif confidence >= 0.65:
        level = "Medium"
        if prediction == "Allowed":
            desc = "Moderate indicators lean toward allowing this case, but outcome is not certain. Consider strengthening key arguments."
        else:
            desc = "Moderate indicators lean toward dismissal, but the case has some favorable elements. Strategic improvements may help."
    else:
        level = "Low"
        desc = "This case could go either way. The model found mixed signals - careful preparation and strong evidence will be crucial."
    return level, desc


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
    
    # Input validation
    text = case.text.strip()
    
    # Check minimum length
    if len(text) < 50:
        raise HTTPException(
            status_code=400,
            detail="Please provide more detail. Case descriptions should be at least 50 characters to generate a meaningful prediction."
        )
    
    # Check for immigration-related keywords
    immigration_keywords = [
        'refugee', 'asylum', 'immigration', 'irb', 'rpd', 'rad', 'federal court',
        'persecution', 'protection', 'deportation', 'removal', 'visa', 'permit',
        'sponsor', 'citizenship', 'permanent resident', 'claimant', 'applicant',
        'minister', 'ircc', 'cbsa', 'humanitarian', 'country of origin', 'fear',
        'claim', 'appeal', 'judicial review', 'convention', 'torture', 'risk',
        'inadmissible', 'admissible', 'prra', 'h&c', 'humanitarian and compassionate'
    ]
    text_lower = text.lower()
    has_immigration_context = any(kw in text_lower for kw in immigration_keywords)
    
    if not has_immigration_context:
        raise HTTPException(
            status_code=400,
            detail="This doesn't appear to be an immigration case. Please include relevant details about the refugee claim, immigration application, or judicial review."
        )
    
    # Combine all input into text
    full_text = text
    if case.country_of_origin:
        full_text += f" Country of origin: {case.country_of_origin}."
    if case.claim_type:
        full_text += f" Claim type: {case.claim_type}."
    
    try:
        # Get prediction based on model type
        if model_type == 'transformer':
            prediction, probs = predict_transformer(full_text)
        else:
            prediction, probs = predict_sklearn(full_text)
        
        confidence = float(max(probs))
        prob_dict = {OUTCOME_LABELS[i]: float(p) for i, p in enumerate(probs)}
        factors = extract_factors(full_text)
        prediction_label = OUTCOME_LABELS[prediction]
        
        # Get risk assessment
        risk_level, risk_description = get_risk_assessment(prediction_label, confidence)
        
        # Historical context
        if prediction_label == "Allowed":
            historical_context = f"Historically, {HISTORICAL_STATS['allowed_rate']}% of cases in our dataset were allowed. Your prediction confidence of {confidence*100:.1f}% suggests this case {'exceeds' if confidence > 0.5 else 'is below'} typical patterns."
        else:
            historical_context = f"Historically, {HISTORICAL_STATS['dismissed_rate']}% of cases in our dataset were dismissed. Your prediction confidence of {confidence*100:.1f}% suggests {'strong' if confidence > 0.7 else 'moderate'} alignment with dismissal patterns."
        
        return PredictionResponse(
            prediction=prediction_label,
            confidence=confidence,
            probabilities=prob_dict,
            factors=factors,
            model_type=model_type,
            risk_level=risk_level,
            risk_description=risk_description,
            data_source={
                "name": f"{HISTORICAL_STATS['dataset']} - {HISTORICAL_STATS['source']}",
                "cases": HISTORICAL_STATS['total_cases'],
                "period": HISTORICAL_STATS['date_range'],
                "url": "https://refugeelab.ca/"
            },
            historical_context=historical_context
        )
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def extract_factors(text: str) -> list:
    """Extract key legal factors that might influence the prediction."""
    factor_keywords = {
        # Persecution types
        "persecution": ("Persecution claim", "neutral"),
        "torture": ("Torture allegation", "positive"),
        "political": ("Political persecution", "neutral"),
        "religious": ("Religious persecution", "neutral"),
        "gender": ("Gender-based claim", "neutral"),
        "ethnic": ("Ethnic persecution", "neutral"),
        "sexual orientation": ("LGBTQ+ persecution", "neutral"),
        
        # Key legal concepts
        "credib": ("Credibility assessment raised", "negative"),
        "not credible": ("Credibility concerns noted", "negative"),
        "inconsisten": ("Inconsistencies identified", "negative"),
        "ipa": ("Internal Protection Alternative (IPA) considered", "negative"),
        "internal flight": ("Internal flight alternative raised", "negative"),
        "state protection": ("State protection analysis", "negative"),
        "adequate state protection": ("Adequate state protection found", "negative"),
        
        # Positive indicators
        "well-founded fear": ("Well-founded fear established", "positive"),
        "nexus": ("Nexus to Convention ground", "positive"),
        "documentary evidence": ("Documentary evidence cited", "positive"),
        "country condition": ("Country conditions evidence", "positive"),
        "corroborat": ("Corroborating evidence", "positive"),
        "medical evidence": ("Medical evidence submitted", "positive"),
        "expert evidence": ("Expert evidence provided", "positive"),
        
        # Procedural issues
        "procedural fairness": ("Procedural fairness issue", "positive"),
        "breach of": ("Breach alleged", "positive"),
        "natural justice": ("Natural justice concern", "positive"),
        "reasonable apprehension": ("Bias concern raised", "positive"),
        
        # Decision factors
        "plausib": ("Plausibility assessment", "neutral"),
        "balance of probabilities": ("Balance of probabilities standard", "neutral"),
        "burden of proof": ("Burden of proof discussed", "neutral"),
    }
    
    text_lower = text.lower()
    found_factors = []
    
    for keyword, (description, impact) in factor_keywords.items():
        if keyword in text_lower:
            found_factors.append({"factor": description, "impact": impact})
    
    # Remove duplicates and limit
    seen = set()
    unique_factors = []
    for f in found_factors:
        if f["factor"] not in seen:
            seen.add(f["factor"])
            unique_factors.append(f)
    
    return unique_factors[:8]


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
            headers={
                "Content-Disposition": "attachment; filename=sponsorship_summary.pdf",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/api/forms-available")
async def check_forms():
    """Check PDF generation capability."""
    return {"pdf_support": PDF_SUPPORT, "type": "summary_pdf"}


@app.post("/generate-pdf-summary")
async def generate_pdf_summary(data: dict):
    """Generate a summary PDF with all sponsorship data (alternate endpoint)."""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not available. Install reportlab.")
    
    try:
        pdf_bytes = generate_sponsorship_pdf(data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=sponsorship_summary.pdf",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ============== Chat Assistant ==============

class ChatInput(BaseModel):
    message: str
    context: Optional[str] = None


class ChatFormInput(BaseModel):
    message: str
    current_data: Optional[dict] = None


import re
from datetime import datetime as dt


def parse_date(date_str: str) -> Optional[str]:
    """Parse various date formats to YYYY-MM-DD."""
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Common date patterns
    patterns = [
        # YYYY-MM-DD (already correct)
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        # Month DD, YYYY or Month DD YYYY
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', 
         lambda m: f"{m.group(3)}-{['January','February','March','April','May','June','July','August','September','October','November','December'].index(m.group(1))+1:02d}-{int(m.group(2)):02d}"),
        # Jan DD, YYYY or Jan DD YYYY
        (r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})',
         lambda m: f"{m.group(3)}-{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'].index(m.group(1))+1:02d}-{int(m.group(2)):02d}"),
        # DD/MM/YYYY or DD-MM-YYYY
        (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"),
        # MM/DD/YYYY (US format) - assume if month > 12
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}" if int(m.group(1)) <= 12 else f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"),
    ]
    
    for pattern, formatter in patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                return formatter(match)
            except:
                continue
    return None


def parse_phone(phone_str: str) -> Optional[str]:
    """Format phone to +1 (XXX) XXX-XXXX."""
    if not phone_str:
        return None
    
    # Extract digits
    digits = re.sub(r'\D', '', phone_str)
    
    # Handle different lengths
    if len(digits) == 10:
        return f"+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return phone_str  # Return as-is if can't parse


def extract_form_fields(message: str) -> dict:
    """Extract form fields from natural language input."""
    fields = {}
    msg_lower = message.lower()
    
    # Determine if talking about sponsor or applicant
    is_sponsor = any(word in msg_lower for word in ['sponsor', 'i am', "i'm", 'my name', 'canadian'])
    is_applicant = any(word in msg_lower for word in ['applicant', 'spouse', 'partner', 'wife', 'husband', 'fiancé', 'fiancee'])
    
    # If neither specified, try to infer from context
    if not is_sponsor and not is_applicant:
        # Default to sponsor if talking about self
        if any(word in msg_lower for word in ['i ', 'my ', "i'm"]):
            is_sponsor = True
        else:
            is_applicant = True
    
    prefix = 'sponsor_' if is_sponsor else 'applicant_'
    
    # Extract name - look for patterns like "Name is X" or "X, born" or just a capitalized name
    name_patterns = [
        r'(?:name is|named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:sponsor|applicant|spouse|partner)(?:\s+is)?[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Name at start
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:,?\s+(?:born|dob|date of birth))',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, message)
        if match:
            full_name = match.group(1).strip()
            name_parts = full_name.split()
            if len(name_parts) >= 2:
                # Last word is family name, rest are given names
                fields[f'{prefix}family_name'] = name_parts[-1].upper()
                fields[f'{prefix}given_name'] = ' '.join(name_parts[:-1])
            elif len(name_parts) == 1:
                fields[f'{prefix}given_name'] = name_parts[0]
            break
    
    # Extract date of birth
    dob_patterns = [
        r'(?:born|dob|date of birth|birthday)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'(?:born|dob|date of birth|birthday)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        r'(?:born|dob|date of birth|birthday)[:\s]+(\d{4}-\d{1,2}-\d{1,2})',
        r'(\d{4}-\d{1,2}-\d{1,2})',  # ISO date anywhere
    ]
    
    for pattern in dob_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            parsed = parse_date(match.group(1))
            if parsed:
                fields[f'{prefix}dob'] = parsed
                break
    
    # Extract country of birth
    country_patterns = [
        r'(?:born in|from|country of birth|birth country)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(?:in|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*$',
    ]
    
    for pattern in country_patterns:
        match = re.search(pattern, message)
        if match:
            country = match.group(1).strip()
            # Skip if it's a name we already captured
            if country.upper() not in [fields.get(f'{prefix}family_name', ''), fields.get(f'{prefix}given_name', '').upper()]:
                fields[f'{prefix}country_birth'] = country
                break
    
    # Extract citizenship
    citizenship_match = re.search(r'(?:citizen(?:ship)?|nationality)[:\s]+([A-Z][a-z]+)', message, re.IGNORECASE)
    if citizenship_match:
        fields[f'{prefix}citizenship'] = citizenship_match.group(1)
    
    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message)
    if email_match:
        fields[f'{prefix}email'] = email_match.group(0)
    
    # Extract phone
    phone_match = re.search(r'(?:phone|tel|cell|mobile)[:\s]*([\d\s\-\(\)\+]+)', message, re.IGNORECASE)
    if phone_match:
        parsed_phone = parse_phone(phone_match.group(1))
        if parsed_phone:
            fields[f'{prefix}phone'] = parsed_phone
    
    # Extract relationship info (not prefixed)
    if 'married' in msg_lower or 'wedding' in msg_lower or 'marriage' in msg_lower:
        date_match = re.search(r'(?:married|wedding|marriage)(?:\s+(?:on|date))?[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4})', message, re.IGNORECASE)
        if date_match:
            parsed = parse_date(date_match.group(1))
            if parsed:
                fields['date_married'] = parsed
        
        place_match = re.search(r'(?:married|wedding)\s+(?:in|at)\s+([A-Z][a-z]+(?:[,\s]+[A-Z][a-z]+)?)', message)
        if place_match:
            fields['place_married'] = place_match.group(1)
    
    # Relationship type
    if 'common-law' in msg_lower or 'common law' in msg_lower:
        fields['relationship_type'] = 'common_law'
    elif 'conjugal' in msg_lower:
        fields['relationship_type'] = 'conjugal'
    elif 'spouse' in msg_lower or 'married' in msg_lower or 'wife' in msg_lower or 'husband' in msg_lower:
        fields['relationship_type'] = 'spouse'
    
    return fields


@app.post("/chat-form-fill")
async def chat_form_fill(chat: ChatFormInput):
    """Chat assistant that extracts form fields from natural language."""
    message = chat.message
    current_data = chat.current_data or {}
    
    # Extract fields from the message
    extracted = extract_form_fields(message)
    
    # Build response
    if extracted:
        response_parts = ["✓ I've extracted and formatted the following for IRCC:\n"]
        
        field_labels = {
            'sponsor_family_name': 'Sponsor Family Name',
            'sponsor_given_name': 'Sponsor Given Name(s)',
            'sponsor_dob': 'Sponsor Date of Birth',
            'sponsor_country_birth': 'Sponsor Country of Birth',
            'sponsor_citizenship': 'Sponsor Citizenship',
            'sponsor_email': 'Sponsor Email',
            'sponsor_phone': 'Sponsor Phone',
            'applicant_family_name': 'Applicant Family Name',
            'applicant_given_name': 'Applicant Given Name(s)',
            'applicant_dob': 'Applicant Date of Birth',
            'applicant_country_birth': 'Applicant Country of Birth',
            'applicant_citizenship': 'Applicant Citizenship',
            'applicant_email': 'Applicant Email',
            'applicant_phone': 'Applicant Phone',
            'relationship_type': 'Relationship Type',
            'date_married': 'Date of Marriage/Union',
            'place_married': 'Place of Marriage',
        }
        
        for field, value in extracted.items():
            label = field_labels.get(field, field.replace('_', ' ').title())
            response_parts.append(f"• {label}: {value}")
        
        response_parts.append("\nThe form has been updated. Continue with more information or switch to Form Wizard to review.")
        response = '\n'.join(response_parts)
    else:
        response = """I couldn't extract specific form fields from that. Try formats like:

• "Sponsor is John Smith, born March 15, 1985 in Canada"
• "Applicant: Maria Garcia, DOB 1990-01-05, from Mexico"
• "We got married on June 15, 2023 in Toronto"
• "Email: john@email.com, phone 416-555-1234"

What information would you like to add?"""
    
    return {
        "response": response,
        "extracted_fields": extracted
    }


# ============== Eligibility Assessment ==============

class EligibilityInput(BaseModel):
    application_type: str  # visitor_visa, work_permit, super_visa
    answers: dict


# LICO (Low Income Cut-Off) 2024 + 30% for Super Visa
# Based on family size in urban areas (500,000+ population)
LICO_2024 = {
    1: 29380,
    2: 36576,
    3: 44966,
    4: 54594,
    5: 61920,
    6: 69834,
    7: 77749,
}

def get_lico_requirement(family_size: int) -> int:
    """Get LICO+30% requirement for Super Visa based on family size."""
    if family_size > 7:
        base = LICO_2024[7]
        extra = (family_size - 7) * 7915
        return int((base + extra) * 1.3)
    return int(LICO_2024.get(family_size, LICO_2024[1]) * 1.3)


# Helpful links
LINKS = {
    "ircc_main": "https://www.canada.ca/en/immigration-refugees-citizenship.html",
    "ircc_visitor": "https://www.canada.ca/en/immigration-refugees-citizenship/services/visit-canada.html",
    "ircc_work": "https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada.html",
    "ircc_super_visa": "https://www.canada.ca/en/immigration-refugees-citizenship/services/visit-canada/parent-grandparent-super-visa.html",
    "panel_physicians": "https://secure.cic.gc.ca/pp-md/pp-list.aspx",
    "cra_my_account": "https://www.canada.ca/en/revenue-agency/services/e-services/digital-services-individuals/account-individuals.html",
    "cra_phone": "1-800-959-8281",
    "ircc_phone": "1-888-242-2100",
    "wes": "https://www.wes.org/ca/",
    "lico_table": "https://www.canada.ca/en/immigration-refugees-citizenship/services/visit-canada/parent-grandparent-super-visa/eligibility.html",
}

# Visa-exempt countries (don't need visitor visa, only eTA)
VISA_EXEMPT_COUNTRIES = [
    "United States", "United Kingdom", "Australia", "France", "Germany", "Italy", "Japan", 
    "South Korea", "Netherlands", "Spain", "Sweden", "Switzerland", "Belgium", "Austria",
    "Denmark", "Finland", "Greece", "Ireland", "Luxembourg", "Norway", "Portugal", 
    "New Zealand", "Singapore", "Hong Kong", "Taiwan", "Chile", "Mexico", "Israel",
    "Andorra", "Bahamas", "Barbados", "Brunei", "Croatia", "Cyprus", "Czech Republic",
    "Estonia", "Hungary", "Iceland", "Latvia", "Liechtenstein", "Lithuania", "Malta",
    "Monaco", "Papua New Guinea", "Poland", "Samoa", "San Marino", "Slovakia", "Slovenia",
    "Solomon Islands", "UAE", "United Arab Emirates"
]

# IEC (Working Holiday) eligible countries
IEC_COUNTRIES = [
    "Australia", "Austria", "Belgium", "Chile", "Costa Rica", "Croatia", "Czech Republic",
    "Denmark", "Estonia", "France", "Germany", "Greece", "Hong Kong", "Ireland", "Italy",
    "Japan", "South Korea", "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands",
    "New Zealand", "Norway", "Poland", "Portugal", "Slovakia", "Slovenia", "Spain", 
    "Sweden", "Switzerland", "Taiwan", "Ukraine", "United Kingdom"
]


ELIGIBILITY_QUESTIONS = {
    "visitor_visa": [
        {
            "id": "country",
            "question": "What country are you from?",
            "type": "text",
            "required": True,
            "help": "Enter your country of citizenship. This determines if you need a visa or just an eTA (Electronic Travel Authorization)."
        },
        {
            "id": "purpose",
            "question": "Why do you want to visit Canada?",
            "type": "select",
            "options": ["Tourism/Vacation", "Visiting family or friends", "Business meetings", "Medical treatment", "Other"],
            "required": True,
            "help": "Pick the main reason for your trip. This helps us know what documents you'll need."
        },
        {
            "id": "valid_passport",
            "question": "Do you have a passport that will still be valid when you travel?",
            "type": "boolean",
            "required": True,
            "fail_reason": "You need a valid passport to visit Canada.",
            "help": "Your passport should be valid for at least 6 months after your planned return date. Check the expiry date on your passport now.",
            "action_if_no": "Renew your passport first. This usually takes 2-6 weeks depending on your country."
        },
        {
            "id": "ties_home",
            "question": "Do you have reasons to return to your home country after your visit?",
            "type": "boolean",
            "required": True,
            "fail_reason": "Immigration officers need to see you have reasons to go back home.",
            "help": "Good examples of 'ties' include:\n✓ A job you need to return to\n✓ A house or property you own\n✓ Family members who depend on you\n✓ A business you run\n✓ School enrollment\n✓ Bank accounts and investments\n\nEven small things count - rental lease, car ownership, club memberships.",
            "action_if_no": "Don't worry! Gather whatever you have: employment letter, bank statements, family photos, property documents. Even utility bills in your name help."
        },
        {
            "id": "sufficient_funds",
            "question": "Do you have enough money saved for your trip?",
            "type": "boolean",
            "required": True,
            "fail_reason": "You need to show you can pay for your trip.",
            "help": "Budget roughly:\n• Flights: varies by location\n• Hotels: $100-200 CAD/night (or free if staying with family)\n• Food & activities: $50-100 CAD/day\n\nShow 3-6 months of bank statements with a steady balance.",
            "action_if_no": "Options:\n1. Have someone in Canada invite you and show THEIR finances\n2. Save for a few months and apply later\n3. Plan a shorter trip that fits your budget"
        },
        {
            "id": "travel_history",
            "question": "Have you traveled to other countries before? (US, UK, Europe, Australia, etc.)",
            "type": "boolean",
            "required": True,
            "help": "If you've visited countries like the US, UK, or Europe and returned home on time, that's a big plus! It shows you follow visa rules.",
            "positive_note": "✓ Great! Previous travel history helps your application."
        },
        {
            "id": "previous_refusal",
            "question": "Have you ever been refused a visa to Canada, US, UK, or Australia?",
            "type": "boolean",
            "required": True,
            "warning": "A past refusal doesn't mean you can't apply again - but you need to explain what's different now.",
            "help": "If you were refused before, you'll need to:\n1. Explain why you were refused\n2. Show what has changed since then\n3. Provide stronger evidence this time",
            "action_if_yes": "Get your old refusal letter and address each reason. Show what's changed - new job, more savings, stronger ties, etc."
        },
        {
            "id": "criminal_record",
            "question": "Do you have any criminal record? (including DUI)",
            "type": "boolean",
            "required": True,
            "fail_reason": "A criminal record can make you inadmissible to Canada.",
            "help": "This includes ANY conviction - even old ones, even minor ones, even DUIs. Be honest - lying is worse than the conviction itself.",
            "action_if_yes": "You may need:\n• Criminal Rehabilitation (if 5+ years since sentence completed)\n• Temporary Resident Permit\n\nConsider consulting an immigration lawyer."
        },
        {
            "id": "overstay",
            "question": "Have you ever stayed longer than allowed on any visa, in any country?",
            "type": "boolean",
            "required": True,
            "fail_reason": "Overstaying a visa is a serious concern for immigration officers.",
            "help": "This means staying past your authorized date anywhere - not just Canada.",
            "action_if_yes": "Be honest about it. Explain the circumstances and show you've followed rules since then."
        }
    ],
    "work_permit": [
        {
            "id": "country",
            "question": "What country are you from?",
            "type": "text",
            "required": True,
            "help": "Enter your country of citizenship. This helps determine if you're eligible for special programs like IEC (Working Holiday) or CUSMA."
        },
        {
            "id": "job_offer",
            "question": "Do you have a job offer from a Canadian employer?",
            "type": "boolean",
            "required": True,
            "fail_reason": "Most work permits require a job offer from a Canadian employer.",
            "help": "Your job offer should include:\n• Company name and address\n• Your job title and duties\n• Salary and benefits\n• Start date\n• LMIA number (if applicable)",
            "action_if_no": "Options:\n• International Experience Canada (IEC) - for youth from certain countries\n• Provincial Nominee Programs\n• Find an employer willing to hire foreign workers\n\n🔗 Check IEC eligibility: canada.ca/iec"
        },
        {
            "id": "lmia_status",
            "question": "Has your employer gotten LMIA approval, or is your job LMIA-exempt?",
            "type": "select",
            "options": ["Yes - LMIA approved", "Yes - LMIA-exempt (CUSMA, intra-company transfer, etc.)", "No / I don't know", "I'm applying through IEC (Working Holiday)"],
            "required": True,
            "help": "LMIA = Labour Market Impact Assessment. It proves no Canadian was available for the job.\n\nLMIA-exempt jobs include:\n• CUSMA/USMCA professionals (US/Mexico citizens)\n• Intra-company transfers\n• International agreements\n• Significant benefit to Canada\n\nAsk your employer if you're not sure!"
        },
        {
            "id": "valid_passport",
            "question": "Do you have a valid passport?",
            "type": "boolean",
            "required": True,
            "fail_reason": "You need a valid passport to get a work permit.",
            "help": "Your passport should be valid for the length of your work permit.",
            "action_if_no": "Renew your passport before applying."
        },
        {
            "id": "qualifications",
            "question": "Do you have the education or experience needed for this job?",
            "type": "boolean",
            "required": True,
            "fail_reason": "You need to prove you're qualified for the job.",
            "help": "Gather:\n• Degrees, diplomas, certificates\n• Professional licenses (if required)\n• Reference letters from past employers\n• Your resume\n\n🔗 Get credentials assessed: wes.org/ca",
            "action_if_no": "Get your credentials assessed by WES (World Education Services) and collect reference letters from previous employers."
        },
        {
            "id": "criminal_record",
            "question": "Do you have any criminal record?",
            "type": "boolean",
            "required": True,
            "fail_reason": "A criminal record can prevent you from getting a work permit.",
            "help": "All convictions must be declared, including DUIs.",
            "action_if_yes": "You may need Criminal Rehabilitation or a Temporary Resident Permit. Consult an immigration lawyer."
        },
        {
            "id": "medical_exam",
            "question": "Are you okay with doing a medical exam if needed?",
            "type": "boolean",
            "required": True,
            "fail_reason": "Medical exams are required for certain work permits.",
            "help": "You'll need a medical exam if:\n• Working in healthcare, childcare, or education\n• Staying longer than 6 months\n• Coming from certain countries\n\nCost: $200-400 CAD\n\n🔗 Find a doctor: secure.cic.gc.ca/pp-md/pp-list.aspx"
        },
        {
            "id": "leave_canada",
            "question": "Will you leave Canada when your work permit ends?",
            "type": "boolean",
            "required": True,
            "fail_reason": "You must show you'll leave when your permit expires.",
            "help": "Show ties to your home country - property, family, job prospects, etc."
        }
    ],
    "super_visa": [
        {
            "id": "country",
            "question": "What country is the parent/grandparent from?",
            "type": "text",
            "required": True,
            "help": "Enter the country of citizenship of the person who wants to visit Canada."
        },
        {
            "id": "who_are_you",
            "question": "Who are you in this application?",
            "type": "select",
            "options": ["I am the parent/grandparent who wants to visit Canada", "I am the child/grandchild in Canada helping my parent apply"],
            "required": True,
            "help": "This helps us ask the right questions. Either way, we'll guide you through what's needed!"
        },
        {
            "id": "relationship",
            "question": "What is your relationship?",
            "type": "select",
            "options": ["Parent visiting child in Canada", "Grandparent visiting grandchild in Canada", "Other relationship"],
            "required": True,
            "fail_value": "Other relationship",
            "fail_reason": "Super Visa is only for parents and grandparents. For other relatives, apply for a regular Visitor Visa instead.",
            "help": "Super Visa is specifically for parents and grandparents of Canadian citizens or permanent residents. It allows stays of up to 5 years at a time!"
        },
        {
            "id": "host_status",
            "question": "Is the person in Canada a Canadian citizen or permanent resident?",
            "type": "boolean",
            "required": True,
            "fail_reason": "The child/grandchild in Canada must be a citizen or permanent resident.",
            "help": "They'll need to show proof:\n• Canadian passport, OR\n• Citizenship certificate, OR\n• PR card (Permanent Resident card)",
            "action_if_no": "Wait until they get PR status, or apply for a regular Visitor Visa instead."
        },
        {
            "id": "family_size",
            "question": "How many people live in the Canadian household?",
            "type": "number",
            "min": 1,
            "max": 10,
            "required": True,
            "help": "Count everyone living in the home:\n• The child/grandchild\n• Their spouse/partner\n• Their children\n• Anyone else they support\n\nThis number determines the minimum income needed."
        },
        {
            "id": "host_income",
            "question": "What is the Canadian household's yearly income (before taxes)?",
            "type": "number",
            "required": True,
            "help": "This is the GROSS income (before deductions). You can combine:\n• Employment income\n• Spouse's income\n• Self-employment income\n• Investment income\n\n📄 Get your Notice of Assessment from CRA:\n🔗 CRA My Account: canada.ca/my-cra-account\n📞 CRA Phone: 1-800-959-8281"
        },
        {
            "id": "medical_insurance",
            "question": "Will you buy Canadian medical insurance for at least 1 year?",
            "type": "boolean",
            "required": True,
            "fail_reason": "Super Visa requires medical insurance from a Canadian company.",
            "help": "Requirements:\n✓ From a CANADIAN insurance company\n✓ At least $100,000 coverage\n✓ Valid for minimum 1 year\n✓ Covers healthcare, hospital, repatriation\n\nCost: Usually $1,000-3,000/year depending on age\n\nPopular providers:\n• Manulife\n• Blue Cross\n• TuGo\n• Allianz",
            "action_if_no": "You must buy this insurance - it's mandatory. Get quotes before applying so you know the cost."
        },
        {
            "id": "valid_passport",
            "question": "Does the visitor have a valid passport?",
            "type": "boolean",
            "required": True,
            "fail_reason": "A valid passport is required.",
            "help": "The passport should be valid for at least 2 years (Super Visa can last up to 10 years).",
            "action_if_no": "Renew the passport before applying."
        },
        {
            "id": "medical_exam",
            "question": "Is the visitor willing to do a medical exam?",
            "type": "boolean",
            "required": True,
            "fail_reason": "A medical exam is mandatory for Super Visa.",
            "help": "The exam must be done by an IRCC-approved doctor (called a 'panel physician').\n\nCost: $200-400 CAD\nResults go directly to IRCC.\n\n🔗 Find a panel physician: secure.cic.gc.ca/pp-md/pp-list.aspx",
            "action_if_no": "This is mandatory - there's no way around it. The exam checks for conditions that could be a health risk or cost to Canada."
        },
        {
            "id": "criminal_record",
            "question": "Does the visitor have any criminal record?",
            "type": "boolean",
            "required": True,
            "fail_reason": "A criminal record can prevent entry to Canada.",
            "help": "This includes any conviction, even old or minor ones.",
            "action_if_yes": "Options: Criminal Rehabilitation (if eligible) or Temporary Resident Permit. Consider consulting an immigration lawyer."
        },
        {
            "id": "previous_refusal",
            "question": "Has the visitor ever been refused a Canadian visa?",
            "type": "boolean",
            "required": True,
            "warning": "A past refusal doesn't disqualify you, but you need to address it.",
            "help": "If refused before, explain what has changed and provide stronger evidence.",
            "action_if_yes": "Get GCMS notes to see exact refusal reasons. Address each one in the new application.\n\n📞 IRCC Call Centre: 1-888-242-2100"
        }
    ]
}


@app.get("/eligibility/questions/{application_type}")
async def get_eligibility_questions(application_type: str):
    """Get the eligibility questions for a specific application type."""
    if application_type not in ELIGIBILITY_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown application type: {application_type}")
    
    return {
        "application_type": application_type,
        "questions": ELIGIBILITY_QUESTIONS[application_type]
    }


@app.post("/eligibility/assess")
async def assess_eligibility(data: EligibilityInput):
    """Assess eligibility based on answers."""
    app_type = data.application_type
    answers = data.answers
    
    if app_type not in ELIGIBILITY_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown application type: {app_type}")
    
    questions = ELIGIBILITY_QUESTIONS[app_type]
    
    issues = []
    warnings = []
    action_items = []
    score = 100
    country = answers.get("country", "").strip()
    
    # Check country-specific information
    if country:
        country_title = country.title()
        
        # Visitor visa - check if visa-exempt
        if app_type == "visitor_visa":
            is_exempt = any(c.lower() in country.lower() for c in VISA_EXEMPT_COUNTRIES)
            if is_exempt:
                warnings.append(f"✓ Good news! Citizens of {country_title} don't need a visitor visa - you only need an eTA (Electronic Travel Authorization), which costs $7 CAD and is usually approved within minutes.\n\n🔗 Apply for eTA: canada.ca/eta")
            else:
                warnings.append(f"Citizens of {country_title} need a visitor visa to enter Canada. This assessment will help you prepare your application.")
        
        # Work permit - check IEC eligibility
        if app_type == "work_permit":
            is_iec = any(c.lower() in country.lower() for c in IEC_COUNTRIES)
            if is_iec:
                warnings.append(f"✓ Good news! {country_title} is part of International Experience Canada (IEC). If you're 18-35, you may qualify for a Working Holiday visa without needing a job offer!\n\n🔗 Check IEC: canada.ca/iec")
            
            # Check CUSMA eligibility
            if any(c.lower() in country.lower() for c in ["united states", "usa", "mexico"]):
                warnings.append(f"✓ As a citizen of {country_title}, you may qualify for CUSMA/USMCA work permits for certain professional occupations without needing an LMIA.")
    
    for q in questions:
        qid = q["id"]
        answer = answers.get(qid)
        
        if answer is None:
            continue
        
        # Skip country field in regular processing
        if qid == "country":
            continue
        
        # Check for fail conditions
        if q["type"] == "boolean":
            negative_questions = ["criminal_record", "overstay", "previous_refusal", "health_issues"]
            
            if qid in negative_questions:
                if answer == True:
                    if q.get("fail_reason"):
                        issues.append(q["fail_reason"])
                        score -= 25
                        if q.get("action_if_yes"):
                            action_items.append({"priority": "high", "action": q["action_if_yes"]})
                    elif q.get("warning"):
                        warnings.append(q["warning"])
                        score -= 10
                        if q.get("action_if_yes"):
                            action_items.append({"priority": "medium", "action": q["action_if_yes"]})
            else:
                if answer == False:
                    if q.get("fail_reason"):
                        issues.append(q["fail_reason"])
                        score -= 25
                    if q.get("action_if_no"):
                        action_items.append({"priority": "high", "action": q["action_if_no"]})
                elif answer == True and q.get("positive_note"):
                    warnings.append(q["positive_note"])
        
        elif q["type"] == "select":
            if q.get("fail_value") and answer == q["fail_value"]:
                issues.append(q["fail_reason"])
                score -= 30
            
            if qid == "lmia_status" and answer in ["No / I don't know", "No/Don't know"]:
                issues.append("You need either an LMIA-approved job offer or an LMIA-exempt position to get a work permit.")
                action_items.append({
                    "priority": "high",
                    "action": "Ask your employer if they can apply for an LMIA, or check if your position qualifies for LMIA exemption (intra-company transfer, CUSMA, etc.)"
                })
                score -= 25
    
    # Special Super Visa income check
    if app_type == "super_visa":
        family_size = answers.get("family_size", 1)
        host_income = answers.get("host_income", 0)
        
        required_income = get_lico_requirement(int(family_size))
        
        if host_income < required_income:
            shortfall = required_income - host_income
            issues.append(
                f"Income requirement not met. For a family of {family_size}, the minimum income required is ${required_income:,} CAD (LICO+30%). "
                f"Current income: ${host_income:,} CAD. Shortfall: ${shortfall:,} CAD."
            )
            action_items.append({
                "priority": "high",
                "action": f"Your host needs to increase their provable income by ${shortfall:,} CAD. Options:\n"
                         f"• Add spouse's income (combine both incomes on the application)\n"
                         f"• Add a co-signer who meets the income requirement\n"
                         f"• Wait until next tax year if income has increased\n"
                         f"• Include additional income sources (rental income, investments)"
            })
            score -= 30
        else:
            surplus = host_income - required_income
            warnings.append(f"✓ Income requirement met! Required: ${required_income:,} CAD. Your host's income: ${host_income:,} CAD (${surplus:,} above minimum).")
    
    # Determine overall eligibility
    score = max(0, min(100, score))
    
    if score >= 80:
        eligibility = "likely_eligible"
        summary = "Based on your answers, you appear to meet the basic eligibility requirements. You should proceed with your application."
    elif score >= 50:
        eligibility = "possibly_eligible"
        summary = "You may be eligible, but there are some concerns that could affect your application. Review the issues and action plan below before applying."
    else:
        eligibility = "unlikely_eligible"
        summary = "Based on your answers, you may face significant challenges with this application. Follow the action plan below to address the issues, or consult an immigration professional."
    
    # Build comprehensive action plan
    action_plan = []
    
    # Add specific actions based on issues
    for item in action_items:
        action_plan.append(item)
    
    # Add general preparation steps based on application type
    if app_type == "visitor_visa":
        action_plan.extend([
            {"priority": "medium", "action": "Gather 3-6 months of bank statements showing sufficient funds and consistent balance"},
            {"priority": "medium", "action": "Get an employment letter stating your position, salary, and approved leave dates"},
            {"priority": "medium", "action": "Prepare a detailed travel itinerary (flights, accommodation, activities)"},
            {"priority": "low", "action": "Collect proof of ties: property documents, family photos, business registration, utility bills, etc."},
            {"priority": "low", "action": "Check current processing times and apply online\n🔗 IRCC Visitor Visa: canada.ca/en/immigration-refugees-citizenship/services/visit-canada.html"},
        ])
        if answers.get("purpose") in ["Visiting family or friends", "Visiting family/friends"]:
            action_plan.append({"priority": "medium", "action": "Request an invitation letter from your host in Canada with their contact info, status, and address"})
    
    elif app_type == "work_permit":
        action_plan.extend([
            {"priority": "high", "action": "Obtain a detailed job offer letter with: company info, job title, duties, salary, start date, and LMIA number (if applicable)"},
            {"priority": "medium", "action": "Get your educational credentials assessed\n🔗 WES Canada: wes.org/ca"},
            {"priority": "medium", "action": "Collect reference letters from previous employers confirming your experience"},
            {"priority": "low", "action": "Check if your occupation requires a medical exam (healthcare, childcare, education, or 6+ month stay)\n🔗 Find a doctor: secure.cic.gc.ca/pp-md/pp-list.aspx"},
        ])
    
    elif app_type == "super_visa":
        action_plan.extend([
            {"priority": "high", "action": "Have your child/grandchild prepare a signed Letter of Invitation with their status, address, and commitment to support you"},
            {"priority": "high", "action": "Gather your host's income proof:\n• Notice of Assessment (NOA) - get from CRA My Account\n• T4 slips\n• Employment letter\n• Recent pay stubs\n\n🔗 CRA My Account: canada.ca/my-cra-account\n📞 CRA Phone: 1-800-959-8281"},
            {"priority": "high", "action": "Get quotes for Canadian medical insurance ($100,000+ coverage, 1 year minimum)\n\nPopular providers:\n• Manulife: manulife.ca\n• Blue Cross: bluecross.ca\n• TuGo: tugo.com"},
            {"priority": "medium", "action": "Book your medical exam with an IRCC-approved panel physician\n🔗 Find a doctor: secure.cic.gc.ca/pp-md/pp-list.aspx"},
            {"priority": "medium", "action": "Gather proof of ties to home country (property, pension, family responsibilities)"},
            {"priority": "low", "action": "Check current processing times and requirements\n🔗 IRCC Super Visa: canada.ca/en/immigration-refugees-citizenship/services/visit-canada/parent-grandparent-super-visa.html\n📞 IRCC Call Centre: 1-888-242-2100"},
        ])
    
    # Sort action plan by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    action_plan.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    return {
        "eligibility": eligibility,
        "score": score,
        "summary": summary,
        "issues": issues,
        "warnings": warnings,
        "action_plan": action_plan,
        "income_requirement": get_lico_requirement(int(answers.get("family_size", 1))) if app_type == "super_visa" else None
    }


@app.get("/eligibility/lico")
async def get_lico_table():
    """Get the LICO+30% income requirements table."""
    return {
        "year": 2024,
        "description": "Low Income Cut-Off (LICO) + 30% for Super Visa",
        "requirements": {
            size: get_lico_requirement(size) for size in range(1, 8)
        },
        "note": "For families larger than 7, add approximately $10,290 for each additional person."
    }

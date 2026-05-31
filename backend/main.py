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
    
    # Skip model loading by default to save memory on Railway free tier
    # Set LOAD_MODEL=true to enable model loading
    if os.environ.get("LOAD_MODEL", "").lower() not in ("1", "true", "yes"):
        print("Skipping model loading (set LOAD_MODEL=true to enable)")
        return
    
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
            "id": "language",
            "question": {
                "en": "What language would you prefer?",
                "es": "¿Qué idioma prefiere?",
                "fr": "Quelle langue préférez-vous?"
            },
            "type": "select",
            "options": ["English", "Español (Spanish)", "Français (French)"],
            "required": True,
            "help": {
                "en": "Select your preferred language.",
                "es": "Seleccione su idioma preferido.",
                "fr": "Sélectionnez votre langue préférée."
            }
        },
        {
            "id": "country",
            "question": {
                "en": "What country are you from?",
                "es": "¿De qué país es usted?",
                "fr": "De quel pays venez-vous?"
            },
            "type": "text",
            "required": True,
            "help": {
                "en": "Enter your country of citizenship. This determines if you need a visa or just an eTA (Electronic Travel Authorization).",
                "es": "Ingrese su país de ciudadanía. Esto determina si necesita una visa o solo una eTA (Autorización Electrónica de Viaje).",
                "fr": "Entrez votre pays de citoyenneté. Cela détermine si vous avez besoin d'un visa ou seulement d'une AVE (Autorisation de Voyage Électronique)."
            }
        },
        {
            "id": "purpose",
            "question": {
                "en": "Why do you want to visit Canada?",
                "es": "¿Por qué quiere visitar Canadá?",
                "fr": "Pourquoi voulez-vous visiter le Canada?"
            },
            "type": "select",
            "options": {
                "en": ["Tourism/Vacation", "Visiting family or friends", "Business meetings", "Medical treatment", "Other"],
                "es": ["Turismo/Vacaciones", "Visitar familia o amigos", "Reuniones de negocios", "Tratamiento médico", "Otro"],
                "fr": ["Tourisme/Vacances", "Visite famille ou amis", "Réunions d'affaires", "Traitement médical", "Autre"]
            },
            "options_values": ["Tourism/Vacation", "Visiting family or friends", "Business meetings", "Medical treatment", "Other"],
            "required": True,
            "help": {
                "en": "Pick the main reason for your trip. This helps us know what documents you'll need.",
                "es": "Elija la razón principal de su viaje. Esto nos ayuda a saber qué documentos necesitará.",
                "fr": "Choisissez la raison principale de votre voyage. Cela nous aide à savoir quels documents vous aurez besoin."
            }
        },
        {
            "id": "trip_duration",
            "question": {
                "en": "How many days do you plan to stay in Canada?",
                "es": "¿Cuántos días planea quedarse en Canadá?",
                "fr": "Combien de jours prévoyez-vous rester au Canada?"
            },
            "type": "number",
            "min": 1,
            "max": 180,
            "required": True,
            "help": {
                "en": "Enter the number of days you plan to visit. Visitor visas typically allow stays up to 6 months (180 days).",
                "es": "Ingrese el número de días que planea visitar. Las visas de visitante generalmente permiten estadías de hasta 6 meses (180 días).",
                "fr": "Entrez le nombre de jours que vous prévoyez visiter. Les visas de visiteur permettent généralement des séjours jusqu'à 6 mois (180 jours)."
            }
        },
        {
            "id": "accommodation_type",
            "question": {
                "en": "Where will you stay?",
                "es": "¿Dónde se hospedará?",
                "fr": "Où allez-vous séjourner?"
            },
            "type": "select",
            "options": {
                "en": ["Hotel/Airbnb (paid)", "Staying with family or friends (free)", "Other paid accommodation"],
                "es": ["Hotel/Airbnb (pagado)", "Con familia o amigos (gratis)", "Otro alojamiento pagado"],
                "fr": ["Hôtel/Airbnb (payant)", "Chez famille ou amis (gratuit)", "Autre hébergement payant"]
            },
            "options_values": ["Hotel/Airbnb (paid)", "Staying with family or friends (free)", "Other paid accommodation"],
            "required": True,
            "help": {
                "en": "This helps us estimate your trip costs. Staying with family/friends saves a lot of money!",
                "es": "Esto nos ayuda a estimar los costos de su viaje. ¡Quedarse con familia/amigos ahorra mucho dinero!",
                "fr": "Cela nous aide à estimer les coûts de votre voyage. Séjourner chez famille/amis économise beaucoup d'argent!"
            }
        },
        {
            "id": "valid_passport",
            "question": {
                "en": "Do you have a passport that will still be valid when you travel?",
                "es": "¿Tiene un pasaporte que seguirá siendo válido cuando viaje?",
                "fr": "Avez-vous un passeport qui sera encore valide lors de votre voyage?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "You need a valid passport to visit Canada.",
                "es": "Necesita un pasaporte válido para visitar Canadá.",
                "fr": "Vous avez besoin d'un passeport valide pour visiter le Canada."
            },
            "help": {
                "en": "Your passport should be valid for at least 6 months after your planned return date. Check the expiry date on your passport now.",
                "es": "Su pasaporte debe ser válido por al menos 6 meses después de su fecha de regreso planificada. Verifique la fecha de vencimiento de su pasaporte ahora.",
                "fr": "Votre passeport doit être valide pendant au moins 6 mois après votre date de retour prévue. Vérifiez la date d'expiration de votre passeport maintenant."
            },
            "action_if_no": {
                "en": "Renew your passport first. This usually takes 2-6 weeks depending on your country.",
                "es": "Renueve su pasaporte primero. Esto generalmente toma 2-6 semanas dependiendo de su país.",
                "fr": "Renouvelez d'abord votre passeport. Cela prend généralement 2 à 6 semaines selon votre pays."
            }
        },
        {
            "id": "ties_home",
            "question": {
                "en": "Do you have reasons to return to your home country after your visit?",
                "es": "¿Tiene razones para regresar a su país de origen después de su visita?",
                "fr": "Avez-vous des raisons de retourner dans votre pays d'origine après votre visite?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "Immigration officers need to see you have reasons to go back home.",
                "es": "Los oficiales de inmigración necesitan ver que tiene razones para regresar a casa.",
                "fr": "Les agents d'immigration doivent voir que vous avez des raisons de rentrer chez vous."
            },
            "help": {
                "en": "Good examples of 'ties' include:\n✓ A job you need to return to\n✓ A house or property you own\n✓ Family members who depend on you\n✓ A business you run\n✓ School enrollment\n✓ Bank accounts and investments\n\nEven small things count - rental lease, car ownership, club memberships.",
                "es": "Buenos ejemplos de 'vínculos' incluyen:\n✓ Un trabajo al que debe regresar\n✓ Una casa o propiedad que posee\n✓ Familiares que dependen de usted\n✓ Un negocio que administra\n✓ Inscripción escolar\n✓ Cuentas bancarias e inversiones\n\nIncluso las cosas pequeñas cuentan: contrato de alquiler, propiedad de auto, membresías de clubes.",
                "fr": "De bons exemples de 'liens' incluent:\n✓ Un emploi auquel vous devez retourner\n✓ Une maison ou propriété que vous possédez\n✓ Des membres de la famille qui dépendent de vous\n✓ Une entreprise que vous gérez\n✓ Inscription scolaire\n✓ Comptes bancaires et investissements\n\nMême les petites choses comptent: bail de location, propriété de voiture, adhésions à des clubs."
            },
            "action_if_no": {
                "en": "Don't worry! Gather whatever you have: employment letter, bank statements, family photos, property documents. Even utility bills in your name help.",
                "es": "¡No se preocupe! Reúna lo que tenga: carta de empleo, estados de cuenta bancarios, fotos familiares, documentos de propiedad. Incluso las facturas de servicios a su nombre ayudan.",
                "fr": "Ne vous inquiétez pas! Rassemblez ce que vous avez: lettre d'emploi, relevés bancaires, photos de famille, documents de propriété. Même les factures de services à votre nom aident."
            }
        },
        {
            "id": "sufficient_funds",
            "question": {
                "en": "Do you have enough money saved for your trip?",
                "es": "¿Tiene suficiente dinero ahorrado para su viaje?",
                "fr": "Avez-vous assez d'argent économisé pour votre voyage?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "You need to show you can pay for your trip.",
                "es": "Necesita demostrar que puede pagar su viaje.",
                "fr": "Vous devez montrer que vous pouvez payer votre voyage."
            },
            "help": {
                "en": "Budget roughly:\n• Flights: varies by location\n• Hotels: $100-200 CAD/night (or free if staying with family)\n• Food & activities: $50-100 CAD/day\n\nShow 3-6 months of bank statements with a steady balance.",
                "es": "Presupuesto aproximado:\n• Vuelos: varía según la ubicación\n• Hoteles: $100-200 CAD/noche (o gratis si se queda con familia)\n• Comida y actividades: $50-100 CAD/día\n\nMuestre 3-6 meses de estados de cuenta bancarios con un saldo estable.",
                "fr": "Budget approximatif:\n• Vols: varie selon l'emplacement\n• Hôtels: 100-200 CAD/nuit (ou gratuit si vous restez avec la famille)\n• Nourriture et activités: 50-100 CAD/jour\n\nMontrez 3-6 mois de relevés bancaires avec un solde stable."
            },
            "action_if_no": {
                "en": "Options:\n1. Have someone in Canada invite you and show THEIR finances\n2. Save for a few months and apply later\n3. Plan a shorter trip that fits your budget",
                "es": "Opciones:\n1. Que alguien en Canadá lo invite y muestre SUS finanzas\n2. Ahorre por unos meses y aplique después\n3. Planee un viaje más corto que se ajuste a su presupuesto",
                "fr": "Options:\n1. Demandez à quelqu'un au Canada de vous inviter et de montrer SES finances\n2. Économisez pendant quelques mois et postulez plus tard\n3. Planifiez un voyage plus court qui correspond à votre budget"
            }
        },
        {
            "id": "travel_history",
            "question": {
                "en": "Have you traveled to other countries before? (US, UK, Europe, Australia, etc.)",
                "es": "¿Ha viajado a otros países antes? (EE.UU., Reino Unido, Europa, Australia, etc.)",
                "fr": "Avez-vous voyagé dans d'autres pays avant? (États-Unis, Royaume-Uni, Europe, Australie, etc.)"
            },
            "type": "boolean",
            "required": True,
            "help": {
                "en": "If you've visited countries like the US, UK, or Europe and returned home on time, that's a big plus! It shows you follow visa rules.",
                "es": "Si ha visitado países como EE.UU., Reino Unido o Europa y regresó a casa a tiempo, ¡eso es un gran punto a favor! Demuestra que sigue las reglas de visa.",
                "fr": "Si vous avez visité des pays comme les États-Unis, le Royaume-Uni ou l'Europe et êtes rentré à temps, c'est un gros plus! Cela montre que vous respectez les règles de visa."
            },
            "positive_note": {
                "en": "✓ Great! Previous travel history helps your application.",
                "es": "✓ ¡Excelente! El historial de viajes previos ayuda a su solicitud.",
                "fr": "✓ Super! L'historique de voyage précédent aide votre demande."
            }
        },
        {
            "id": "previous_refusal",
            "question": {
                "en": "Have you ever been refused a visa to Canada, US, UK, or Australia?",
                "es": "¿Alguna vez le han rechazado una visa para Canadá, EE.UU., Reino Unido o Australia?",
                "fr": "Avez-vous déjà été refusé un visa pour le Canada, les États-Unis, le Royaume-Uni ou l'Australie?"
            },
            "type": "boolean",
            "required": True,
            "warning": {
                "en": "A past refusal doesn't mean you can't apply again - but you need to explain what's different now.",
                "es": "Un rechazo pasado no significa que no pueda aplicar de nuevo, pero necesita explicar qué es diferente ahora.",
                "fr": "Un refus passé ne signifie pas que vous ne pouvez pas postuler à nouveau - mais vous devez expliquer ce qui est différent maintenant."
            },
            "help": {
                "en": "If you were refused before, you'll need to:\n1. Explain why you were refused\n2. Show what has changed since then\n3. Provide stronger evidence this time",
                "es": "Si fue rechazado antes, necesitará:\n1. Explicar por qué fue rechazado\n2. Mostrar qué ha cambiado desde entonces\n3. Proporcionar evidencia más fuerte esta vez",
                "fr": "Si vous avez été refusé avant, vous devrez:\n1. Expliquer pourquoi vous avez été refusé\n2. Montrer ce qui a changé depuis\n3. Fournir des preuves plus solides cette fois"
            },
            "action_if_yes": {
                "en": "Get your old refusal letter and address each reason. Show what's changed - new job, more savings, stronger ties, etc.",
                "es": "Obtenga su carta de rechazo anterior y aborde cada razón. Muestre qué ha cambiado: nuevo trabajo, más ahorros, vínculos más fuertes, etc.",
                "fr": "Obtenez votre ancienne lettre de refus et adressez chaque raison. Montrez ce qui a changé - nouvel emploi, plus d'économies, liens plus forts, etc."
            }
        },
        {
            "id": "criminal_record",
            "question": {
                "en": "Do you have any criminal record? (including DUI)",
                "es": "¿Tiene algún antecedente penal? (incluyendo DUI)",
                "fr": "Avez-vous un casier judiciaire? (y compris conduite en état d'ivresse)"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "A criminal record can make you inadmissible to Canada.",
                "es": "Un antecedente penal puede hacerlo inadmisible a Canadá.",
                "fr": "Un casier judiciaire peut vous rendre inadmissible au Canada."
            },
            "help": {
                "en": "This includes ANY conviction - even old ones, even minor ones, even DUIs. Be honest - lying is worse than the conviction itself.",
                "es": "Esto incluye CUALQUIER condena, incluso las antiguas, incluso las menores, incluso los DUI. Sea honesto: mentir es peor que la condena misma.",
                "fr": "Cela inclut TOUTE condamnation - même anciennes, même mineures, même les conduites en état d'ivresse. Soyez honnête - mentir est pire que la condamnation elle-même."
            },
            "action_if_yes": {
                "en": "You may need:\n• Criminal Rehabilitation (if 5+ years since sentence completed)\n• Temporary Resident Permit\n\nConsider consulting an immigration lawyer.",
                "es": "Puede necesitar:\n• Rehabilitación Criminal (si han pasado 5+ años desde que completó la sentencia)\n• Permiso de Residente Temporal\n\nConsidere consultar a un abogado de inmigración.",
                "fr": "Vous pourriez avoir besoin de:\n• Réhabilitation criminelle (si 5+ ans depuis la fin de la peine)\n• Permis de résident temporaire\n\nConsidérez consulter un avocat en immigration."
            }
        },
        {
            "id": "overstay",
            "question": {
                "en": "Have you ever stayed longer than allowed on any visa, in any country?",
                "es": "¿Alguna vez se ha quedado más tiempo del permitido con alguna visa, en algún país?",
                "fr": "Avez-vous déjà séjourné plus longtemps que permis avec un visa, dans n'importe quel pays?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "Overstaying a visa is a serious concern for immigration officers.",
                "es": "Quedarse más tiempo del permitido con una visa es una preocupación seria para los oficiales de inmigración.",
                "fr": "Dépasser la durée d'un visa est une préoccupation sérieuse pour les agents d'immigration."
            },
            "help": {
                "en": "This means staying past your authorized date anywhere - not just Canada.",
                "es": "Esto significa quedarse más allá de su fecha autorizada en cualquier lugar, no solo en Canadá.",
                "fr": "Cela signifie rester au-delà de votre date autorisée n'importe où - pas seulement au Canada."
            },
            "action_if_yes": {
                "en": "Be honest about it. Explain the circumstances and show you've followed rules since then.",
                "es": "Sea honesto al respecto. Explique las circunstancias y demuestre que ha seguido las reglas desde entonces.",
                "fr": "Soyez honnête à ce sujet. Expliquez les circonstances et montrez que vous avez suivi les règles depuis."
            }
        }
    ],
    "work_permit": [
        {
            "id": "language",
            "question": {
                "en": "What language would you prefer?",
                "es": "¿Qué idioma prefiere?",
                "fr": "Quelle langue préférez-vous?"
            },
            "type": "select",
            "options": ["English", "Español (Spanish)", "Français (French)"],
            "required": True,
            "help": {
                "en": "Select your preferred language.",
                "es": "Seleccione su idioma preferido.",
                "fr": "Sélectionnez votre langue préférée."
            }
        },
        {
            "id": "country",
            "question": {
                "en": "What country are you from?",
                "es": "¿De qué país es usted?",
                "fr": "De quel pays venez-vous?"
            },
            "type": "text",
            "required": True,
            "help": {
                "en": "Enter your country of citizenship. This helps determine if you're eligible for special programs like IEC (Working Holiday) or CUSMA.",
                "es": "Ingrese su país de ciudadanía. Esto ayuda a determinar si es elegible para programas especiales como IEC (Working Holiday) o CUSMA.",
                "fr": "Entrez votre pays de citoyenneté. Cela aide à déterminer si vous êtes éligible à des programmes spéciaux comme EIC (Vacances-travail) ou ACEUM."
            }
        },
        {
            "id": "job_offer",
            "question": {
                "en": "Do you have a job offer from a Canadian employer?",
                "es": "¿Tiene una oferta de trabajo de un empleador canadiense?",
                "fr": "Avez-vous une offre d'emploi d'un employeur canadien?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "Most work permits require a job offer from a Canadian employer.",
                "es": "La mayoría de los permisos de trabajo requieren una oferta de trabajo de un empleador canadiense.",
                "fr": "La plupart des permis de travail nécessitent une offre d'emploi d'un employeur canadien."
            },
            "help": {
                "en": "Your job offer should include:\n• Company name and address\n• Your job title and duties\n• Salary and benefits\n• Start date\n• LMIA number (if applicable)",
                "es": "Su oferta de trabajo debe incluir:\n• Nombre y dirección de la empresa\n• Su título de trabajo y funciones\n• Salario y beneficios\n• Fecha de inicio\n• Número de LMIA (si aplica)",
                "fr": "Votre offre d'emploi doit inclure:\n• Nom et adresse de l'entreprise\n• Votre titre de poste et fonctions\n• Salaire et avantages\n• Date de début\n• Numéro EIMT (si applicable)"
            },
            "action_if_no": {
                "en": "Options:\n• International Experience Canada (IEC) - for youth from certain countries\n• Provincial Nominee Programs\n• Find an employer willing to hire foreign workers\n\n🔗 Check IEC eligibility: canada.ca/iec",
                "es": "Opciones:\n• Experiencia Internacional Canadá (IEC) - para jóvenes de ciertos países\n• Programas de Nominación Provincial\n• Encontrar un empleador dispuesto a contratar trabajadores extranjeros\n\n🔗 Verificar elegibilidad IEC: canada.ca/iec",
                "fr": "Options:\n• Expérience internationale Canada (EIC) - pour les jeunes de certains pays\n• Programmes des candidats des provinces\n• Trouver un employeur prêt à embaucher des travailleurs étrangers\n\n🔗 Vérifier l'éligibilité EIC: canada.ca/iec"
            }
        },
        {
            "id": "lmia_status",
            "question": {
                "en": "Has your employer gotten LMIA approval, or is your job LMIA-exempt?",
                "es": "¿Su empleador obtuvo aprobación de LMIA, o su trabajo está exento de LMIA?",
                "fr": "Votre employeur a-t-il obtenu l'approbation de l'EIMT, ou votre emploi est-il exempté d'EIMT?"
            },
            "type": "select",
            "options": {
                "en": ["Yes - LMIA approved", "Yes - LMIA-exempt (CUSMA, intra-company transfer, etc.)", "No / I don't know", "I'm applying through IEC (Working Holiday)"],
                "es": ["Sí - LMIA aprobado", "Sí - Exento de LMIA (CUSMA, transferencia intraempresa, etc.)", "No / No sé", "Estoy aplicando a través de IEC (Working Holiday)"],
                "fr": ["Oui - EIMT approuvé", "Oui - Exempté d'EIMT (ACEUM, transfert intra-entreprise, etc.)", "Non / Je ne sais pas", "Je postule via EIC (Vacances-travail)"]
            },
            "options_values": ["Yes - LMIA approved", "Yes - LMIA-exempt (CUSMA, intra-company transfer, etc.)", "No / I don't know", "I'm applying through IEC (Working Holiday)"],
            "required": True,
            "help": {
                "en": "LMIA = Labour Market Impact Assessment. It proves no Canadian was available for the job.\n\nLMIA-exempt jobs include:\n• CUSMA/USMCA professionals (US/Mexico citizens)\n• Intra-company transfers\n• International agreements\n• Significant benefit to Canada\n\nAsk your employer if you're not sure!",
                "es": "LMIA = Evaluación de Impacto en el Mercado Laboral. Demuestra que ningún canadiense estaba disponible para el trabajo.\n\nTrabajos exentos de LMIA incluyen:\n• Profesionales CUSMA/T-MEC (ciudadanos de EE.UU./México)\n• Transferencias intraempresa\n• Acuerdos internacionales\n• Beneficio significativo para Canadá\n\n¡Pregunte a su empleador si no está seguro!",
                "fr": "EIMT = Étude d'impact sur le marché du travail. Elle prouve qu'aucun Canadien n'était disponible pour le poste.\n\nEmplois exemptés d'EIMT incluent:\n• Professionnels ACEUM (citoyens américains/mexicains)\n• Transferts intra-entreprise\n• Accords internationaux\n• Avantage significatif pour le Canada\n\nDemandez à votre employeur si vous n'êtes pas sûr!"
            }
        },
        {
            "id": "valid_passport",
            "question": {
                "en": "Do you have a valid passport?",
                "es": "¿Tiene un pasaporte válido?",
                "fr": "Avez-vous un passeport valide?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "You need a valid passport to get a work permit.",
                "es": "Necesita un pasaporte válido para obtener un permiso de trabajo.",
                "fr": "Vous avez besoin d'un passeport valide pour obtenir un permis de travail."
            },
            "help": {
                "en": "Your passport should be valid for the length of your work permit.",
                "es": "Su pasaporte debe ser válido por la duración de su permiso de trabajo.",
                "fr": "Votre passeport doit être valide pour la durée de votre permis de travail."
            },
            "action_if_no": {
                "en": "Renew your passport before applying.",
                "es": "Renueve su pasaporte antes de aplicar.",
                "fr": "Renouvelez votre passeport avant de postuler."
            }
        },
        {
            "id": "qualifications",
            "question": {
                "en": "Do you have the education or experience needed for this job?",
                "es": "¿Tiene la educación o experiencia necesaria para este trabajo?",
                "fr": "Avez-vous l'éducation ou l'expérience nécessaire pour ce poste?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "You need to prove you're qualified for the job.",
                "es": "Necesita demostrar que está calificado para el trabajo.",
                "fr": "Vous devez prouver que vous êtes qualifié pour le poste."
            },
            "help": {
                "en": "Gather:\n• Degrees, diplomas, certificates\n• Professional licenses (if required)\n• Reference letters from past employers\n• Your resume\n\n🔗 Get credentials assessed: wes.org/ca",
                "es": "Reúna:\n• Títulos, diplomas, certificados\n• Licencias profesionales (si se requieren)\n• Cartas de referencia de empleadores anteriores\n• Su currículum\n\n🔗 Evalúe sus credenciales: wes.org/ca",
                "fr": "Rassemblez:\n• Diplômes, certificats\n• Licences professionnelles (si requises)\n• Lettres de référence d'anciens employeurs\n• Votre CV\n\n🔗 Faites évaluer vos diplômes: wes.org/ca"
            },
            "action_if_no": {
                "en": "Get your credentials assessed by WES (World Education Services) and collect reference letters from previous employers.",
                "es": "Haga evaluar sus credenciales por WES (World Education Services) y recopile cartas de referencia de empleadores anteriores.",
                "fr": "Faites évaluer vos diplômes par WES (World Education Services) et collectez des lettres de référence d'anciens employeurs."
            }
        },
        {
            "id": "criminal_record",
            "question": {
                "en": "Do you have any criminal record?",
                "es": "¿Tiene algún antecedente penal?",
                "fr": "Avez-vous un casier judiciaire?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "A criminal record can prevent you from getting a work permit.",
                "es": "Un antecedente penal puede impedirle obtener un permiso de trabajo.",
                "fr": "Un casier judiciaire peut vous empêcher d'obtenir un permis de travail."
            },
            "help": {
                "en": "All convictions must be declared, including DUIs.",
                "es": "Todas las condenas deben ser declaradas, incluyendo DUI.",
                "fr": "Toutes les condamnations doivent être déclarées, y compris les conduites en état d'ivresse."
            },
            "action_if_yes": {
                "en": "You may need Criminal Rehabilitation or a Temporary Resident Permit. Consult an immigration lawyer.",
                "es": "Puede necesitar Rehabilitación Criminal o un Permiso de Residente Temporal. Consulte a un abogado de inmigración.",
                "fr": "Vous pourriez avoir besoin d'une réhabilitation criminelle ou d'un permis de résident temporaire. Consultez un avocat en immigration."
            }
        },
        {
            "id": "medical_exam",
            "question": {
                "en": "Are you okay with doing a medical exam if needed?",
                "es": "¿Está dispuesto a hacer un examen médico si es necesario?",
                "fr": "Êtes-vous d'accord pour faire un examen médical si nécessaire?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "Medical exams are required for certain work permits.",
                "es": "Los exámenes médicos son requeridos para ciertos permisos de trabajo.",
                "fr": "Les examens médicaux sont requis pour certains permis de travail."
            },
            "help": {
                "en": "You'll need a medical exam if:\n• Working in healthcare, childcare, or education\n• Staying longer than 6 months\n• Coming from certain countries\n\nCost: $200-400 CAD\n\n🔗 Find a doctor: secure.cic.gc.ca/pp-md/pp-list.aspx",
                "es": "Necesitará un examen médico si:\n• Trabaja en salud, cuidado infantil o educación\n• Se queda más de 6 meses\n• Viene de ciertos países\n\nCosto: $200-400 CAD\n\n🔗 Encontrar un médico: secure.cic.gc.ca/pp-md/pp-list.aspx",
                "fr": "Vous aurez besoin d'un examen médical si:\n• Vous travaillez dans la santé, la garde d'enfants ou l'éducation\n• Vous restez plus de 6 mois\n• Vous venez de certains pays\n\nCoût: 200-400 CAD\n\n🔗 Trouver un médecin: secure.cic.gc.ca/pp-md/pp-list.aspx"
            }
        },
        {
            "id": "leave_canada",
            "question": {
                "en": "Will you leave Canada when your work permit ends?",
                "es": "¿Dejará Canadá cuando termine su permiso de trabajo?",
                "fr": "Quitterez-vous le Canada à la fin de votre permis de travail?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "You must show you'll leave when your permit expires.",
                "es": "Debe demostrar que se irá cuando expire su permiso.",
                "fr": "Vous devez montrer que vous partirez à l'expiration de votre permis."
            },
            "help": {
                "en": "Show ties to your home country - property, family, job prospects, etc.",
                "es": "Muestre vínculos con su país de origen: propiedad, familia, perspectivas de trabajo, etc.",
                "fr": "Montrez des liens avec votre pays d'origine: propriété, famille, perspectives d'emploi, etc."
            }
        }
    ],
    "super_visa": [
        {
            "id": "language",
            "question": {
                "en": "What language would you prefer?",
                "es": "¿Qué idioma prefiere?",
                "fr": "Quelle langue préférez-vous?"
            },
            "type": "select",
            "options": ["English", "Español (Spanish)", "Français (French)"],
            "required": True,
            "help": {
                "en": "Select your preferred language.",
                "es": "Seleccione su idioma preferido.",
                "fr": "Sélectionnez votre langue préférée."
            }
        },
        {
            "id": "country",
            "question": {
                "en": "What country is the parent/grandparent from?",
                "es": "¿De qué país es el padre/abuelo?",
                "fr": "De quel pays vient le parent/grand-parent?"
            },
            "type": "text",
            "required": True,
            "help": {
                "en": "Enter the country of citizenship of the person who wants to visit Canada.",
                "es": "Ingrese el país de ciudadanía de la persona que quiere visitar Canadá.",
                "fr": "Entrez le pays de citoyenneté de la personne qui veut visiter le Canada."
            }
        },
        {
            "id": "who_are_you",
            "question": {
                "en": "Who are you in this application?",
                "es": "¿Quién es usted en esta solicitud?",
                "fr": "Qui êtes-vous dans cette demande?"
            },
            "type": "select",
            "options": {
                "en": ["I am the parent/grandparent who wants to visit Canada", "I am the child/grandchild in Canada helping my parent apply"],
                "es": ["Soy el padre/abuelo que quiere visitar Canadá", "Soy el hijo/nieto en Canadá ayudando a mi padre a aplicar"],
                "fr": ["Je suis le parent/grand-parent qui veut visiter le Canada", "Je suis l'enfant/petit-enfant au Canada qui aide mon parent à postuler"]
            },
            "options_values": ["I am the parent/grandparent who wants to visit Canada", "I am the child/grandchild in Canada helping my parent apply"],
            "required": True,
            "help": {
                "en": "This helps us ask the right questions. Either way, we'll guide you through what's needed!",
                "es": "Esto nos ayuda a hacer las preguntas correctas. ¡De cualquier manera, lo guiaremos a través de lo que se necesita!",
                "fr": "Cela nous aide à poser les bonnes questions. Dans tous les cas, nous vous guiderons à travers ce qui est nécessaire!"
            }
        },
        {
            "id": "relationship",
            "question": {
                "en": "What is your relationship?",
                "es": "¿Cuál es su relación?",
                "fr": "Quelle est votre relation?"
            },
            "type": "select",
            "options": {
                "en": ["Parent visiting child in Canada", "Grandparent visiting grandchild in Canada", "Other relationship"],
                "es": ["Padre visitando hijo en Canadá", "Abuelo visitando nieto en Canadá", "Otra relación"],
                "fr": ["Parent visitant enfant au Canada", "Grand-parent visitant petit-enfant au Canada", "Autre relation"]
            },
            "options_values": ["Parent visiting child in Canada", "Grandparent visiting grandchild in Canada", "Other relationship"],
            "required": True,
            "fail_value": "Other relationship",
            "fail_reason": {
                "en": "Super Visa is only for parents and grandparents. For other relatives, apply for a regular Visitor Visa instead.",
                "es": "La Super Visa es solo para padres y abuelos. Para otros familiares, solicite una Visa de Visitante regular.",
                "fr": "Le Super Visa est uniquement pour les parents et grands-parents. Pour d'autres proches, demandez un visa de visiteur régulier."
            },
            "help": {
                "en": "Super Visa is specifically for parents and grandparents of Canadian citizens or permanent residents. It allows stays of up to 5 years at a time!",
                "es": "La Super Visa es específicamente para padres y abuelos de ciudadanos canadienses o residentes permanentes. ¡Permite estadías de hasta 5 años a la vez!",
                "fr": "Le Super Visa est spécifiquement pour les parents et grands-parents de citoyens canadiens ou résidents permanents. Il permet des séjours jusqu'à 5 ans à la fois!"
            }
        },
        {
            "id": "host_status",
            "question": {
                "en": "Is the person in Canada a Canadian citizen or permanent resident?",
                "es": "¿La persona en Canadá es ciudadano canadiense o residente permanente?",
                "fr": "La personne au Canada est-elle citoyenne canadienne ou résidente permanente?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "The child/grandchild in Canada must be a citizen or permanent resident.",
                "es": "El hijo/nieto en Canadá debe ser ciudadano o residente permanente.",
                "fr": "L'enfant/petit-enfant au Canada doit être citoyen ou résident permanent."
            },
            "help": {
                "en": "They'll need to show proof:\n• Canadian passport, OR\n• Citizenship certificate, OR\n• PR card (Permanent Resident card)",
                "es": "Necesitarán mostrar prueba:\n• Pasaporte canadiense, O\n• Certificado de ciudadanía, O\n• Tarjeta PR (tarjeta de Residente Permanente)",
                "fr": "Ils devront montrer une preuve:\n• Passeport canadien, OU\n• Certificat de citoyenneté, OU\n• Carte RP (carte de Résident Permanent)"
            },
            "action_if_no": {
                "en": "Wait until they get PR status, or apply for a regular Visitor Visa instead.",
                "es": "Espere hasta que obtengan el estatus de PR, o solicite una Visa de Visitante regular.",
                "fr": "Attendez qu'ils obtiennent le statut de RP, ou demandez un visa de visiteur régulier."
            }
        },
        {
            "id": "family_size",
            "question": {
                "en": "How many people live in the Canadian household?",
                "es": "¿Cuántas personas viven en el hogar canadiense?",
                "fr": "Combien de personnes vivent dans le foyer canadien?"
            },
            "type": "number",
            "min": 1,
            "max": 10,
            "required": True,
            "help": {
                "en": "Count everyone living in the home:\n• The child/grandchild\n• Their spouse/partner\n• Their children\n• Anyone else they support\n\nThis number determines the minimum income needed.",
                "es": "Cuente a todos los que viven en el hogar:\n• El hijo/nieto\n• Su cónyuge/pareja\n• Sus hijos\n• Cualquier otra persona que mantengan\n\nEste número determina el ingreso mínimo necesario.",
                "fr": "Comptez tous ceux qui vivent dans la maison:\n• L'enfant/petit-enfant\n• Leur conjoint/partenaire\n• Leurs enfants\n• Toute autre personne qu'ils soutiennent\n\nCe nombre détermine le revenu minimum nécessaire."
            }
        },
        {
            "id": "host_income",
            "question": {
                "en": "What is the Canadian household's yearly income (before taxes)?",
                "es": "¿Cuál es el ingreso anual del hogar canadiense (antes de impuestos)?",
                "fr": "Quel est le revenu annuel du foyer canadien (avant impôts)?"
            },
            "type": "number",
            "required": True,
            "help": {
                "en": "This is the GROSS income (before deductions). You can combine:\n• Employment income\n• Spouse's income\n• Self-employment income\n• Investment income\n\n📄 Get your Notice of Assessment from CRA:\n🔗 CRA My Account: canada.ca/my-cra-account\n📞 CRA Phone: 1-800-959-8281",
                "es": "Este es el ingreso BRUTO (antes de deducciones). Puede combinar:\n• Ingreso de empleo\n• Ingreso del cónyuge\n• Ingreso de trabajo independiente\n• Ingreso de inversiones\n\n📄 Obtenga su Aviso de Evaluación de CRA:\n🔗 CRA Mi Cuenta: canada.ca/my-cra-account\n📞 Teléfono CRA: 1-800-959-8281",
                "fr": "C'est le revenu BRUT (avant déductions). Vous pouvez combiner:\n• Revenu d'emploi\n• Revenu du conjoint\n• Revenu de travail indépendant\n• Revenu de placements\n\n📄 Obtenez votre Avis de cotisation de l'ARC:\n🔗 Mon dossier ARC: canada.ca/my-cra-account\n📞 Téléphone ARC: 1-800-959-8281"
            }
        },
        {
            "id": "medical_insurance",
            "question": {
                "en": "Will you buy Canadian medical insurance for at least 1 year?",
                "es": "¿Comprará seguro médico canadiense por al menos 1 año?",
                "fr": "Achèterez-vous une assurance médicale canadienne pour au moins 1 an?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "Super Visa requires medical insurance from a Canadian company.",
                "es": "La Super Visa requiere seguro médico de una compañía canadiense.",
                "fr": "Le Super Visa nécessite une assurance médicale d'une compagnie canadienne."
            },
            "help": {
                "en": "Requirements:\n✓ From a CANADIAN insurance company\n✓ At least $100,000 coverage\n✓ Valid for minimum 1 year\n✓ Covers healthcare, hospital, repatriation\n\nCost: Usually $1,000-3,000/year depending on age\n\nPopular providers:\n• Manulife\n• Blue Cross\n• TuGo\n• Allianz",
                "es": "Requisitos:\n✓ De una compañía de seguros CANADIENSE\n✓ Al menos $100,000 de cobertura\n✓ Válido por mínimo 1 año\n✓ Cubre atención médica, hospital, repatriación\n\nCosto: Usualmente $1,000-3,000/año dependiendo de la edad\n\nProveedores populares:\n• Manulife\n• Blue Cross\n• TuGo\n• Allianz",
                "fr": "Exigences:\n✓ D'une compagnie d'assurance CANADIENNE\n✓ Au moins 100 000 $ de couverture\n✓ Valide pour minimum 1 an\n✓ Couvre soins de santé, hôpital, rapatriement\n\nCoût: Habituellement 1 000-3 000 $/an selon l'âge\n\nFournisseurs populaires:\n• Manulife\n• Blue Cross\n• TuGo\n• Allianz"
            },
            "action_if_no": {
                "en": "You must buy this insurance - it's mandatory. Get quotes before applying so you know the cost.",
                "es": "Debe comprar este seguro - es obligatorio. Obtenga cotizaciones antes de aplicar para saber el costo.",
                "fr": "Vous devez acheter cette assurance - c'est obligatoire. Obtenez des devis avant de postuler pour connaître le coût."
            }
        },
        {
            "id": "valid_passport",
            "question": {
                "en": "Does the visitor have a valid passport?",
                "es": "¿El visitante tiene un pasaporte válido?",
                "fr": "Le visiteur a-t-il un passeport valide?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "A valid passport is required.",
                "es": "Se requiere un pasaporte válido.",
                "fr": "Un passeport valide est requis."
            },
            "help": {
                "en": "The passport should be valid for at least 2 years (Super Visa can last up to 10 years).",
                "es": "El pasaporte debe ser válido por al menos 2 años (la Super Visa puede durar hasta 10 años).",
                "fr": "Le passeport doit être valide pendant au moins 2 ans (le Super Visa peut durer jusqu'à 10 ans)."
            },
            "action_if_no": {
                "en": "Renew the passport before applying.",
                "es": "Renueve el pasaporte antes de aplicar.",
                "fr": "Renouvelez le passeport avant de postuler."
            }
        },
        {
            "id": "medical_exam",
            "question": {
                "en": "Is the visitor willing to do a medical exam?",
                "es": "¿El visitante está dispuesto a hacer un examen médico?",
                "fr": "Le visiteur est-il prêt à faire un examen médical?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "A medical exam is mandatory for Super Visa.",
                "es": "Un examen médico es obligatorio para la Super Visa.",
                "fr": "Un examen médical est obligatoire pour le Super Visa."
            },
            "help": {
                "en": "The exam must be done by an IRCC-approved doctor (called a 'panel physician').\n\nCost: $200-400 CAD\nResults go directly to IRCC.\n\n🔗 Find a panel physician: secure.cic.gc.ca/pp-md/pp-list.aspx",
                "es": "El examen debe ser realizado por un médico aprobado por IRCC (llamado 'médico de panel').\n\nCosto: $200-400 CAD\nLos resultados van directamente a IRCC.\n\n🔗 Encontrar un médico de panel: secure.cic.gc.ca/pp-md/pp-list.aspx",
                "fr": "L'examen doit être fait par un médecin approuvé par IRCC (appelé 'médecin désigné').\n\nCoût: 200-400 CAD\nLes résultats vont directement à IRCC.\n\n🔗 Trouver un médecin désigné: secure.cic.gc.ca/pp-md/pp-list.aspx"
            },
            "action_if_no": {
                "en": "This is mandatory - there's no way around it. The exam checks for conditions that could be a health risk or cost to Canada.",
                "es": "Esto es obligatorio - no hay forma de evitarlo. El examen verifica condiciones que podrían ser un riesgo de salud o costo para Canadá.",
                "fr": "C'est obligatoire - il n'y a pas moyen de l'éviter. L'examen vérifie les conditions qui pourraient être un risque pour la santé ou un coût pour le Canada."
            }
        },
        {
            "id": "criminal_record",
            "question": {
                "en": "Does the visitor have any criminal record?",
                "es": "¿El visitante tiene algún antecedente penal?",
                "fr": "Le visiteur a-t-il un casier judiciaire?"
            },
            "type": "boolean",
            "required": True,
            "fail_reason": {
                "en": "A criminal record can prevent entry to Canada.",
                "es": "Un antecedente penal puede impedir la entrada a Canadá.",
                "fr": "Un casier judiciaire peut empêcher l'entrée au Canada."
            },
            "help": {
                "en": "This includes any conviction, even old or minor ones.",
                "es": "Esto incluye cualquier condena, incluso las antiguas o menores.",
                "fr": "Cela inclut toute condamnation, même ancienne ou mineure."
            },
            "action_if_yes": {
                "en": "Options: Criminal Rehabilitation (if eligible) or Temporary Resident Permit. Consider consulting an immigration lawyer.",
                "es": "Opciones: Rehabilitación Criminal (si es elegible) o Permiso de Residente Temporal. Considere consultar a un abogado de inmigración.",
                "fr": "Options: Réhabilitation criminelle (si éligible) ou Permis de résident temporaire. Considérez consulter un avocat en immigration."
            }
        },
        {
            "id": "previous_refusal",
            "question": {
                "en": "Has the visitor ever been refused a Canadian visa?",
                "es": "¿Al visitante alguna vez le han rechazado una visa canadiense?",
                "fr": "Le visiteur s'est-il déjà vu refuser un visa canadien?"
            },
            "type": "boolean",
            "required": True,
            "warning": {
                "en": "A past refusal doesn't disqualify you, but you need to address it.",
                "es": "Un rechazo pasado no lo descalifica, pero necesita abordarlo.",
                "fr": "Un refus passé ne vous disqualifie pas, mais vous devez l'aborder."
            },
            "help": {
                "en": "If refused before, explain what has changed and provide stronger evidence.",
                "es": "Si fue rechazado antes, explique qué ha cambiado y proporcione evidencia más fuerte.",
                "fr": "Si refusé avant, expliquez ce qui a changé et fournissez des preuves plus solides."
            },
            "action_if_yes": {
                "en": "Get GCMS notes to see exact refusal reasons. Address each one in the new application.\n\n📞 IRCC Call Centre: 1-888-242-2100",
                "es": "Obtenga notas GCMS para ver las razones exactas del rechazo. Aborde cada una en la nueva solicitud.\n\n📞 Centro de llamadas IRCC: 1-888-242-2100",
                "fr": "Obtenez les notes SMGC pour voir les raisons exactes du refus. Adressez chacune dans la nouvelle demande.\n\n📞 Centre d'appels IRCC: 1-888-242-2100"
            }
        }
    ]
}


@app.get("/eligibility/questions/{application_type}")
async def get_eligibility_questions(application_type: str, lang: str = "en"):
    """Get the eligibility questions for a specific application type."""
    if application_type not in ELIGIBILITY_QUESTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown application type: {application_type}")
    
    # Map language selection to code
    lang_map = {
        "English": "en",
        "Español (Spanish)": "es", 
        "Français (French)": "fr",
        "en": "en",
        "es": "es",
        "fr": "fr"
    }
    lang_code = lang_map.get(lang, "en")
    
    # Translate questions
    questions = []
    for q in ELIGIBILITY_QUESTIONS[application_type]:
        translated_q = {"id": q["id"], "type": q["type"], "required": q.get("required", False)}
        
        # Translate question text
        if isinstance(q.get("question"), dict):
            translated_q["question"] = q["question"].get(lang_code, q["question"].get("en", ""))
        else:
            translated_q["question"] = q.get("question", "")
        
        # Translate help text
        if isinstance(q.get("help"), dict):
            translated_q["help"] = q["help"].get(lang_code, q["help"].get("en", ""))
        elif q.get("help"):
            translated_q["help"] = q["help"]
        
        # Handle options - translate display but keep values
        if "options" in q:
            if isinstance(q["options"], dict):
                translated_q["options"] = q["options"].get(lang_code, q["options"].get("en", []))
                translated_q["options_values"] = q.get("options_values", q["options"].get("en", []))
            else:
                translated_q["options"] = q["options"]
                translated_q["options_values"] = q.get("options_values", q["options"])
        
        # Copy other fields
        for field in ["min", "max", "fail_value"]:
            if field in q:
                translated_q[field] = q[field]
        
        # Translate fail_reason, warning, action_if_no, action_if_yes, positive_note
        for field in ["fail_reason", "warning", "action_if_no", "action_if_yes", "positive_note"]:
            if field in q:
                if isinstance(q[field], dict):
                    translated_q[field] = q[field].get(lang_code, q[field].get("en", ""))
                else:
                    translated_q[field] = q[field]
        
        questions.append(translated_q)
    
    return {
        "application_type": application_type,
        "questions": questions,
        "language": lang_code
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
    
    # Budget calculation for visitor visa
    budget_estimate = None
    if app_type == "visitor_visa":
        trip_days = answers.get("trip_duration", 0)
        accommodation = answers.get("accommodation_type", "")
        
        if trip_days > 0:
            # Estimate costs in CAD
            if "free" in accommodation.lower() or "family" in accommodation.lower():
                accommodation_cost = 0
                accommodation_note = "Staying with family/friends"
            else:
                accommodation_cost = trip_days * 150  # Average $150/night
                accommodation_note = f"{trip_days} nights × $150 CAD"
            
            daily_expenses = trip_days * 75  # Food, transport, activities
            flight_estimate = 1500  # Average round-trip
            visa_fee = 100  # Visitor visa fee
            
            total_estimate = accommodation_cost + daily_expenses + flight_estimate + visa_fee
            
            budget_estimate = {
                "trip_days": trip_days,
                "breakdown": [
                    {"item": "Accommodation", "amount": accommodation_cost, "note": accommodation_note},
                    {"item": "Daily expenses (food, transport, activities)", "amount": daily_expenses, "note": f"{trip_days} days × $75 CAD"},
                    {"item": "Round-trip flights (estimate)", "amount": flight_estimate, "note": "Varies by origin"},
                    {"item": "Visa application fee", "amount": visa_fee, "note": "Government fee"},
                ],
                "total": total_estimate,
                "currency": "CAD",
                "exchange_link": "https://www.xe.com/currencyconverter/convert/?From=CAD",
                "note": "This is an estimate. Actual costs vary based on your travel dates, origin, and spending habits."
            }
            
            warnings.append(
                f"💰 Estimated trip budget: ${total_estimate:,} CAD for {trip_days} days\n\n"
                f"You should show at least this amount in your bank statements.\n\n"
                f"🔗 Check exchange rates: xe.com/currencyconverter"
            )
    
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
    
    # Add form filling step for visitor visa and super visa
    if app_type in ["visitor_visa", "super_visa"]:
        form_name = "Visitor Visa" if app_type == "visitor_visa" else "Super Visa"
        action_plan.append({
            "priority": "medium",
            "action": f"📝 Use our Visa Forms tab to organize your {form_name} application information. The form wizard will help you prepare all the details you need before applying on the official IRCC website."
        })
    
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
        "income_requirement": get_lico_requirement(int(answers.get("family_size", 1))) if app_type == "super_visa" else None,
        "budget_estimate": budget_estimate
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


# ============== Visa Form Chat & PDF ==============

class VisaFormChatInput(BaseModel):
    message: str
    form_type: str
    current_data: Optional[dict] = None


class VisaFormPDFInput(BaseModel):
    form_type: str
    # Visitor Visa fields
    family_name: Optional[str] = None
    given_name: Optional[str] = None
    dob: Optional[str] = None
    country: Optional[str] = None
    passport_number: Optional[str] = None
    passport_expiry: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    purpose: Optional[str] = None
    trip_duration: Optional[int] = None
    arrival_date: Optional[str] = None
    departure_date: Optional[str] = None
    accommodation: Optional[str] = None
    canada_contact: Optional[str] = None
    occupation: Optional[str] = None
    employer: Optional[str] = None
    monthly_income: Optional[str] = None
    savings: Optional[str] = None
    ties: Optional[str] = None
    travel_history: Optional[str] = None
    # Super Visa fields
    visitor_family_name: Optional[str] = None
    visitor_given_name: Optional[str] = None
    visitor_dob: Optional[str] = None
    visitor_country: Optional[str] = None
    visitor_passport: Optional[str] = None
    visitor_passport_expiry: Optional[str] = None
    visitor_address: Optional[str] = None
    relationship: Optional[str] = None
    host_name: Optional[str] = None
    host_status: Optional[str] = None
    host_phone: Optional[str] = None
    host_email: Optional[str] = None
    host_address: Optional[str] = None
    family_size: Optional[int] = None
    host_income: Optional[int] = None
    insurance_provider: Optional[str] = None
    insurance_amount: Optional[str] = None
    notes: Optional[str] = None


def extract_visa_form_fields(message: str, form_type: str) -> dict:
    """Extract form fields from natural language for visa applications."""
    fields = {}
    msg_lower = message.lower()
    
    # Extract name
    name_patterns = [
        r'(?:name is|named|called|I am|I\'m)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:mother|father|parent|grandparent)(?:\s+is)?[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    ]
    
    prefix = 'visitor_' if form_type == 'super_visa' else ''
    
    for pattern in name_patterns:
        match = re.search(pattern, message)
        if match:
            full_name = match.group(1).strip()
            name_parts = full_name.split()
            if len(name_parts) >= 2:
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
    ]
    
    for pattern in dob_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            parsed = parse_date(match.group(1))
            if parsed:
                fields[f'{prefix}dob'] = parsed
                break
    
    # Extract country
    country_patterns = [
        r'(?:from|in|citizen of|citizenship)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    
    for pattern in country_patterns:
        match = re.search(pattern, message)
        if match:
            country = match.group(1).strip()
            if country.upper() not in [fields.get(f'{prefix}family_name', ''), fields.get(f'{prefix}given_name', '').upper()]:
                fields[f'{prefix}country'] = country
                break
    
    # Extract purpose (visitor visa)
    if form_type == 'visitor_visa':
        if any(word in msg_lower for word in ['visit', 'see', 'meet']):
            if any(word in msg_lower for word in ['sister', 'brother', 'family', 'friend', 'relative']):
                fields['purpose'] = 'family'
            elif any(word in msg_lower for word in ['business', 'meeting', 'conference']):
                fields['purpose'] = 'business'
            elif any(word in msg_lower for word in ['vacation', 'holiday', 'tourism', 'travel']):
                fields['purpose'] = 'tourism'
    
    # Extract duration
    duration_match = re.search(r'(\d+)\s*(?:days?|weeks?|months?)', message, re.IGNORECASE)
    if duration_match:
        num = int(duration_match.group(1))
        unit = duration_match.group(0).lower()
        if 'week' in unit:
            num *= 7
        elif 'month' in unit:
            num *= 30
        fields['trip_duration'] = num
    
    # Extract income (super visa)
    if form_type == 'super_visa':
        income_match = re.search(r'(?:income|earn|salary|make)[:\s]*\$?([\d,]+)', message, re.IGNORECASE)
        if income_match:
            income = int(income_match.group(1).replace(',', ''))
            fields['host_income'] = income
        
        family_match = re.search(r'(\d+)\s*(?:people|person|members?|of us)', message, re.IGNORECASE)
        if family_match:
            fields['family_size'] = int(family_match.group(1))
        
        # Host status
        if 'citizen' in msg_lower:
            fields['host_status'] = 'citizen'
        elif 'permanent resident' in msg_lower or ' pr ' in msg_lower:
            fields['host_status'] = 'pr'
    
    # Extract occupation
    occupation_match = re.search(r'(?:work as|job is|occupation is|I am a)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)', message, re.IGNORECASE)
    if occupation_match:
        fields['occupation'] = occupation_match.group(1)
    
    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message)
    if email_match:
        fields['email'] = email_match.group(0)
    
    # Extract phone
    phone_match = re.search(r'(?:phone|tel|cell|mobile)[:\s]*([\d\s\-\(\)\+]+)', message, re.IGNORECASE)
    if phone_match:
        parsed_phone = parse_phone(phone_match.group(1))
        if parsed_phone:
            fields['phone'] = parsed_phone
    
    return fields


@app.post("/chat-visa-form")
async def chat_visa_form(chat: VisaFormChatInput):
    """Chat assistant for visa form filling."""
    message = chat.message
    form_type = chat.form_type
    current_data = chat.current_data or {}
    
    extracted = extract_visa_form_fields(message, form_type)
    
    if extracted:
        response_parts = ["✓ I've extracted and formatted the following for IRCC:\n"]
        
        field_labels = {
            'family_name': 'Family Name',
            'given_name': 'Given Name(s)',
            'dob': 'Date of Birth',
            'country': 'Country of Citizenship',
            'purpose': 'Purpose of Visit',
            'trip_duration': 'Duration (days)',
            'occupation': 'Occupation',
            'email': 'Email',
            'phone': 'Phone',
            'visitor_family_name': 'Visitor Family Name',
            'visitor_given_name': 'Visitor Given Name(s)',
            'visitor_dob': 'Visitor Date of Birth',
            'visitor_country': 'Visitor Country',
            'host_status': 'Host Status',
            'host_income': 'Host Income (CAD)',
            'family_size': 'Family Size',
        }
        
        for field, value in extracted.items():
            label = field_labels.get(field, field.replace('_', ' ').title())
            response_parts.append(f"• {label}: {value}")
        
        response_parts.append("\nThe form has been updated. Continue with more information or switch to Form Wizard to review.")
        response = '\n'.join(response_parts)
    else:
        if form_type == 'visitor_visa':
            response = """I couldn't extract specific form fields from that. Try formats like:

• "My name is Maria Garcia, born January 5, 1990 in Mexico"
• "I want to visit my sister in Toronto for 2 weeks"
• "I work as a teacher and earn $3000/month"
• "I own a house and have 2 children in school"

What information would you like to add?"""
        else:
            response = """I couldn't extract specific form fields from that. Try formats like:

• "My mother is Rosa Martinez, born March 15, 1955 in Colombia"
• "I am a Canadian citizen living in Vancouver"
• "My household income is $85,000 per year"
• "We are 4 people in the household"

What information would you like to add?"""
    
    return {
        "response": response,
        "extracted_fields": extracted
    }


def generate_visitor_visa_pdf(data: dict) -> bytes:
    """Generate a summary PDF for Visitor Visa application."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20, textColor=colors.HexColor('#4F46E5'))
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor('#1F2937'))
    
    story = []
    
    story.append(Paragraph("🇨🇦 Visitor Visa Application Summary", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
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
    
    # Personal Information
    story.append(Paragraph("Personal Information", section_style))
    personal_rows = [
        ["Family Name:", data.get('family_name', '') or '—'],
        ["Given Name(s):", data.get('given_name', '') or '—'],
        ["Date of Birth:", data.get('dob', '') or '—'],
        ["Country of Citizenship:", data.get('country', '') or '—'],
        ["Passport Number:", data.get('passport_number', '') or '—'],
        ["Passport Expiry:", data.get('passport_expiry', '') or '—'],
        ["Email:", data.get('email', '') or '—'],
        ["Phone:", data.get('phone', '') or '—'],
        ["Current Address:", data.get('address', '') or '—'],
    ]
    story.append(make_table(personal_rows))
    story.append(Spacer(1, 15))
    
    # Travel Details
    story.append(Paragraph("Travel Details", section_style))
    purpose_map = {'tourism': 'Tourism/Vacation', 'family': 'Visiting family or friends', 'business': 'Business meetings', 'medical': 'Medical treatment', 'other': 'Other'}
    travel_rows = [
        ["Purpose of Visit:", purpose_map.get(data.get('purpose', ''), data.get('purpose', '')) or '—'],
        ["Duration of Stay:", f"{data.get('trip_duration', '')} days" if data.get('trip_duration') else '—'],
        ["Planned Arrival:", data.get('arrival_date', '') or '—'],
        ["Planned Departure:", data.get('departure_date', '') or '—'],
        ["Accommodation:", data.get('accommodation', '') or '—'],
        ["Contact in Canada:", data.get('canada_contact', '') or '—'],
    ]
    story.append(make_table(travel_rows))
    story.append(Spacer(1, 15))
    
    # Financial & Ties
    story.append(Paragraph("Financial Information & Ties to Home Country", section_style))
    financial_rows = [
        ["Occupation:", data.get('occupation', '') or '—'],
        ["Employer:", data.get('employer', '') or '—'],
        ["Monthly Income:", data.get('monthly_income', '') or '—'],
        ["Savings/Bank Balance:", data.get('savings', '') or '—'],
    ]
    story.append(make_table(financial_rows))
    
    if data.get('ties'):
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Ties to Home Country:</b>", styles['Normal']))
        story.append(Paragraph(data.get('ties', ''), styles['Normal']))
    
    if data.get('travel_history'):
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Previous Travel History:</b>", styles['Normal']))
        story.append(Paragraph(data.get('travel_history', ''), styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.gray)
    story.append(Paragraph(
        "<b>Note:</b> This is a summary document for your records. You must still complete the official IRCC application "
        "available at canada.ca/immigration. Use this summary to organize your information before applying online.",
        disclaimer_style
    ))
    
    doc.build(story)
    return buffer.getvalue()


def generate_super_visa_pdf(data: dict) -> bytes:
    """Generate a summary PDF for Super Visa application."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20, textColor=colors.HexColor('#4F46E5'))
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor('#1F2937'))
    
    story = []
    
    story.append(Paragraph("🇨🇦 Super Visa Application Summary", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
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
    
    # Visitor Information
    story.append(Paragraph("Visitor Information (Parent/Grandparent)", section_style))
    visitor_rows = [
        ["Family Name:", data.get('visitor_family_name', '') or '—'],
        ["Given Name(s):", data.get('visitor_given_name', '') or '—'],
        ["Date of Birth:", data.get('visitor_dob', '') or '—'],
        ["Country of Citizenship:", data.get('visitor_country', '') or '—'],
        ["Passport Number:", data.get('visitor_passport', '') or '—'],
        ["Passport Expiry:", data.get('visitor_passport_expiry', '') or '—'],
        ["Relationship to Host:", data.get('relationship', '').title() if data.get('relationship') else '—'],
        ["Current Address:", data.get('visitor_address', '') or '—'],
    ]
    story.append(make_table(visitor_rows))
    story.append(Spacer(1, 15))
    
    # Host Information
    story.append(Paragraph("Host Information (Child/Grandchild in Canada)", section_style))
    status_map = {'citizen': 'Canadian Citizen', 'pr': 'Permanent Resident'}
    host_rows = [
        ["Full Name:", data.get('host_name', '') or '—'],
        ["Immigration Status:", status_map.get(data.get('host_status', ''), data.get('host_status', '')) or '—'],
        ["Phone:", data.get('host_phone', '') or '—'],
        ["Email:", data.get('host_email', '') or '—'],
        ["Address in Canada:", data.get('host_address', '') or '—'],
    ]
    story.append(make_table(host_rows))
    story.append(Spacer(1, 15))
    
    # Income & Insurance
    story.append(Paragraph("Income & Insurance Requirements", section_style))
    family_size = data.get('family_size', 1) or 1
    host_income = data.get('host_income', 0) or 0
    required_income = get_lico_requirement(int(family_size))
    meets_requirement = host_income >= required_income
    
    income_rows = [
        ["Family Size:", str(family_size)],
        ["Required Income (LICO+30%):", f"${required_income:,} CAD"],
        ["Host's Annual Income:", f"${host_income:,} CAD"],
        ["Meets Requirement:", "✓ Yes" if meets_requirement else "✗ No - Shortfall: ${:,} CAD".format(required_income - host_income)],
        ["Insurance Provider:", data.get('insurance_provider', '') or '—'],
        ["Insurance Coverage:", data.get('insurance_amount', '') or '—'],
    ]
    story.append(make_table(income_rows))
    
    if data.get('notes'):
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Additional Notes:</b>", styles['Normal']))
        story.append(Paragraph(data.get('notes', ''), styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.gray)
    story.append(Paragraph(
        "<b>Note:</b> This is a summary document for your records. You must still complete the official IRCC Super Visa application "
        "available at canada.ca/immigration. The host must provide a signed Letter of Invitation, proof of income (NOA, T4), "
        "and the visitor must obtain medical insurance from a Canadian provider.",
        disclaimer_style
    ))
    
    doc.build(story)
    return buffer.getvalue()


@app.post("/generate-visa-pdf")
async def generate_visa_pdf(data: VisaFormPDFInput):
    """Generate a summary PDF for visa applications."""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not available. Install reportlab.")
    
    try:
        data_dict = data.dict()
        form_type = data_dict.get('form_type')
        
        if form_type == 'visitor_visa':
            pdf_bytes = generate_visitor_visa_pdf(data_dict)
        elif form_type == 'super_visa':
            pdf_bytes = generate_super_visa_pdf(data_dict)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown form type: {form_type}")
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={form_type}_summary.pdf",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ============== Proof of Relationship PDF ==============

class ProofEntry(BaseModel):
    id: int
    type: str
    date: str
    content: str
    description: Optional[str] = None


class ProofPDFInput(BaseModel):
    entries: list


def generate_proof_pdf(entries: list) -> bytes:
    """Generate a high-quality PDF of proof of relationship evidence."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        topMargin=0.5*inch, 
        bottomMargin=0.5*inch,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch
    )
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title', 
        parent=styles['Heading1'], 
        fontSize=18, 
        spaceAfter=10, 
        textColor=colors.HexColor('#1F2937'),
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#6B7280'),
        alignment=1,
        spaceAfter=20
    )
    section_style = ParagraphStyle(
        'Section', 
        parent=styles['Heading2'], 
        fontSize=11, 
        spaceBefore=8, 
        spaceAfter=4, 
        textColor=colors.HexColor('#374151'),
    )
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=4
    )
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#6B7280'),
        spaceAfter=3
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Proof of Relationship", title_style))
    story.append(Paragraph("Communication Evidence for Spousal Sponsorship Application", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')} | Total Entries: {len(entries)}", meta_style))
    
    # Type labels
    type_labels = {
        'text_message': 'Text Message',
        'email': 'Email',
        'social_media': 'Social Media',
        'letter': 'Letter',
        'call_log': 'Call Log',
        'photo': 'Photo',
        'other': 'Other'
    }
    
    # Sort entries by date
    sorted_entries = sorted(entries, key=lambda x: x.get('date', ''))
    
    for i, entry in enumerate(sorted_entries, 1):
        entry_type = entry.get('type', 'other')
        entry_date = entry.get('date', 'Unknown date')
        entry_content = entry.get('content', '')
        entry_desc = entry.get('description', '')
        entry_image = entry.get('image', '')
        
        type_label = type_labels.get(entry_type, 'Other')
        
        # Build entry as a group to keep together
        from reportlab.platypus import KeepTogether
        entry_elements = []
        
        # Entry header - compact
        entry_elements.append(Paragraph(f"<b>Entry {i}: {type_label}</b> — {entry_date}", section_style))
        
        if entry_desc:
            entry_elements.append(Paragraph(f"<i>{entry_desc}</i>", meta_style))
        
        # Add screenshot image if present - LARGER, fills page width
        if entry_image:
            try:
                import base64
                from PIL import Image as PILImage
                from reportlab.platypus import Image as RLImage
                
                if entry_image.startswith('data:'):
                    base64_data = entry_image.split(',')[1]
                else:
                    base64_data = entry_image
                
                image_bytes = base64.b64decode(base64_data)
                pil_img = PILImage.open(io.BytesIO(image_bytes))
                
                if pil_img.mode in ('RGBA', 'P'):
                    pil_img = pil_img.convert('RGB')
                
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='JPEG', quality=92)
                img_buffer.seek(0)
                
                # Fit 4 entries per page - small images
                img_width, img_height = pil_img.size
                max_width = 3 * inch
                max_height = 2 * inch
                scale = min(max_width / img_width, max_height / img_height)
                
                img_flowable = RLImage(img_buffer, width=img_width * scale, height=img_height * scale)
                entry_elements.append(img_flowable)
            except Exception as e:
                print(f"Image processing error: {e}")
                entry_elements.append(Paragraph("[Screenshot could not be processed]", meta_style))
        
        # Entry content
        if entry_content:
            content_text = entry_content.replace('\n', '<br/>')
            entry_elements.append(Paragraph(content_text, content_style))
        
        # Keep title + image together, allow 4 per page
        from reportlab.platypus import KeepTogether
        story.append(KeepTogether(entry_elements))
        
        # Thin separator
        if i < len(sorted_entries):
            story.append(Spacer(1, 5))
            story.append(Paragraph("─" * 80, ParagraphStyle('Line', textColor=colors.HexColor('#E5E7EB'), alignment=1, fontSize=6)))
            story.append(Spacer(1, 5))
    
    # Footer note
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.gray, alignment=1)
    story.append(Paragraph(
        "This document contains communication evidence submitted as part of a spousal sponsorship application "
        "to Immigration, Refugees and Citizenship Canada (IRCC). All content is authentic and represents "
        "genuine communication between the sponsor and applicant.",
        footer_style
    ))
    
    doc.build(story)
    return buffer.getvalue()


@app.post("/generate-proof-pdf")
async def generate_proof_pdf_endpoint(data: ProofPDFInput):
    """Generate a PDF of proof of relationship evidence."""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not available. Install reportlab.")
    
    try:
        entries = data.entries
        if not entries:
            raise HTTPException(status_code=400, detail="No entries provided")
        
        pdf_bytes = generate_proof_pdf(entries)
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=proof_of_relationship.pdf",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ============== Photo Album PDF Generation ==============

class PhotoAlbumInput(BaseModel):
    categories: list  # List of {category: str, photos: [{image, date, location, description}]}


def generate_photo_album_pdf(categories: list) -> bytes:
    """Generate a professional photo album PDF for spousal sponsorship."""
    from reportlab.lib.utils import ImageReader
    import base64
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title', 
        parent=styles['Heading1'], 
        fontSize=22, 
        spaceAfter=10, 
        textColor=colors.HexColor('#7C3AED'),
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#6B7280'),
        alignment=1,
        spaceAfter=30
    )
    category_style = ParagraphStyle(
        'Category', 
        parent=styles['Heading2'], 
        fontSize=14, 
        spaceBefore=20, 
        spaceAfter=10, 
        textColor=colors.HexColor('#7C3AED'),
        borderPadding=5
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#374151'),
        spaceAfter=5
    )
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#6B7280'),
        spaceAfter=3
    )
    
    story = []
    
    # Title page
    story.append(Paragraph("📷 Relationship Photo Album", title_style))
    story.append(Paragraph("Photographic Evidence for Spousal Sponsorship Application", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", meta_style))
    
    # Count total photos
    total_photos = sum(len(cat.get('photos', [])) for cat in categories)
    story.append(Paragraph(f"Total Photos: {total_photos}", meta_style))
    story.append(Spacer(1, 30))
    
    # Process each category
    for cat in categories:
        cat_title = cat.get('category', 'Photos')
        photos = cat.get('photos', [])
        
        if not photos:
            continue
        
        # Category header
        story.append(Paragraph(cat_title, category_style))
        
        for i, photo in enumerate(photos):
            image_data = photo.get('image', '')
            date = photo.get('date', '')
            location = photo.get('location', '')
            people = photo.get('people', '')
            description = photo.get('description', '')
            
            # Try to add the image
            if image_data:
                try:
                    # Handle base64 image data
                    if image_data.startswith('data:'):
                        # Extract base64 part after the comma
                        base64_data = image_data.split(',')[1]
                    else:
                        base64_data = image_data
                    
                    image_bytes = base64.b64decode(base64_data)
                    
                    # Use PIL to get dimensions and convert to a format reportlab handles well
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(io.BytesIO(image_bytes))
                    img_width, img_height = pil_img.size
                    
                    # Convert to RGB if necessary (handles PNG with transparency)
                    if pil_img.mode in ('RGBA', 'P'):
                        pil_img = pil_img.convert('RGB')
                    
                    # Save to new buffer as JPEG for reportlab
                    img_buffer = io.BytesIO()
                    pil_img.save(img_buffer, format='JPEG', quality=85)
                    img_buffer.seek(0)
                    
                    max_width = 5 * inch
                    max_height = 3.5 * inch
                    
                    # Calculate scale to fit within bounds
                    scale_w = max_width / img_width
                    scale_h = max_height / img_height
                    scale = min(scale_w, scale_h)
                    
                    final_width = img_width * scale
                    final_height = img_height * scale
                    
                    # Create image flowable
                    from reportlab.platypus import Image as RLImage
                    img_flowable = RLImage(img_buffer, width=final_width, height=final_height)
                    story.append(img_flowable)
                except Exception as e:
                    # If image fails, add placeholder text
                    print(f"Image processing error: {e}")
                    story.append(Paragraph(f"[Photo {i+1} - Image could not be processed]", caption_style))
            
            # Add photo metadata
            if date or location:
                meta_text = []
                if date:
                    # Format date nicely
                    try:
                        from datetime import datetime as dt
                        parsed_date = dt.strptime(date, '%Y-%m-%d')
                        formatted_date = parsed_date.strftime('%B %d, %Y')
                        meta_text.append(f"📅 {formatted_date}")
                    except:
                        meta_text.append(f"📅 {date}")
                if location:
                    meta_text.append(f"📍 {location}")
                story.append(Paragraph(" | ".join(meta_text), meta_style))
            
            # Add people in photo
            if people:
                story.append(Paragraph(f"👥 <b>People:</b> {people}", caption_style))
            
            # Add description
            if description:
                story.append(Paragraph(description, caption_style))
            
            story.append(Spacer(1, 15))
        
        # Add separator between categories
        story.append(Paragraph("─" * 70, ParagraphStyle('Line', textColor=colors.HexColor('#E5E7EB'), alignment=1)))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.gray, alignment=1)
    story.append(Paragraph(
        "This photo album is submitted as evidence of a genuine relationship for a spousal sponsorship application "
        "to Immigration, Refugees and Citizenship Canada (IRCC). All photographs are authentic and unaltered.",
        footer_style
    ))
    
    doc.build(story)
    return buffer.getvalue()


@app.post("/generate-photo-album-pdf")
async def generate_photo_album_pdf_endpoint(data: PhotoAlbumInput):
    """Generate a photo album PDF for spousal sponsorship."""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not available. Install reportlab.")
    
    try:
        categories = data.categories
        if not categories:
            raise HTTPException(status_code=400, detail="No photos provided")
        
        pdf_bytes = generate_photo_album_pdf(categories)
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=relationship_photo_album.pdf",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ============== Client Intake PDF Generation ==============

@app.post("/generate-intake-pdf")
async def generate_intake_pdf(data: dict):
    """Generate a client intake summary PDF."""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not available.")
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20, textColor=colors.HexColor('#1F2937'), alignment=1)
        section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=13, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#DC2626'))
        content_style = ParagraphStyle('Content', parent=styles['Normal'], fontSize=10, leading=14)
        
        story = []
        story.append(Paragraph("Client Intake Form / Formulario de Admision", title_style))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Helper
        def add_section(title, fields):
            story.append(Paragraph(title, section_style))
            rows = []
            for label, key in fields:
                value = data.get(key, '') or '—'
                rows.append([label + ":", str(value)])
            if rows:
                table = Table(rows, colWidths=[2.5*inch, 4*inch])
                table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                story.append(table)
            story.append(Spacer(1, 10))
        
        add_section("Personal Information / Informacion Personal", [
            ("Full Name", "full_name"),
            ("Date of Birth", "dob"),
            ("Country of Birth", "country_birth"),
            ("Citizenship", "citizenship"),
            ("Address", "address"),
            ("Phone", "phone"),
            ("Email", "email"),
            ("Marital Status", "marital_status"),
            ("Dependents", "dependents"),
        ])
        
        add_section("Immigration Status / Estado Migratorio", [
            ("Current Status", "current_status"),
            ("Permit Number", "permit_number"),
            ("Date Issued", "date_issued"),
            ("Expiry Date", "expiry_date"),
            ("Time in Canada", "time_in_canada"),
            ("Ever Refused", "ever_refused"),
            ("Refusal Details", "refusal_details"),
        ])
        
        # Services needed
        services = data.get('services_needed', [])
        if services:
            story.append(Paragraph("Services Needed / Servicios Necesarios", section_style))
            for s in services:
                story.append(Paragraph(f"• {s.replace('_', ' ').title()}", content_style))
            story.append(Spacer(1, 10))
        
        add_section("Education & Work / Educacion y Trabajo", [
            ("Education Level", "education_level"),
            ("Field of Study", "field_of_study"),
            ("Education Country", "edu_country"),
            ("ECA Assessed", "eca_assessed"),
            ("Occupation", "occupation"),
            ("NOC Code", "noc_code"),
            ("Years Exp (Canada)", "years_exp_canada"),
            ("Years Exp (Total)", "years_exp_total"),
            ("Job Offer", "has_job_offer"),
            ("LMIA Status", "lmia_status"),
        ])
        
        add_section("Language & Other / Idioma y Otros", [
            ("English Level", "english_level"),
            ("French Level", "french_level"),
            ("IELTS Score", "ielts_score"),
            ("TEF Score", "tef_score"),
            ("Family in Canada", "family_in_canada"),
            ("Family Details", "family_details"),
            ("Criminal History", "criminal_history"),
            ("Medical Issues", "medical_issues"),
            ("Previous Applications", "previous_apps"),
            ("Budget", "budget"),
            ("Notes", "notes"),
        ])
        
        # Footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray, alignment=1)
        story.append(Paragraph("This document is confidential and for immigration consultation purposes only.", footer_style))
        
        doc.build(story)
        
        name = data.get('full_name', 'client').replace(' ', '_')
        return Response(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=intake_{name}.pdf",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ============== Form Package Auto-Fill (IMM 1344, IMM 0008, IMM 0008 DEP, IMM 5532) ==============

class FormPackageInput(BaseModel):
    intake_data: dict
    children: Optional[list] = []
    sponsorship_data: Optional[dict] = {}


def generate_imm1344_pages(story, data, styles, section_style, make_table):
    """Generate IMM 1344 - Application to Sponsor pages."""
    story.append(Paragraph("IMM 1344 — Application to Sponsor, Sponsorship Agreement and Undertaking", section_style))
    story.append(Spacer(1, 10))
    
    # Map intake data to IMM 1344 fields
    full_name = data.get('full_name', '')
    name_parts = full_name.split() if full_name else []
    family_name = name_parts[-1].upper() if name_parts else ''
    given_names = ' '.join(name_parts[:-1]) if len(name_parts) > 1 else ''
    
    rows = [
        ["SECTION A — SPONSOR INFORMATION", ""],
        ["Family Name:", family_name],
        ["Given Name(s):", given_names],
        ["Date of Birth:", data.get('dob', '')],
        ["Sex:", data.get('gender', '')],
        ["Country of Birth:", data.get('country_birth', '')],
        ["Country of Citizenship:", data.get('citizenship', '')],
        ["Current Immigration Status:", data.get('current_status', '')],
        ["Marital Status:", data.get('marital_status', '').replace('_', ' ').title()],
        ["", ""],
        ["MAILING ADDRESS", ""],
        ["Address:", data.get('address', '')],
        ["Phone:", data.get('phone', '')],
        ["Email:", data.get('email', '')],
        ["", ""],
        ["SECTION B — PRINCIPAL APPLICANT (PERSON BEING SPONSORED)", ""],
        ["Family Name:", data.get('spouse_family_name', '') or data.get('applicant_family_name', '')],
        ["Given Name(s):", data.get('spouse_given_name', '') or data.get('applicant_given_name', '')],
        ["Date of Birth:", data.get('spouse_dob', '') or data.get('applicant_dob', '')],
        ["Country of Birth:", data.get('spouse_country_birth', '') or data.get('applicant_country_birth', '')],
        ["Country of Citizenship:", data.get('spouse_citizenship', '') or data.get('applicant_citizenship', '')],
        ["Relationship to Sponsor:", data.get('relationship_type', 'Spouse').replace('_', ' ').title()],
    ]
    story.append(make_table(rows))
    story.append(Spacer(1, 20))


def generate_imm0008_pages(story, data, styles, section_style, make_table):
    """Generate IMM 0008 - Generic Application Form for Canada pages."""
    story.append(Paragraph("IMM 0008 — Generic Application Form for Canada (Principal Applicant)", section_style))
    story.append(Spacer(1, 10))
    
    # The applicant is the person being sponsored (spouse/partner)
    rows = [
        ["SECTION A — PERSONAL DETAILS", ""],
        ["Family Name:", data.get('spouse_family_name', '') or data.get('applicant_family_name', '')],
        ["Given Name(s):", data.get('spouse_given_name', '') or data.get('applicant_given_name', '')],
        ["Date of Birth:", data.get('spouse_dob', '') or data.get('applicant_dob', '')],
        ["Sex:", data.get('spouse_sex', '') or data.get('applicant_sex', '')],
        ["Country of Birth:", data.get('spouse_country_birth', '') or data.get('applicant_country_birth', '')],
        ["Country of Citizenship:", data.get('spouse_citizenship', '') or data.get('applicant_citizenship', '')],
        ["Current Country of Residence:", data.get('spouse_country_residence', '') or data.get('applicant_country_residence', '')],
        ["Marital Status:", data.get('spouse_marital_status', 'Married').replace('_', ' ').title()],
        ["", ""],
        ["SECTION B — PASSPORT / TRAVEL DOCUMENT", ""],
        ["Passport Number:", data.get('spouse_passport', '') or data.get('applicant_passport', '') or data.get('permit_number', '')],
        ["Country of Issue:", data.get('spouse_passport_country', '') or data.get('applicant_passport_country', '')],
        ["Date of Issue:", data.get('spouse_passport_issued', '') or data.get('date_issued', '')],
        ["Expiry Date:", data.get('spouse_passport_expiry', '') or data.get('applicant_passport_expiry', '') or data.get('expiry_date', '')],
        ["", ""],
        ["SECTION C — CONTACT INFORMATION", ""],
        ["Current Address:", data.get('spouse_address', '') or data.get('applicant_address', '')],
        ["Phone:", data.get('spouse_phone', '') or data.get('applicant_phone', '')],
        ["Email:", data.get('spouse_email', '') or data.get('applicant_email', '')],
        ["", ""],
        ["SECTION D — EDUCATION", ""],
        ["Highest Level:", data.get('education_level', '').replace('_', ' ').title()],
        ["Field of Study:", data.get('field_of_study', '')],
        ["Country of Education:", data.get('edu_country', '')],
        ["", ""],
        ["SECTION E — LANGUAGE", ""],
        ["English Proficiency:", data.get('english_level', '').replace('_', ' ').title()],
        ["French Proficiency:", data.get('french_level', '').replace('_', ' ').title()],
        ["IELTS/CELPIP Score:", data.get('ielts_score', '')],
        ["TEF/TCF Score:", data.get('tef_score', '')],
    ]
    story.append(make_table(rows))
    story.append(Spacer(1, 20))


def generate_imm0008dep_pages(story, children, styles, section_style, make_table):
    """Generate IMM 0008 DEP - Additional Dependants/Declaration form for each child."""
    if not children:
        return
    
    story.append(Paragraph("IMM 0008 DEP — Additional Dependants / Declaration", section_style))
    story.append(Spacer(1, 10))
    
    for i, child in enumerate(children):
        story.append(Paragraph(f"Dependant {i + 1}", ParagraphStyle('ChildHeader', parent=styles['Heading3'], fontSize=12, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor('#374151'))))
        
        rows = [
            ["Family Name:", child.get('name', '').split()[-1].upper() if child.get('name', '').split() else ''],
            ["Given Name(s):", ' '.join(child.get('name', '').split()[:-1]) if len(child.get('name', '').split()) > 1 else child.get('name', '')],
            ["Date of Birth:", child.get('dob', '')],
            ["Sex:", child.get('gender', '').title()],
            ["Relationship to PA:", child.get('relationship', '').replace('_', ' ').title()],
            ["Country of Birth:", child.get('country_birth', '')],
            ["Country of Citizenship:", child.get('citizenship', '')],
            ["Passport Number:", child.get('passport_number', '')],
            ["Passport Expiry:", child.get('passport_expiry', '')],
            ["Marital Status:", child.get('marital_status', 'Single').replace('_', ' ').title()],
            ["Accompanying:", child.get('accompanying', 'yes').title()],
        ]
        story.append(make_table(rows))
        story.append(Spacer(1, 15))


def generate_imm5532_pages(story, data, styles, section_style, make_table):
    """Generate IMM 5532 - Relationship Information and Sponsorship Evaluation."""
    story.append(Paragraph("IMM 5532 — Relationship Information and Sponsorship Evaluation", section_style))
    story.append(Spacer(1, 10))
    
    # Sponsor info
    full_name = data.get('full_name', '')
    name_parts = full_name.split() if full_name else []
    
    rows = [
        ["SECTION A — YOU (THE SPONSOR)", ""],
        ["Family Name:", name_parts[-1].upper() if name_parts else ''],
        ["Given Name(s):", ' '.join(name_parts[:-1]) if len(name_parts) > 1 else ''],
        ["Date of Birth:", data.get('dob', '')],
        ["", ""],
        ["SECTION B — YOUR SPOUSE/PARTNER", ""],
        ["Family Name:", data.get('spouse_family_name', '') or data.get('applicant_family_name', '')],
        ["Given Name(s):", data.get('spouse_given_name', '') or data.get('applicant_given_name', '')],
        ["Date of Birth:", data.get('spouse_dob', '') or data.get('applicant_dob', '')],
        ["", ""],
        ["SECTION C — YOUR RELATIONSHIP", ""],
        ["How did you meet?:", data.get('how_met', '') or data.get('first_met_location', '')],
        ["Date you first met:", data.get('first_met_date', '')],
        ["Date relationship began:", data.get('relationship_start', '')],
        ["Date of marriage/union:", data.get('marriage_date', '') or data.get('date_married', '')],
        ["Place of marriage:", data.get('marriage_location', '') or data.get('place_married', '')],
        ["Currently living together?:", data.get('living_together', '')],
        ["", ""],
        ["SECTION D — DEVELOPMENT OF RELATIONSHIP", ""],
        ["Communication methods:", data.get('communication_methods', 'Phone, Video calls, Text messages')],
        ["Frequency of communication:", data.get('communication_frequency', '')],
        ["Number of visits:", data.get('number_of_visits', '')],
        ["", ""],
        ["SECTION E — FAMILY INFORMATION", ""],
        ["Number of dependents:", data.get('dependents', '0')],
        ["Family in Canada:", data.get('family_in_canada', '')],
        ["Family details:", data.get('family_details', '')],
    ]
    story.append(make_table(rows))
    story.append(Spacer(1, 20))


@app.post("/generate-form-package")
async def generate_form_package(input_data: FormPackageInput):
    """Generate a complete IRCC form package PDF with all forms auto-filled from intake data."""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not available. Install reportlab.")
    
    try:
        data = input_data.intake_data
        children = input_data.children or []
        sponsorship = input_data.sponsorship_data or {}
        
        # Merge sponsorship data into intake data (sponsorship fields take priority)
        merged = {**data, **sponsorship}
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, spaceAfter=20, textColor=colors.HexColor('#DC2626'))
        section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=14, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor('#1F2937'))
        
        story = []
        
        # Cover page
        story.append(Spacer(1, 100))
        story.append(Paragraph("🇨🇦 IRCC Form Package", title_style))
        story.append(Paragraph("Auto-filled from Client Intake Data", styles['Heading3']))
        story.append(Spacer(1, 30))
        
        applicant_name = merged.get('full_name', 'Unknown')
        story.append(Paragraph(f"<b>Sponsor:</b> {applicant_name}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        forms_included = ["IMM 1344 — Sponsorship Application", "IMM 0008 — Generic Application (Principal Applicant)"]
        if children:
            forms_included.append(f"IMM 0008 DEP — Dependants ({len(children)} child{'ren' if len(children) > 1 else ''})")
        forms_included.append("IMM 5532 — Relationship Information")
        
        story.append(Paragraph("<b>Forms Included:</b>", styles['Normal']))
        for form in forms_included:
            story.append(Paragraph(f"  • {form}", styles['Normal']))
        
        story.append(Spacer(1, 40))
        story.append(Paragraph("<i>Note: This is a pre-filled summary. Transfer this data to the official IRCC PDF forms available at canada.ca/immigration.</i>", 
                              ParagraphStyle('Note', parent=styles['Normal'], fontSize=9, textColor=colors.gray)))
        
        # Helper for tables
        def make_table(rows):
            table = Table(rows, colWidths=[2.5*inch, 4.5*inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ]))
            return table
        
        # Page break before each form
        from reportlab.platypus import PageBreak
        
        # IMM 1344
        story.append(PageBreak())
        generate_imm1344_pages(story, merged, styles, section_style, make_table)
        
        # IMM 0008
        story.append(PageBreak())
        generate_imm0008_pages(story, merged, styles, section_style, make_table)
        
        # IMM 0008 DEP (if children exist)
        if children:
            story.append(PageBreak())
            generate_imm0008dep_pages(story, children, styles, section_style, make_table)
        
        # IMM 5532
        story.append(PageBreak())
        generate_imm5532_pages(story, merged, styles, section_style, make_table)
        
        doc.build(story)
        
        name = merged.get('full_name', 'client').replace(' ', '_')
        return Response(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=IRCC_Form_Package_{name}.pdf",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Form package generation failed: {str(e)}")


# ============== Photo Album Word Document Generation ==============

@app.post("/generate-photo-album-word")
async def generate_photo_album_word(data: dict):
    """Generate a Word document for the photo album (editable)."""
    try:
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        doc = Document()
        
        # Title
        title = doc.add_heading('Relationship Photo Album', level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f'Generated: {datetime.now().strftime("%B %d, %Y")}')
        doc.add_paragraph('Photos showing the genuineness of our relationship, organized by category.')
        doc.add_paragraph('')
        
        categories = data.get('categories', [])
        
        for cat in categories:
            doc.add_heading(cat.get('category', ''), level=2)
            
            for i, photo in enumerate(cat.get('photos', [])):
                # Add photo placeholder
                p = doc.add_paragraph()
                p.add_run(f'Photo {i + 1}').bold = True
                
                # Add details table
                table = doc.add_table(rows=4, cols=2)
                table.style = 'Light Grid Accent 1'
                
                fields = [
                    ('Date:', photo.get('date', '')),
                    ('Location:', photo.get('location', '')),
                    ('People:', photo.get('people', '')),
                    ('Description:', photo.get('description', '')),
                ]
                
                for row_idx, (label, value) in enumerate(fields):
                    table.rows[row_idx].cells[0].text = label
                    table.rows[row_idx].cells[1].text = value or '—'
                
                doc.add_paragraph('')  # spacing
        
        # Tips section
        doc.add_heading('Tips for IRCC Submission', level=2)
        doc.add_paragraph('• Include dates on all photos')
        doc.add_paragraph('• Both partners should be clearly visible')
        doc.add_paragraph('• Show photos from different time periods')
        doc.add_paragraph('• Include photos with family members')
        doc.add_paragraph('• Print photos and write descriptions on the back')
        
        # Save to buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        return Response(
            content=buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": "attachment; filename=relationship_photo_album.docx",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="python-docx not installed. Add 'python-docx' to requirements.txt")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Word generation failed: {str(e)}")

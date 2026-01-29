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


SPONSORSHIP_FAQ = {
    "documents": """For a spousal sponsorship application, you'll need:

**Sponsor (Canadian citizen/PR):**
• Proof of status (citizenship certificate, PR card, or passport)
• Income documents (NOA, T4s, employment letter)
• Proof of relationship (photos, communication records, joint accounts)

**Applicant (spouse):**
• Valid passport
• Birth certificate
• Police certificates from all countries lived in 6+ months since age 18
• Medical exam results (from IRCC-approved doctor)
• Marriage certificate
• Photos together with dates

**Relationship proof:**
• Photos together (with dates/locations)
• Communication records (chat logs, call history)
• Travel records showing visits
• Joint financial documents if applicable
• Letters from family/friends confirming relationship""",

    "timeline": """Current processing times for spousal sponsorship:

**Inland applications (spouse in Canada):** 12-18 months
**Outland applications (spouse outside Canada):** 12-15 months

Key stages:
1. **Acknowledgment of Receipt (AOR):** 2-4 weeks after submission
2. **Eligibility decision:** 3-6 months
3. **Background checks:** Varies by country
4. **Medical request:** After eligibility approved
5. **Final decision:** After all checks complete

Note: Processing times vary and can change. Check IRCC website for current estimates.""",

    "eligibility": """**Sponsor eligibility requirements:**
• Must be a Canadian citizen or permanent resident
• Must be 18 years or older
• Must not be in prison, bankrupt, or under a removal order
• Must not have sponsored a spouse in the past 5 years (if that relationship ended)
• Must not have been sponsored as a spouse yourself in the past 5 years
• Must sign an undertaking to financially support your spouse for 3 years

**Applicant eligibility:**
• Must be legally married to the sponsor OR
• Must be in a common-law relationship (12+ months cohabitation) OR
• Must be in a conjugal relationship (if marriage/cohabitation not possible)
• Must pass medical and security checks
• Must not be inadmissible to Canada""",

    "work": """**Can the applicant work while waiting?**

**Inland applications (spouse in Canada):**
• Can apply for an Open Work Permit (OWP) at the same time as sponsorship
• OWP allows working for any employer in Canada
• Processing takes 2-4 months typically
• Valid until a decision is made on the PR application

**Outland applications (spouse outside Canada):**
• Cannot work in Canada until they receive their PR visa
• Must wait outside Canada during processing
• Can visit Canada as a visitor (if eligible) but cannot work

**After PR approval:**
• Full work authorization as a permanent resident
• No restrictions on employment""",

    "cost": """**Spousal sponsorship fees (2024):**

• Sponsorship fee: $75
• Principal applicant processing fee: $490
• Right of Permanent Residence Fee (RPRF): $515
• Biometrics: $85

**Total government fees:** ~$1,165 CAD

**Additional costs to budget for:**
• Medical exam: $200-400
• Police certificates: $50-100 per country
• Translation/notarization: Varies
• Photos: $15-30
• Courier/mailing: $50-100

**Optional:**
• Immigration lawyer/consultant: $2,000-5,000+"""
}


@app.post("/chat")
async def chat_assistant(chat: ChatInput):
    """Simple FAQ-based chat assistant for sponsorship questions."""
    message = chat.message.lower()
    
    # Match to FAQ topics
    if any(word in message for word in ['document', 'need', 'require', 'what do i need', 'checklist']):
        response = SPONSORSHIP_FAQ["documents"]
    elif any(word in message for word in ['how long', 'timeline', 'processing', 'time', 'wait']):
        response = SPONSORSHIP_FAQ["timeline"]
    elif any(word in message for word in ['eligib', 'qualify', 'requirement', 'can i sponsor', 'who can']):
        response = SPONSORSHIP_FAQ["eligibility"]
    elif any(word in message for word in ['work', 'job', 'employ', 'owp', 'work permit']):
        response = SPONSORSHIP_FAQ["work"]
    elif any(word in message for word in ['cost', 'fee', 'price', 'how much', 'pay']):
        response = SPONSORSHIP_FAQ["cost"]
    elif any(word in message for word in ['inland', 'outland', 'inside canada', 'outside canada']):
        response = """**Inland vs Outland Sponsorship:**

**Inland (spouse is IN Canada):**
• Spouse must have valid status in Canada
• Can apply for Open Work Permit
• Cannot leave Canada during processing (or application may be abandoned)
• Processing: 12-18 months

**Outland (spouse is OUTSIDE Canada):**
• Spouse waits in their home country
• Cannot work in Canada during processing
• Can visit Canada as a visitor
• Processing: 12-15 months
• Generally faster processing

Choose based on your situation and whether your spouse needs to work in Canada during processing."""
    elif any(word in message for word in ['common-law', 'common law', 'not married']):
        response = """**Common-law sponsorship requirements:**

To qualify as common-law partners, you must:
• Have lived together continuously for at least 12 months
• Be in a conjugal (marriage-like) relationship
• Provide proof of cohabitation (lease, bills, mail to same address)

**Evidence needed:**
• Joint lease or mortgage
• Joint bank accounts or bills
• Statutory declarations from both partners
• Letters from friends/family confirming cohabitation
• Photos together in shared home

If you cannot live together due to immigration barriers, you may qualify as "conjugal partners" instead."""
    elif any(word in message for word in ['reject', 'denied', 'refuse', 'appeal']):
        response = """**If your sponsorship is refused:**

1. **Review the refusal letter** - understand the specific reasons
2. **Options available:**
   • Request reconsideration (within 30 days)
   • Apply to Federal Court for judicial review (within 15-60 days)
   • Submit a new application addressing the concerns

**Common refusal reasons:**
• Insufficient proof of genuine relationship
• Sponsor doesn't meet income requirements
• Applicant is inadmissible (medical, criminal, misrepresentation)
• Missing documents

**Tip:** Consider consulting an immigration lawyer if refused."""
    else:
        response = """I can help you with questions about Canadian spousal sponsorship. Here are some topics I can assist with:

• **Documents required** - What you need to submit
• **Processing times** - How long it takes
• **Eligibility** - Who can sponsor and be sponsored
• **Work permits** - Can your spouse work while waiting?
• **Costs and fees** - Government and other expenses
• **Inland vs Outland** - Which option to choose
• **Common-law** - Requirements for unmarried couples

What would you like to know more about?"""
    
    return {"response": response}

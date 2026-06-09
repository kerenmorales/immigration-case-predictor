"""
Fraud Check routes — $4.99 AI second opinion on suspicious immigration documents.

Business model:
  - User uploads a document/screenshot of something suspicious
  - Pays $4.99 via Stripe
  - Claude Vision analyzes for known fraud patterns (NOT to accuse anyone)
  - Returns a confidence-scored "second opinion" with educational analysis
  - High-confidence fraud → upsell to $29 consultation with Keren

CRITICAL DESIGN PRINCIPLES:
  1. NEVER name specific people or businesses as "fraudsters"
  2. NEVER say "definitely safe" or "100% fraud"
  3. ALWAYS recommend professional verification
  4. Patterns described, not accusations made
  5. Confidence capped at 95% — there's always uncertainty

Endpoints:
  POST /fraud/create-checkout-session  - Start a $4.99 checkout
  POST /fraud/upload                    - Upload document + analyze
  GET  /fraud/checks/{user_id}          - List user's fraud checks
  GET  /fraud/check/{check_id}          - Get a specific check result
"""

import os
import io
import base64
import json
from datetime import datetime, timezone
from typing import Optional

import stripe
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/fraud", tags=["fraud"])

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FRAUD_PRICE_ID = os.environ.get("STRIPE_FRAUD_PRICE_ID", "")
FRAUD_SUCCESS_URL = os.environ.get(
    "STRIPE_FRAUD_SUCCESS_URL",
    "https://lovely-comfort-production-da4b.up.railway.app/?fraud_paid=true"
)
FRAUD_CANCEL_URL = os.environ.get(
    "STRIPE_FRAUD_CANCEL_URL",
    "https://lovely-comfort-production-da4b.up.railway.app/?fraud_canceled=true"
)

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "ImmigrationAI <onboarding@resend.dev>")
CALENDLY_LINK = os.environ.get("CALENDLY_LINK", "https://calendly.com/keren-morales/30min")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

ACCEPTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "application/pdf",
}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
STORAGE_BUCKET = "fraud-documents"

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError("Supabase env vars missing")
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


# ============================================================
# The Claude Vision system prompt — CAREFULLY ENGINEERED
# ============================================================

FRAUD_DETECTION_PROMPT = """You are a careful, conservative analyst helping Spanish-speaking immigrants in Canada get a SECOND OPINION on a document or message they suspect might be fraudulent. You are NOT a fraud authority. You DO NOT make accusations against specific people or businesses. You only describe patterns observable in the document.

CONTEXT: The user is checking whether a document they received (an "IRCC" letter, job offer, message from a "consultant", etc.) shows signs of being a scam targeting Spanish-speaking immigrants.

YOUR JOB: Analyze the image. Return a structured JSON response. Be CONSERVATIVE in your scoring — when in doubt, lower the score and recommend professional verification.

KNOWN FRAUD PATTERNS to look for (these are HIGH CONFIDENCE signals when present):

1. WRONG SENDER DOMAIN — Real IRCC emails come from @cic.gc.ca or @canada.ca. NOT @gmail.com, @yahoo.com, or any custom domain claiming to be IRCC.

2. WRONG WEBSITE URL — Real IRCC URLs are canada.ca only. Lookalike domains (canada-ircc.com, ircc-canada.org, gov-canada.ca) are FRAUD.

3. PAYMENT REQUESTS via WIRE/WESTERN UNION/BITCOIN/E-TRANSFER to a person — IRCC fees go through canada.ca portal only, never to "Officer Smith" via Western Union.

4. URGENT THREATS WITH SHORT DEADLINES — "Pay $X within 24 hours or face deportation." IRCC does NOT operate this way.

5. PERSONAL "OFFICER" CONTACT (WhatsApp, cell phone, gmail) — Real IRCC officers don't operate via WhatsApp.

6. GUARANTEED APPROVAL CLAIMS — "100% guaranteed visa approval." No legitimate lawyer or RCIC can guarantee outcomes.

7. UNUSUAL PERSONAL INFO REQUESTS — SIN, full credit card, mother's maiden name via email/text. IRCC doesn't collect these.

8. PRESSURE TO SKIP OFFICIAL CHANNELS — "Don't call IRCC, just send the money."

9. TYPOGRAPHIC RED FLAGS — Wrong logos, blurry seals, formatting errors, misspellings of official terms ("IRCCC", "Inmigration Canada", etc.).

10. CASH PAYMENT TO A PERSON — Real fees go to "Receiver General for Canada", not to an individual.

LEGITIMATE PATTERNS (lower fraud score when present):
- Sender uses @cic.gc.ca, @canada.ca, or other gc.ca subdomain
- References specific IRCC application numbers in proper format
- No payment requests, or payment via canada.ca portal only
- Reasonable timelines (weeks/months, not hours)
- No threatening language
- Clean, professional formatting matching IRCC templates

OUTPUT FORMAT — return ONLY this JSON, no other text:
{
  "confidence_score": <int 0-95>,
  "confidence_label": "<looks_legitimate | unclear | possible_fraud | likely_fraud>",
  "document_type": "<ircc_letter | email | whatsapp_screenshot | job_offer | text_message | other>",
  "patterns_detected": [
    {
      "pattern": "<short name in Spanish>",
      "severity": "<low | medium | high>",
      "explanation": "<one sentence in Spanish explaining what was observed and why it's concerning>"
    }
  ],
  "patterns_legitimate": [
    "<one sentence in Spanish describing a legitimate-looking aspect>"
  ],
  "could_not_verify": [
    "<one sentence in Spanish about something we couldn't determine, like 'whether the application number is real'>"
  ],
  "recommended_action": "<2-3 sentences in Spanish telling user what to do next, based on the score>",
  "educational_notes": "<3-4 sentences in Spanish explaining how legitimate IRCC documents look/work, relevant to this case>"
}

CONFIDENCE SCORING:
- 80-95: 4+ high-severity fraud patterns clearly present (wire transfer, threatening deadline, wrong domain, personal officer contact, etc.)
- 60-79: 2-3 high-severity OR many medium-severity patterns
- 40-59: Mixed signals, some concerning but inconclusive
- 20-39: Few markers, likely legitimate but verify
- 0-19: No fraud markers detected

CRITICAL:
- NEVER use specific person/business names from the document in patterns_detected. Say "El remitente usa..." not "Robert Smith usa..."
- NEVER score above 95
- ALWAYS include at least one entry in could_not_verify
- ALWAYS recommend verification with IRCC at 1-888-242-2100 or with a licensed lawyer/RCIC
- Use respectful Spanish ("usted")
- If image isn't an immigration document at all, set confidence_label to "looks_legitimate" with score 0 and explain in recommended_action that you couldn't analyze it"""


# ============================================================
# Pydantic models
# ============================================================

class CreateCheckoutRequest(BaseModel):
    user_id: str
    email: str


# ============================================================
# Endpoints
# ============================================================

@router.post("/create-checkout-session")
async def create_fraud_checkout(req: CreateCheckoutRequest):
    """Start a $4.99 Stripe Checkout for a Fraud Check. Creates a pending fraud_checks row."""
    if not stripe.api_key or not FRAUD_PRICE_ID:
        raise HTTPException(500, "Fraud check is not configured on this server")

    try:
        sb = get_supabase()

        check_resp = sb.table("fraud_checks").insert({
            "user_id": req.user_id,
            "is_paid": False,
            "status": "pending_payment",
        }).execute()
        check_id = check_resp.data[0]["id"]

        # Look up or create Stripe customer
        profile_resp = sb.table("user_profiles").select("stripe_customer_id").eq("id", req.user_id).execute()
        rows = profile_resp.data or []
        customer_id = rows[0].get("stripe_customer_id") if rows else None

        if not customer_id:
            customer = stripe.Customer.create(
                email=req.email,
                metadata={"supabase_user_id": req.user_id},
            )
            customer_id = customer.id
            sb.table("user_profiles").update({"stripe_customer_id": customer_id}).eq("id", req.user_id).execute()

        checkout = stripe.checkout.Session.create(
            customer=customer_id,
            mode="payment",
            line_items=[{"price": FRAUD_PRICE_ID, "quantity": 1}],
            success_url=FRAUD_SUCCESS_URL + f"&check_id={check_id}",
            cancel_url=FRAUD_CANCEL_URL,
            metadata={
                "supabase_user_id": req.user_id,
                "fraud_check_id": check_id,
                "product_type": "fraud_check",
            },
            payment_intent_data={
                "metadata": {
                    "supabase_user_id": req.user_id,
                    "fraud_check_id": check_id,
                    "product_type": "fraud_check",
                },
            },
        )

        sb.table("fraud_checks").update({
            "stripe_checkout_session_id": checkout.id,
        }).eq("id", check_id).execute()

        return {"url": checkout.url, "fraud_check_id": check_id}
    except stripe.error.StripeError as e:
        raise HTTPException(400, f"Stripe error: {e.user_message or str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Could not create fraud check checkout: {str(e)}")


@router.get("/checks/{user_id}")
async def list_user_fraud_checks(user_id: str):
    """List all fraud checks for a user. Verifies payment with Stripe as a backup."""
    sb = get_supabase()
    resp = sb.table("fraud_checks").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    checks = resp.data or []

    # Backup payment verification
    if stripe.api_key:
        for c in checks:
            if not c.get("is_paid") and c.get("stripe_checkout_session_id") and c.get("status") == "pending_payment":
                try:
                    checkout = stripe.checkout.Session.retrieve(c["stripe_checkout_session_id"])
                    if checkout.payment_status == "paid":
                        sb.table("fraud_checks").update({
                            "is_paid": True,
                            "paid_at": datetime.now(timezone.utc).isoformat(),
                        }).eq("id", c["id"]).execute()
                        c["is_paid"] = True
                except Exception as e:
                    print(f"Backup verification failed for fraud_check {c['id']}: {e}")

    return {"checks": checks}


@router.get("/check/{check_id}")
async def get_fraud_check(check_id: str, user_id: str):
    """Get a single fraud check (with ownership verification)."""
    sb = get_supabase()
    resp = sb.table("fraud_checks").select("*").eq("id", check_id).eq("user_id", user_id).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(404, "Fraud check not found")
    return rows[0]


@router.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    fraud_check_id: str = Form(...),
    user_id: str = Form(...),
    user_context: Optional[str] = Form(default=""),
    user_email: Optional[str] = Form(default=""),
):
    """
    Upload a document and run fraud analysis with Claude Vision.
    Requires the fraud_check to be paid.
    """
    sb = get_supabase()

    # Verify ownership and payment
    resp = sb.table("fraud_checks").select("*").eq("id", fraud_check_id).eq("user_id", user_id).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(404, "Fraud check not found")
    check = rows[0]

    # Backup verify with Stripe if needed
    if not check.get("is_paid") and check.get("stripe_checkout_session_id"):
        try:
            checkout = stripe.checkout.Session.retrieve(check["stripe_checkout_session_id"])
            if checkout.payment_status == "paid":
                sb.table("fraud_checks").update({
                    "is_paid": True,
                    "paid_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", fraud_check_id).execute()
                check["is_paid"] = True
        except Exception:
            pass

    if not check.get("is_paid"):
        raise HTTPException(402, "Fraud check not paid yet")

    if check.get("status") == "completed":
        raise HTTPException(400, "This fraud check has already been completed")

    # Validate file
    if file.content_type not in ACCEPTED_MIME_TYPES:
        raise HTTPException(400, f"File type not allowed: {file.content_type}. Use JPG, PNG, WEBP, or PDF.")

    content = await file.read()
    size = len(content)
    if size > MAX_FILE_BYTES:
        raise HTTPException(400, f"File too large. Max {MAX_FILE_BYTES // (1024*1024)}MB.")

    # Mark as analyzing
    sb.table("fraud_checks").update({
        "status": "analyzing",
        "document_filename": file.filename,
        "document_mime_type": file.content_type,
        "document_size_bytes": size,
        "user_context": user_context,
    }).eq("id", fraud_check_id).execute()

    # Upload to Supabase Storage
    storage_path = f"{user_id}/{fraud_check_id}/{file.filename}"
    try:
        sb.storage.from_(STORAGE_BUCKET).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type, "upsert": "true"},
        )
        sb.table("fraud_checks").update({"document_storage_path": storage_path}).eq("id", fraud_check_id).execute()
    except Exception as e:
        print(f"Storage upload failed (continuing with analysis): {e}")

    # Run Claude Vision analysis
    try:
        analysis = _analyze_with_claude(
            content=content,
            mime_type=file.content_type,
            user_context=user_context,
        )
    except Exception as e:
        sb.table("fraud_checks").update({"status": "failed"}).eq("id", fraud_check_id).execute()
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    # Save results
    sb.table("fraud_checks").update({
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "confidence_score": analysis.get("confidence_score", 0),
        "confidence_label": analysis.get("confidence_label", "unclear"),
        "patterns_detected": analysis.get("patterns_detected", []),
        "patterns_legitimate": analysis.get("patterns_legitimate", []),
        "document_type": analysis.get("document_type", "other"),
        "recommended_action": analysis.get("recommended_action", ""),
        "educational_notes": analysis.get("educational_notes", ""),
    }).eq("id", fraud_check_id).execute()

    # Email a copy to the user (best-effort, non-fatal)
    if user_email:
        try:
            _send_fraud_result_email(
                user_email=user_email,
                analysis=analysis,
                document_filename=file.filename,
            )
        except Exception as e:
            print(f"Fraud result email failed (non-fatal): {e}")

    # Re-fetch and return
    final = sb.table("fraud_checks").select("*").eq("id", fraud_check_id).execute()
    return final.data[0]


def _analyze_with_claude(content: bytes, mime_type: str, user_context: str) -> dict:
    """
    Send document to Claude Vision for fraud pattern analysis.
    Returns parsed JSON dict.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Encode image as base64
    b64 = base64.b64encode(content).decode("utf-8")

    user_message_parts = []

    # Add image if it's an image type
    if mime_type.startswith("image/"):
        user_message_parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": b64,
            },
        })
    elif mime_type == "application/pdf":
        # Claude supports PDFs as documents
        user_message_parts.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64,
            },
        })

    context_text = f"\n\nContexto del usuario: {user_context}" if user_context else ""
    user_message_parts.append({
        "type": "text",
        "text": f"Por favor analice este documento para señales de fraude.{context_text}\n\nResponda SOLO con el JSON especificado, sin texto antes o después.",
    })

    resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        system=FRAUD_DETECTION_PROMPT,
        messages=[{"role": "user", "content": user_message_parts}],
    )

    raw = resp.content[0].text if resp.content else "{}"
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    try:
        parsed = json.loads(raw)
    except Exception as e:
        print(f"Failed to parse Claude response as JSON: {e}\nRaw: {raw[:500]}")
        # Fallback: return a "couldn't analyze" result rather than failing
        return {
            "confidence_score": 0,
            "confidence_label": "unclear",
            "document_type": "other",
            "patterns_detected": [],
            "patterns_legitimate": [],
            "could_not_verify": ["No pudimos analizar este documento automáticamente."],
            "recommended_action": "Recomendamos consultar directamente con un abogado o RCIC autorizado, o llamar a IRCC al 1-888-242-2100 para verificar.",
            "educational_notes": "Cuando reciba documentos importantes de IRCC, siempre verifique enlaces, dominios de correo, y métodos de pago oficiales.",
        }

    # Safety: cap confidence at 95
    if isinstance(parsed.get("confidence_score"), (int, float)):
        parsed["confidence_score"] = min(95, max(0, int(parsed["confidence_score"])))
    else:
        parsed["confidence_score"] = 0

    return parsed


# ============================================================
# Email helper — sends a copy of the fraud check result to the user
# ============================================================

def _send_fraud_result_email(user_email: str, analysis: dict, document_filename: str):
    """
    Email a Spanish-language copy of the fraud analysis result to the user.
    Non-fatal: caller catches exceptions.
    """
    if not RESEND_API_KEY:
        print("RESEND_API_KEY not set — skipping fraud result email")
        return

    import resend
    resend.api_key = RESEND_API_KEY

    score = int(analysis.get("confidence_score") or 0)
    patterns = analysis.get("patterns_detected") or []
    legitimate = analysis.get("patterns_legitimate") or []
    could_not_verify = analysis.get("could_not_verify") or []
    recommended = analysis.get("recommended_action") or ""
    educational = analysis.get("educational_notes") or ""

    # Score band label + color
    if score >= 80:
        band = "Señales fuertes de fraude"
        band_color = "#dc2626"
        band_emoji = "🚨"
    elif score >= 60:
        band = "Varios patrones preocupantes"
        band_color = "#ea580c"
        band_emoji = "⚠️"
    elif score >= 40:
        band = "Señales mixtas — inconcluyente"
        band_color = "#d97706"
        band_emoji = "❓"
    elif score >= 20:
        band = "Pocos marcadores — probablemente legítimo, verifique"
        band_color = "#ca8a04"
        band_emoji = "🟡"
    else:
        band = "No se detectaron señales de fraude"
        band_color = "#16a34a"
        band_emoji = "✅"

    patterns_html = ""
    if patterns:
        items = []
        for p in patterns:
            sev = (p.get("severity") or "").lower()
            sev_label = {"high": "Alto", "medium": "Medio", "low": "Bajo"}.get(sev, sev)
            items.append(
                f"<li style='margin-bottom:10px;'><strong>[{sev_label}] {p.get('pattern','')}</strong>"
                f"<br><span style='color:#475569;font-size:14px;'>{p.get('explanation','')}</span></li>"
            )
        patterns_html = "<h3 style='color:#1e293b;'>🚩 Patrones detectados</h3><ul>" + "".join(items) + "</ul>"

    legitimate_html = ""
    if legitimate:
        items = "".join(f"<li style='color:#475569;'>{x}</li>" for x in legitimate)
        legitimate_html = f"<h3 style='color:#1e293b;'>✓ Aspectos que parecen legítimos</h3><ul>{items}</ul>"

    cnv_html = ""
    if could_not_verify:
        items = "".join(f"<li style='color:#475569;'>{x}</li>" for x in could_not_verify)
        cnv_html = f"<h3 style='color:#1e293b;'>❓ Lo que no pudimos verificar</h3><ul>{items}</ul>"

    cta_html = ""
    if score >= 50:
        cta_html = f"""
        <div style="background:#fef2f2;border:2px solid #fecaca;border-radius:8px;padding:20px;margin:24px 0;">
            <h3 style="color:#991b1b;margin-top:0;">🆘 Recomendamos consultar antes de tomar acción</h3>
            <p style="color:#7f1d1d;">
                Detectamos señales serias. <strong>NO envíe dinero. NO comparta más información personal.</strong>
                Una consulta de 30 minutos con Keren Morales le dará una respuesta definitiva.
            </p>
            <p>
                <a href="{CALENDLY_LINK}" style="background:#dc2626;color:white;padding:12px 22px;border-radius:6px;text-decoration:none;display:inline-block;font-weight:600;">
                    Reservar consulta de $29 →
                </a>
            </p>
        </div>
        """

    legend_html = """
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:20px 0;font-size:13px;color:#475569;">
        <strong>Cómo leer su puntaje (menor = mejor):</strong>
        <ul style="margin:8px 0 0 0;padding-left:18px;">
            <li><strong>0–19%</strong> · No se detectaron señales — el documento parece legítimo ✅</li>
            <li><strong>20–39%</strong> · Pocos marcadores — probablemente legítimo, pero verifique</li>
            <li><strong>40–59%</strong> · Señales mixtas — inconcluyente</li>
            <li><strong>60–79%</strong> · Varios patrones preocupantes</li>
            <li><strong>80–95%</strong> · Señales fuertes de fraude 🚨</li>
        </ul>
    </div>
    """

    html = f"""
    <div style="font-family:-apple-system,sans-serif;max-width:680px;margin:0 auto;color:#1e293b;">
        <h2 style="color:#b45309;">🔍 Su análisis de fraude — ImmigrationAI</h2>
        <p>Adjuntamos los resultados del análisis del documento <strong>{document_filename or 'que envió'}</strong>.</p>

        <div style="background:#fffbeb;border:2px solid {band_color};border-radius:10px;padding:20px;margin:20px 0;">
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <div>
                    <div style="font-size:24px;">{band_emoji} {band}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:36px;font-weight:bold;color:{band_color};">{score}%</div>
                    <div style="font-size:11px;color:#64748b;">confianza fraude</div>
                </div>
            </div>
            <p style="margin-top:14px;color:#334155;">{recommended}</p>
        </div>

        {legend_html}

        {cta_html}

        {patterns_html}

        {legitimate_html}

        {cnv_html}

        {f'<h3 style="color:#1e293b;">💡 Notas educativas</h3><p style="color:#475569;">{educational}</p>' if educational else ''}

        <h3 style="color:#1e293b;">Cómo verificar oficialmente</h3>
        <ul style="color:#475569;">
            <li>📞 IRCC: 1-888-242-2100</li>
            <li>🌐 Sitio oficial: <a href="https://www.canada.ca/en/services/immigration-citizenship.html">canada.ca/immigration</a></li>
            <li>⚖️ Verificar consultor: <a href="https://college-ic.ca/protecting-the-public/find-an-immigration-consultant">college-ic.ca</a></li>
        </ul>

        <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:14px;margin-top:28px;font-size:12px;color:#7f1d1d;">
            <strong>Aviso legal importante:</strong> Este análisis es una opinión automatizada generada por inteligencia artificial,
            basada en patrones comunes. <strong>No constituye asesoría legal vinculante.</strong> No reemplaza la consulta con
            un abogado autorizado o un consultor de inmigración (RCIC) registrado. Para certeza legal sobre su caso específico,
            consulte con un profesional licenciado o llame directamente a IRCC al 1-888-242-2100.
        </div>

        <p style="color:#94a3b8;font-size:11px;margin-top:24px;">
            ImmigrationAI · Esta herramienta provee información, no asesoría legal.
        </p>
    </div>
    """

    resend.Emails.send({
        "from": EMAIL_FROM,
        "to": [user_email],
        "subject": f"Su análisis de fraude — {score}% · ImmigrationAI",
        "html": html,
    })

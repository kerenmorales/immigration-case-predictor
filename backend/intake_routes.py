"""
AI Intake Interview routes — $29 conversational intake powered by Claude.

Flow:
  1. User pays $29 via Stripe → webhook marks intake_session.is_paid = true
  2. User starts intake → /intake/start returns initial AI greeting
  3. User chats with AI → /intake/message returns next question (Claude generates)
  4. AI gathers all required facts → /intake/transition-to-documents triggers doc upload
  5. User uploads files → /intake/upload (10 files, 50MB max)
  6. User finalizes → /intake/finalize generates dual summaries, sends emails

All Claude API calls require is_paid = true. Hard server-side enforcement.
"""

import os
import io
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import stripe
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, Header
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/intake", tags=["intake"])

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
LAWYER_EMAIL = os.environ.get("LAWYER_EMAIL", "keren.morales@gmail.com")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "ImmigrationAI <onboarding@resend.dev>")
CALENDLY_LINK = os.environ.get("CALENDLY_LINK", "https://calendly.com/keren-morales/30min")
INTAKE_PRICE_ID = os.environ.get("STRIPE_INTAKE_PRICE_ID", "")
INTAKE_SUCCESS_URL = os.environ.get("STRIPE_INTAKE_SUCCESS_URL", "https://lovely-comfort-production-da4b.up.railway.app/?intake_paid=true")
INTAKE_CANCEL_URL = os.environ.get("STRIPE_INTAKE_CANCEL_URL", "https://lovely-comfort-production-da4b.up.railway.app/?intake_canceled=true")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

MAX_FILES_PER_INTAKE = 10
MAX_BYTES_PER_INTAKE = 50 * 1024 * 1024  # 50 MB
ACCEPTED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/zip",
    "application/x-zip-compressed",
}
STORAGE_BUCKET = "intake-documents"

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError("Supabase env vars missing")
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


# ============================================================
# Claude system prompts — these define the conversational flow
# ============================================================

BASE_PROMPT_ES = """Eres un asistente de inmigración bilingüe (español primario), cálido como una "prima" que ya pasó por el proceso de inmigración canadiense. Estás haciendo una entrevista inicial para Keren Morales, abogada de inmigración, antes de su consulta de 30 minutos.

REGLAS CRÍTICAS:
1. Habla SIEMPRE en español castellano latino (México/Centroamérica/Sudamérica), no de España. Usa "usted" para mostrar respeto.
2. Haz UNA SOLA pregunta a la vez. Nunca múltiples preguntas en un mensaje.
3. Sé cálida, no robótica. Reconoce sus respuestas brevemente antes de la siguiente pregunta ("Entiendo, gracias", "Eso es importante saber").
4. Si detecta una BANDERA ROJA (rechazo previo, antecedentes penales, fraude, etc.), no se asuste — diga algo como: "Esta información es muy importante. No se preocupe, Keren la ayudará a ver opciones."
5. Cuando tenga suficiente información (después de 12-15 preguntas relevantes), termine con: "He recopilado lo necesario. Ahora vamos a subir documentos importantes." y devuelva en su respuesta el marcador especial: [INTAKE_READY_FOR_DOCS]
6. NO dé asesoría legal específica. Solo recopile información. La asesoría la dará Keren en la consulta.
7. Si el usuario menciona algo crítico (deportación, detención, fraude reciente), reconozca la urgencia.

OBJETIVO: Recopilar la información que Keren necesita para evaluar el caso y prepararse para la consulta de 30 minutos.

FORMATO DE RESPUESTA: Solo texto en español, sin formato markdown salvo casos necesarios. Mensajes cortos y conversacionales."""


SPONSORSHIP_QUESTIONS = """
PREGUNTAS REQUERIDAS PARA PATROCINIO CONYUGAL (orden flexible, adapte al flujo):

ESTADO CIVIL:
- ¿Está casado/a, en unión libre (common-law), o en relación conyugal (conjugal partner)?
- Si es common-law: ¿Cuánto tiempo han vivido juntos? (necesita 12+ meses)

LA RELACIÓN:
- ¿Cuándo y cómo se conocieron?
- ¿Hace cuánto tiempo se conocen?
- ¿Han vivido juntos? Si sí, ¿cuánto tiempo?
- ¿Cómo se comunican (presencial, llamadas, mensajes)?
- Si están separados: ¿Cuántas visitas presenciales han tenido?
- ¿Las familias de ambos saben de la relación?
- ¿Hay hijos en común o de relaciones previas?

EL MATRIMONIO/UNIÓN:
- ¿Dónde y cuándo se casaron (o comenzaron la unión libre)?
- ¿El matrimonio es legalmente reconocido en su país?
- ¿Tienen acta de matrimonio?
- ¿Es relación entre personas del mismo sexo? (importante para documentación)

EL PATROCINADOR:
- ¿El patrocinador es ciudadano canadiense o residente permanente?
- ¿Vive actualmente en Canadá?
- ¿Trabaja? ¿Cómo se mantiene?
- ¿Ha patrocinado a alguien antes? Si sí, ¿la persona recibió beneficios sociales? (BANDERA ROJA si hay default)

EL SOLICITANTE:
- ¿Dónde vive actualmente?
- ¿Ha estado en Canadá antes?
- ¿Tiene visa, permiso, o estatus actual en Canadá?
- ¿Está fuera de estatus en Canadá ahora? (BANDERA ROJA)

BANDERAS ROJAS A DETECTAR:
- ¿Han tenido algún rechazo previo de IRCC? (de cualquier visa)
- ¿Antecedentes penales en cualquier país? (incluso DUI)
- ¿Alguien les ofreció "garantizar" su aplicación? (víctima de notario)
- ¿Han pagado a alguien que no es abogado/RCIC autorizado?
- ¿La relación tiene menos de 1 año?

EVIDENCIA DISPONIBLE:
- ¿Qué evidencia tienen de su comunicación (WhatsApp, llamadas, fotos)?
- ¿Tienen finanzas conjuntas, cuentas, propiedad?
- ¿Cuántas fotos juntos tienen aproximadamente?
"""


WORK_PERMIT_QUESTIONS = """
PREGUNTAS REQUERIDAS PARA PERMISO DE TRABAJO (orden flexible):

UBICACIÓN:
- ¿Está dentro o fuera de Canadá ahora mismo?
- Si dentro: ¿Con qué estatus? (visitante, estudio, etc.) — BANDERA ROJA si fuera de estatus

OFERTA DE TRABAJO:
- ¿Tiene una oferta de trabajo de un empleador canadiense?
- Si sí: ¿Quién es el empleador? ¿Qué tipo de negocio?
- ¿Cuál es el puesto y salario?
- Si no: ¿Conoce IEC (Working Holiday)? ¿Es ciudadano de país elegible? (México, etc.)

LMIA / EXENCIÓN:
- ¿Sabe si su trabajo requiere LMIA (Labour Market Impact Assessment)?
- Si no sabe: detecte si es exento (CUSMA-USA/México, transferencia intra-empresa, etc.)
- ¿Ya tiene LMIA aprobado o el empleador lo está procesando?

PAÍS DE ORIGEN (importante para CUSMA, IEC):
- ¿De qué país es?

EDUCACIÓN Y EXPERIENCIA:
- ¿Cuál es su nivel de educación más alto?
- ¿En qué campo?
- ¿Cuántos años de experiencia tiene en el campo del trabajo ofrecido?

PASAPORTE:
- ¿Tiene pasaporte vigente por al menos 1 año?

FAMILIA:
- ¿Vendrá solo/a o con familia (esposo/a, hijos)?
- Si con familia, los detalles ayudan a Keren a planear permisos abiertos

HISTORIAL:
- ¿Ha estado en Canadá antes? ¿Con qué estatus?
- ¿Algún rechazo previo de visa o permiso? (Canadá u otro país) — BANDERA ROJA
- ¿Antecedentes penales? Incluyendo DUI — BANDERA ROJA
- ¿Problemas médicos significativos?

URGENCIA:
- ¿Cuándo necesita comenzar a trabajar?
"""


OPEN_ENDED_PROMPT = """
INTAKE ABIERTO — "Cuéntame tu situación":

Empiece con: "Cuénteme su situación con sus propias palabras. ¿Qué le trae a buscar ayuda con inmigración?"

Escuche atentamente. Su trabajo es:
1. Identificar qué tipo de aplicación necesita (sponsorship, work permit, study permit, visitor visa, PR, citizenship, refugee, etc.)
2. Una vez identificado, mencione brevemente: "Entiendo, parece que [tipo de caso]. Voy a hacerle algunas preguntas específicas."
3. Cambie al cuestionario apropiado de las plantillas de Sponsorship o Work Permit, O continúe con preguntas generales si es otro tipo.

Si parece complejo o multi-faceted, mencione que la consulta de 30 minutos será valiosa para discutir todas las opciones.
"""


def build_system_prompt(application_type: str, lang: str = "es") -> str:
    """Construct the full system prompt for Claude based on application type."""
    parts = [BASE_PROMPT_ES]

    if application_type == "spousal_sponsorship":
        parts.append("\n\n" + SPONSORSHIP_QUESTIONS)
    elif application_type == "work_permit":
        parts.append("\n\n" + WORK_PERMIT_QUESTIONS)
    elif application_type == "open_ended":
        parts.append("\n\n" + OPEN_ENDED_PROMPT)

    return "\n".join(parts)


# ============================================================
# Pydantic models
# ============================================================

class StartIntakeRequest(BaseModel):
    user_id: str
    application_type: str  # 'spousal_sponsorship' | 'work_permit' | 'open_ended'


class CreateCheckoutRequest(BaseModel):
    user_id: str
    email: str
    application_type: str


class IntakeMessageRequest(BaseModel):
    session_id: str
    user_id: str
    message: str


class FinalizeRequest(BaseModel):
    session_id: str
    user_id: str


# ============================================================
# Endpoints
# ============================================================

@router.post("/create-checkout-session")
async def create_intake_checkout(req: CreateCheckoutRequest):
    """Start a $29 Stripe Checkout for an AI Intake. Creates a pending intake_sessions row."""
    if not stripe.api_key or not INTAKE_PRICE_ID:
        raise HTTPException(500, "Intake checkout is not configured")

    try:
        sb = get_supabase()

        # Create the intake session record (pending payment)
        session_resp = sb.table("intake_sessions").insert({
            "user_id": req.user_id,
            "application_type": req.application_type,
            "is_paid": False,
            "status": "pending_payment",
        }).execute()
        intake_session_id = session_resp.data[0]["id"]

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
            line_items=[{"price": INTAKE_PRICE_ID, "quantity": 1}],
            success_url=INTAKE_SUCCESS_URL + f"&session_id={intake_session_id}",
            cancel_url=INTAKE_CANCEL_URL,
            metadata={
                "supabase_user_id": req.user_id,
                "intake_session_id": intake_session_id,
                "product_type": "ai_intake",  # disambiguates from $9.99 90-day product
            },
            payment_intent_data={
                "metadata": {
                    "supabase_user_id": req.user_id,
                    "intake_session_id": intake_session_id,
                    "product_type": "ai_intake",
                },
            },
        )

        # Save the Stripe checkout id for our records
        sb.table("intake_sessions").update({
            "stripe_checkout_session_id": checkout.id,
        }).eq("id", intake_session_id).execute()

        return {"url": checkout.url, "intake_session_id": intake_session_id}
    except stripe.error.StripeError as e:
        raise HTTPException(400, f"Stripe error: {e.user_message or str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Could not create intake checkout: {str(e)}")


@router.get("/sessions/{user_id}")
async def list_user_intakes(user_id: str):
    """List all intake sessions for a user (paid and pending). Verifies payment with Stripe as a backup."""
    sb = get_supabase()
    resp = sb.table("intake_sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    sessions = resp.data or []

    # Backup payment verification: for any unpaid session that has a Stripe checkout ID,
    # ask Stripe directly if it's been paid. This handles webhook delivery failures.
    if stripe.api_key:
        for s in sessions:
            if not s.get("is_paid") and s.get("stripe_checkout_session_id"):
                try:
                    checkout = stripe.checkout.Session.retrieve(s["stripe_checkout_session_id"])
                    if checkout.payment_status == "paid":
                        # Mark as paid in DB
                        sb.table("intake_sessions").update({
                            "is_paid": True,
                            "paid_at": datetime.now(timezone.utc).isoformat(),
                            "stripe_payment_intent_id": checkout.payment_intent,
                        }).eq("id", s["id"]).execute()
                        s["is_paid"] = True  # update in-memory copy too
                        print(f"Backup verification: marked intake {s['id']} as paid")
                except Exception as e:
                    print(f"Backup verification failed for {s['id']}: {e}")

    return {"sessions": sessions}


@router.post("/start")
async def start_intake(req: StartIntakeRequest):
    """
    Begin or resume the AI conversation. Requires the intake to be paid.
    Returns the initial greeting from Claude.
    """
    sb = get_supabase()

    # Find the most recent paid, in-progress intake for this user + application type
    resp = sb.table("intake_sessions").select("*").eq("user_id", req.user_id).eq("application_type", req.application_type).order("created_at", desc=True).limit(1).execute()
    rows = resp.data or []

    if not rows:
        raise HTTPException(404, "No intake session found. Please complete payment first.")

    session = rows[0]

    if not session.get("is_paid"):
        raise HTTPException(402, "Intake not paid yet. Please complete checkout.")

    if session.get("status") == "completed":
        return {
            "session_id": session["id"],
            "status": "completed",
            "history": session.get("conversation_history", []),
            "message": "Esta consulta ya fue completada.",
        }

    # If conversation is already started, return existing history
    history = session.get("conversation_history") or []
    if history:
        return {
            "session_id": session["id"],
            "status": session.get("status"),
            "history": history,
            "message": history[-1]["content"] if history else None,
        }

    # First message — generate greeting from Claude
    greeting = _generate_greeting(req.application_type)

    new_history = [{
        "role": "assistant",
        "content": greeting,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]

    sb.table("intake_sessions").update({
        "conversation_history": new_history,
        "status": "in_progress",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", session["id"]).execute()

    return {
        "session_id": session["id"],
        "status": "in_progress",
        "history": new_history,
        "message": greeting,
    }


def _generate_greeting(application_type: str) -> str:
    """Static initial greeting (no Claude call needed for this — saves $)."""
    greetings = {
        "spousal_sponsorship": (
            "Hola, soy su asistente de admisión para inmigración. Voy a hacerle algunas "
            "preguntas para preparar su consulta con Keren Morales. Sus respuestas son "
            "completamente confidenciales.\n\nComencemos: ¿Está casado/a o en unión libre "
            "(common-law)? Si es unión libre, ¿cuánto tiempo han vivido juntos?"
        ),
        "work_permit": (
            "Hola, soy su asistente de admisión para inmigración. Voy a hacerle algunas "
            "preguntas para preparar su consulta con Keren Morales sobre su permiso de "
            "trabajo. Sus respuestas son confidenciales.\n\nComencemos: ¿Está dentro o "
            "fuera de Canadá ahora mismo?"
        ),
        "open_ended": (
            "Hola, soy su asistente de admisión para inmigración. Quiero conocer su "
            "situación para preparar la mejor consulta posible con Keren Morales. Todo "
            "lo que comparta es confidencial.\n\nCuénteme con sus propias palabras: "
            "¿qué le trae a buscar ayuda con inmigración?"
        ),
    }
    return greetings.get(application_type, greetings["open_ended"])


@router.post("/message")
async def send_message(req: IntakeMessageRequest):
    """
    User sends a message. Claude responds with the next question.
    Saves the conversation to the database.
    """
    sb = get_supabase()

    # Verify session ownership and payment
    resp = sb.table("intake_sessions").select("*").eq("id", req.session_id).eq("user_id", req.user_id).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(404, "Session not found")

    session = rows[0]

    if not session.get("is_paid"):
        raise HTTPException(402, "Intake not paid yet")

    if session.get("status") in ("completed", "abandoned"):
        raise HTTPException(400, f"Cannot continue, status is {session['status']}")

    # Append user message
    history = session.get("conversation_history") or []
    history.append({
        "role": "user",
        "content": req.message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Call Claude
    try:
        ai_response, ready_for_docs = _call_claude(
            application_type=session["application_type"],
            history=history,
        )
    except Exception as e:
        raise HTTPException(500, f"AI call failed: {str(e)}")

    # Append AI response
    history.append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    new_status = "awaiting_documents" if ready_for_docs else "in_progress"

    sb.table("intake_sessions").update({
        "conversation_history": history,
        "status": new_status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", req.session_id).execute()

    return {
        "session_id": req.session_id,
        "message": ai_response,
        "status": new_status,
        "ready_for_documents": ready_for_docs,
    }


def _call_claude(application_type: str, history: List[Dict[str, str]]) -> tuple[str, bool]:
    """
    Call Anthropic Claude API. Returns (response_text, ready_for_docs).
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    # Lazy import — don't fail at startup if anthropic isn't installed
    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    system = build_system_prompt(application_type)

    # Anthropic expects only role/content for messages (no timestamps)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]

    resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=600,
        system=system,
        messages=messages,
    )

    text = resp.content[0].text if resp.content else ""

    # Detect the marker
    ready_for_docs = "[INTAKE_READY_FOR_DOCS]" in text
    text = text.replace("[INTAKE_READY_FOR_DOCS]", "").strip()

    return text, ready_for_docs


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: str = Form(...),
    document_category: str = Form(default="other"),
):
    """Upload a document to Supabase Storage. Validates limits."""
    sb = get_supabase()

    # Verify session
    resp = sb.table("intake_sessions").select("*").eq("id", session_id).eq("user_id", user_id).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(404, "Session not found")
    session = rows[0]
    if not session.get("is_paid"):
        raise HTTPException(402, "Intake not paid yet")

    # Validate file
    if file.content_type not in ACCEPTED_MIME_TYPES:
        raise HTTPException(400, f"File type not allowed: {file.content_type}")

    # Read content (stream into memory — capped by 50MB anyway)
    content = await file.read()
    size = len(content)

    # Enforce limits
    count_resp = sb.rpc("intake_file_count", {"session_id": session_id}).execute()
    current_count = count_resp.data if count_resp.data is not None else 0
    if current_count >= MAX_FILES_PER_INTAKE:
        raise HTTPException(400, f"Maximum {MAX_FILES_PER_INTAKE} files per intake")

    bytes_resp = sb.rpc("intake_total_bytes", {"session_id": session_id}).execute()
    current_bytes = bytes_resp.data if bytes_resp.data is not None else 0
    if current_bytes + size > MAX_BYTES_PER_INTAKE:
        raise HTTPException(400, f"Total upload would exceed {MAX_BYTES_PER_INTAKE // (1024*1024)}MB limit")

    # Upload to Supabase Storage
    storage_path = f"{user_id}/{session_id}/{file.filename}"
    try:
        sb.storage.from_(STORAGE_BUCKET).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type, "upsert": "true"},
        )
    except Exception as e:
        raise HTTPException(500, f"Storage upload failed: {str(e)}")

    # Record in DB
    doc_resp = sb.table("intake_documents").insert({
        "intake_session_id": session_id,
        "user_id": user_id,
        "filename": file.filename,
        "storage_path": storage_path,
        "mime_type": file.content_type,
        "size_bytes": size,
        "document_category": document_category,
    }).execute()

    return {"document_id": doc_resp.data[0]["id"], "filename": file.filename, "size_bytes": size}


@router.get("/documents/{session_id}")
async def list_documents(session_id: str, user_id: str):
    """List all documents uploaded for an intake session."""
    sb = get_supabase()

    # Verify ownership
    s_resp = sb.table("intake_sessions").select("id").eq("id", session_id).eq("user_id", user_id).execute()
    if not s_resp.data:
        raise HTTPException(404, "Session not found")

    docs = sb.table("intake_documents").select("*").eq("intake_session_id", session_id).execute()
    return {"documents": docs.data or []}


@router.post("/finalize")
async def finalize_intake(req: FinalizeRequest):
    """
    Generate the dual summaries (lawyer + user), send emails, return success.
    User can now book the Calendly consult.
    """
    sb = get_supabase()
    resp = sb.table("intake_sessions").select("*").eq("id", req.session_id).eq("user_id", req.user_id).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(404, "Session not found")
    session = rows[0]
    if not session.get("is_paid"):
        raise HTTPException(402, "Intake not paid yet")

    # Get user info
    user_resp = sb.table("user_profiles").select("email, full_name").eq("id", req.user_id).execute()
    user = (user_resp.data or [{}])[0]
    user_email = user.get("email", "unknown@example.com")

    # Get documents
    docs = sb.table("intake_documents").select("*").eq("intake_session_id", req.session_id).execute()
    documents = docs.data or []

    # Generate summaries via Claude
    lawyer_summary, user_summary_es, next_steps = _generate_summaries(
        application_type=session["application_type"],
        history=session.get("conversation_history") or [],
        documents=documents,
        user_email=user_email,
    )

    # Save summaries
    sb.table("intake_sessions").update({
        "lawyer_summary": lawyer_summary,
        "user_summary_es": user_summary_es,
        "next_steps": next_steps,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "calendly_link_used": CALENDLY_LINK,
    }).eq("id", req.session_id).execute()

    # Send emails
    try:
        _send_lawyer_email(
            user_email=user_email,
            application_type=session["application_type"],
            summary=lawyer_summary,
            documents=documents,
            session_id=req.session_id,
        )
    except Exception as e:
        print(f"Lawyer email failed: {e}")

    try:
        _send_user_email(
            user_email=user_email,
            application_type=session["application_type"],
            summary_es=user_summary_es,
            next_steps=next_steps,
        )
    except Exception as e:
        print(f"User email failed: {e}")

    return {
        "session_id": req.session_id,
        "lawyer_summary": lawyer_summary,
        "user_summary_es": user_summary_es,
        "next_steps": next_steps,
        "calendly_link": CALENDLY_LINK,
    }


def _generate_summaries(
    application_type: str,
    history: List[Dict[str, str]],
    documents: List[Dict[str, Any]],
    user_email: str,
) -> tuple[str, str, list]:
    """
    Use Claude to generate:
      1. lawyer_summary: detailed structured summary in English
      2. user_summary_es: warm summary in Spanish for the user
      3. next_steps: list of 5 personalized actionable steps
    """
    if not ANTHROPIC_API_KEY:
        # Fallback if Claude is unavailable
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        fallback_lawyer = (
            f"## INTAKE — {application_type.upper()}\n\n"
            f"**Client**: {user_email}\n\n"
            f"**Conversation transcript**:\n\n{history_text}\n\n"
            f"**Documents uploaded**: {len(documents)}"
        )
        return fallback_lawyer, "Resumen pendiente.", []

    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    history_text = "\n\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in history
    )
    docs_text = "\n".join(f"- {d['filename']} ({d.get('document_category', 'other')})" for d in documents)

    summary_prompt = f"""Analiza esta entrevista de admisión para inmigración canadiense ({application_type}).

CONVERSACIÓN COMPLETA:
{history_text}

DOCUMENTOS SUBIDOS:
{docs_text or "(ninguno)"}

Genera 3 cosas en JSON estricto:

1. "lawyer_summary": Un resumen profesional EN INGLÉS para la abogada Keren Morales. Incluye:
   - Client name (extract from conversation if mentioned, else "Pending")
   - Case type
   - Key facts as bullet points
   - 🚩 Red flags identified (with ⚠️ severity for each)
   - 🟢 Strengths of the case
   - Recommended next steps for the lawyer
   - Document review notes
   Use markdown headers and bullet points. ~300-500 words.

2. "user_summary_es": Un resumen cálido EN ESPAÑOL para el cliente. Que entienda su propio caso. Incluye:
   - Resumen breve y empático de su situación
   - Lo positivo de su caso (3-4 puntos)
   - Aspectos que necesitan más atención (sin asustar)
   - Mensaje tranquilizador final
   ~200-300 palabras.

3. "next_steps": Array de 5 pasos personalizados que el cliente debe hacer ANTES de la consulta. Cada paso es un objeto:
   {{"step": "Reúna su pasaporte y copias", "reason": "Lo necesitará para verificar fechas exactas"}}

DEVUELVE SOLO JSON VÁLIDO. Sin texto antes o después."""

    resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2500,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    raw = resp.content[0].text if resp.content else "{}"

    # Extract JSON (Claude sometimes wraps in markdown)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()

    try:
        parsed = json.loads(raw)
        return (
            parsed.get("lawyer_summary", ""),
            parsed.get("user_summary_es", ""),
            parsed.get("next_steps", []),
        )
    except Exception:
        return raw, "Resumen no pudo ser estructurado.", []


def _send_lawyer_email(user_email: str, application_type: str, summary: str, documents: list, session_id: str):
    """Send the structured intake summary to the lawyer."""
    if not RESEND_API_KEY:
        print("RESEND_API_KEY not set — skipping lawyer email")
        return

    import resend
    resend.api_key = RESEND_API_KEY

    docs_html = "".join(f"<li>{d['filename']} ({d.get('size_bytes', 0)//1024} KB)</li>" for d in documents)

    html = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 700px; margin: 0 auto;">
        <h2 style="color: #DC2626;">🇨🇦 New Immigration Intake</h2>
        <p><strong>From:</strong> {user_email}</p>
        <p><strong>Type:</strong> {application_type.replace('_', ' ').title()}</p>
        <p><strong>Submitted:</strong> {datetime.now(timezone.utc).strftime('%B %d, %Y at %I:%M %p UTC')}</p>
        <hr>
        <div style="white-space: pre-wrap;">{summary}</div>
        <hr>
        <h3>Documents Uploaded ({len(documents)})</h3>
        <ul>{docs_html or '<li>No documents</li>'}</ul>
        <p style="color: #6B7280; font-size: 12px;">Session ID: {session_id}</p>
    </div>
    """

    resend.Emails.send({
        "from": EMAIL_FROM,
        "to": [LAWYER_EMAIL],
        "subject": f"[ImmigrationAI] New {application_type.replace('_', ' ').title()} Intake — {user_email}",
        "html": html,
    })


def _send_user_email(user_email: str, application_type: str, summary_es: str, next_steps: list):
    """Send the warm Spanish summary to the user."""
    if not RESEND_API_KEY:
        print("RESEND_API_KEY not set — skipping user email")
        return

    import resend
    resend.api_key = RESEND_API_KEY

    steps_html = "".join(
        f"<li><strong>{s.get('step', '')}</strong><br><small>{s.get('reason', '')}</small></li>"
        for s in next_steps
    )

    html = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 700px; margin: 0 auto;">
        <h2 style="color: #DC2626;">¡Gracias por su consulta!</h2>
        <p>Hemos recibido toda su información. Aquí está su resumen personalizado:</p>
        <div style="background: #F9FAFB; padding: 20px; border-radius: 8px; white-space: pre-wrap; line-height: 1.6;">{summary_es}</div>
        <h3 style="margin-top: 30px;">Sus próximos 5 pasos antes de la consulta:</h3>
        <ol>{steps_html}</ol>
        <p style="margin-top: 30px;">
            <a href="{CALENDLY_LINK}" style="background: #DC2626; color: white; padding: 12px 24px; border-radius: 6px; text-decoration: none; display: inline-block;">Reservar mi consulta de 30 minutos</a>
        </p>
        <p style="color: #6B7280; font-size: 12px; margin-top: 40px;">
            ImmigrationAI — Esta herramienta provee información, no asesoría legal.
            Para asesoría personalizada, su consulta con Keren Morales será valiosa.
        </p>
    </div>
    """

    resend.Emails.send({
        "from": EMAIL_FROM,
        "to": [user_email],
        "subject": "Su resumen de consulta — ImmigrationAI",
        "html": html,
    })

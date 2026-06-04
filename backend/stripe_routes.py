"""
Stripe payment routes — One-time payment for 90 days of access.

Business model:
  - User pays $9.99 once
  - Gets 90 days of access from the date of payment
  - After 90 days, access expires and they must purchase again

Endpoints:
  POST /stripe/create-checkout-session  - Start a checkout (one-time payment)
  POST /stripe/webhook                  - Receive payment events from Stripe
  GET  /stripe/subscription/{user_id}   - Check access status

Required env vars:
  STRIPE_SECRET_KEY
  STRIPE_WEBHOOK_SECRET
  STRIPE_PRICE_ID            (a one-time/one-off price, NOT recurring)
  STRIPE_SUCCESS_URL
  STRIPE_CANCEL_URL
  SUPABASE_URL
  SUPABASE_SERVICE_KEY  (NOT the anon key)
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import stripe
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel
from supabase import create_client, Client

router = APIRouter(prefix="/stripe", tags=["stripe"])

# Configuration
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
SUCCESS_URL = os.environ.get("STRIPE_SUCCESS_URL", "https://your-domain.com/?subscribed=true")
CANCEL_URL = os.environ.get("STRIPE_CANCEL_URL", "https://your-domain.com/?canceled=true")

# Access duration after a successful payment
ACCESS_DAYS = int(os.environ.get("ACCESS_DAYS", "90"))

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


# Request models
class CheckoutRequest(BaseModel):
    user_id: str
    email: str


@router.post("/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest):
    """
    Start a Stripe Checkout session for a one-time $9.99 payment.
    On successful payment, the webhook grants the user 90 days of access.
    """
    if not stripe.api_key or not PRICE_ID:
        raise HTTPException(500, "Stripe is not configured on this server")

    try:
        sb = get_supabase()
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

        # Mode is 'payment' for one-time, NOT 'subscription'
        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="payment",
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            success_url=SUCCESS_URL,
            cancel_url=CANCEL_URL,
            allow_promotion_codes=True,
            metadata={"supabase_user_id": req.user_id},
            payment_intent_data={
                "metadata": {"supabase_user_id": req.user_id},
            },
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        raise HTTPException(400, f"Stripe error: {e.user_message or str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Failed to create checkout: {str(e)}")


@router.get("/subscription/{user_id}")
async def get_subscription(user_id: str):
    """
    Return the current access state for a user.
    Maps to: 'free' (never paid or expired) or 'active' (within 90-day window).
    """
    try:
        sb = get_supabase()
        resp = sb.table("user_profiles").select(
            "subscription_status, subscription_tier, subscription_current_period_end"
        ).eq("id", user_id).execute()
        rows = resp.data or []
        if not rows:
            return {"status": "free"}

        row = rows[0]
        status = row.get("subscription_status", "free")
        period_end = row.get("subscription_current_period_end")

        # Auto-expire: if period_end is in the past, status becomes 'expired'
        if period_end:
            end_dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))
            if end_dt < datetime.now(timezone.utc) and status == "active":
                # Lazy update: mark as expired in DB
                sb.table("user_profiles").update({
                    "subscription_status": "expired"
                }).eq("id", user_id).execute()
                status = "expired"

        return {
            "status": status,
            "tier": row.get("subscription_tier"),
            "current_period_end": period_end,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load subscription: {str(e)}")


@router.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    """
    Handle Stripe webhooks for one-time payments.

    The only event we care about is `checkout.session.completed` — when a user
    successfully completes payment, we grant them ACCESS_DAYS days of access.
    """
    if not WEBHOOK_SECRET:
        raise HTTPException(500, "Webhook secret not configured")

    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(400, "Invalid webhook signature")
    except Exception as e:
        raise HTTPException(400, f"Webhook error: {str(e)}")

    sb = get_supabase()
    event_type = event["type"]
    obj = event["data"]["object"]

    try:
        if event_type == "checkout.session.completed":
            # Confirm payment was actually successful
            payment_status = obj.get("payment_status")
            if payment_status != "paid":
                print(f"Checkout session completed but payment_status={payment_status} — ignoring")
                return {"received": True}

            user_id = obj.get("metadata", {}).get("supabase_user_id")
            if user_id:
                # Grant access for ACCESS_DAYS days from now
                # If they already have active access, EXTEND it from the existing end date
                # (this prevents accidental "lost days" if they pay early)
                resp = sb.table("user_profiles").select(
                    "subscription_status, subscription_current_period_end"
                ).eq("id", user_id).execute()
                rows = resp.data or []
                current = rows[0] if rows else {}

                now = datetime.now(timezone.utc)
                base_dt = now

                if current.get("subscription_status") == "active":
                    existing_end = current.get("subscription_current_period_end")
                    if existing_end:
                        try:
                            existing_dt = datetime.fromisoformat(existing_end.replace("Z", "+00:00"))
                            if existing_dt > now:
                                base_dt = existing_dt  # Stack on top
                        except Exception:
                            pass

                new_end = base_dt + timedelta(days=ACCESS_DAYS)

                sb.table("user_profiles").update({
                    "subscription_status": "active",
                    "subscription_tier": "premium",
                    "subscription_current_period_end": new_end.isoformat(),
                }).eq("id", user_id).execute()

                print(f"Granted {ACCESS_DAYS} days of access to user {user_id} until {new_end.isoformat()}")

    except Exception as e:
        # Don't 500 — Stripe will retry forever. Log and ack.
        print(f"Webhook handler error for {event_type}: {e}")

    return {"received": True}

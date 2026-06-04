"""
Stripe subscription routes.

Endpoints:
  POST /stripe/create-checkout-session  - Start a checkout
  POST /stripe/create-portal-session    - Manage subscription (cancel, update card)
  POST /stripe/webhook                  - Receive subscription state changes from Stripe
  GET  /stripe/subscription/{user_id}   - Check subscription status

Required env vars:
  STRIPE_SECRET_KEY
  STRIPE_WEBHOOK_SECRET
  STRIPE_PRICE_ID
  STRIPE_SUCCESS_URL
  STRIPE_CANCEL_URL
  STRIPE_PORTAL_RETURN_URL
  SUPABASE_URL
  SUPABASE_SERVICE_KEY  (NOT the anon key)
"""

import os
from datetime import datetime, timezone
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
PORTAL_RETURN_URL = os.environ.get("STRIPE_PORTAL_RETURN_URL", "https://your-domain.com/")

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


class PortalRequest(BaseModel):
    user_id: str


@router.post("/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest):
    """Start a Stripe Checkout session."""
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

        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            success_url=SUCCESS_URL,
            cancel_url=CANCEL_URL,
            allow_promotion_codes=True,
            metadata={"supabase_user_id": req.user_id},
            subscription_data={
                "metadata": {"supabase_user_id": req.user_id},
                "trial_period_days": 7,
            },
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        raise HTTPException(400, f"Stripe error: {e.user_message or str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Failed to create checkout: {str(e)}")


@router.post("/create-portal-session")
async def create_portal_session(req: PortalRequest):
    """Open the Stripe Customer Portal for subscription management."""
    if not stripe.api_key:
        raise HTTPException(500, "Stripe is not configured on this server")

    try:
        sb = get_supabase()
        profile_resp = sb.table("user_profiles").select("stripe_customer_id").eq("id", req.user_id).execute()
        rows = profile_resp.data or []
        customer_id = rows[0].get("stripe_customer_id") if rows else None

        if not customer_id:
            raise HTTPException(400, "No Stripe customer found for this user")

        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=PORTAL_RETURN_URL,
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        raise HTTPException(400, f"Stripe error: {e.user_message or str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Failed to create portal session: {str(e)}")


@router.get("/subscription/{user_id}")
async def get_subscription(user_id: str):
    """Return the current subscription state for a user."""
    try:
        sb = get_supabase()
        resp = sb.table("user_profiles").select(
            "subscription_status, subscription_tier, trial_ends_at, subscription_current_period_end"
        ).eq("id", user_id).execute()
        rows = resp.data or []
        if not rows:
            return {"status": "free"}
        row = rows[0]
        return {
            "status": row.get("subscription_status", "free"),
            "tier": row.get("subscription_tier"),
            "trial_ends_at": row.get("trial_ends_at"),
            "current_period_end": row.get("subscription_current_period_end"),
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load subscription: {str(e)}")


def _ts_to_iso(timestamp: Optional[int]) -> Optional[str]:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


@router.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    """Handle Stripe webhooks for subscription state changes."""
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
            user_id = obj.get("metadata", {}).get("supabase_user_id")
            if user_id:
                sb.table("user_profiles").update({
                    "subscription_status": "active",
                    "subscription_tier": "premium",
                    "stripe_subscription_id": obj.get("subscription"),
                }).eq("id", user_id).execute()

        elif event_type in ("customer.subscription.updated", "customer.subscription.created"):
            user_id = obj.get("metadata", {}).get("supabase_user_id")
            if not user_id:
                customer_id = obj.get("customer")
                lookup = sb.table("user_profiles").select("id").eq("stripe_customer_id", customer_id).execute()
                rows = lookup.data or []
                user_id = rows[0].get("id") if rows else None

            if user_id:
                status = obj.get("status")
                mapped_status = {
                    "trialing": "trial",
                    "active": "active",
                    "past_due": "past_due",
                    "canceled": "cancelled",
                    "unpaid": "past_due",
                    "incomplete": "free",
                    "incomplete_expired": "free",
                }.get(status, "free")

                update = {
                    "subscription_status": mapped_status,
                    "stripe_subscription_id": obj.get("id"),
                    "subscription_current_period_end": _ts_to_iso(obj.get("current_period_end")),
                    "trial_ends_at": _ts_to_iso(obj.get("trial_end")),
                }
                update = {k: v for k, v in update.items() if v is not None or k == "subscription_status"}
                sb.table("user_profiles").update(update).eq("id", user_id).execute()

        elif event_type == "customer.subscription.deleted":
            customer_id = obj.get("customer")
            lookup = sb.table("user_profiles").select("id").eq("stripe_customer_id", customer_id).execute()
            rows = lookup.data or []
            if rows:
                sb.table("user_profiles").update({
                    "subscription_status": "cancelled",
                    "stripe_subscription_id": None,
                }).eq("id", rows[0]["id"]).execute()

        elif event_type == "invoice.payment_failed":
            customer_id = obj.get("customer")
            lookup = sb.table("user_profiles").select("id").eq("stripe_customer_id", customer_id).execute()
            rows = lookup.data or []
            if rows:
                sb.table("user_profiles").update({
                    "subscription_status": "past_due",
                }).eq("id", rows[0]["id"]).execute()

    except Exception as e:
        # Don't 500 — Stripe will retry forever. Log and ack.
        print(f"Webhook handler error for {event_type}: {e}")

    return {"received": True}

/**
 * Subscription state hook.
 *
 * Reads subscription status directly from Supabase (user_profiles table)
 * so we don't need to round-trip through the backend on every render.
 *
 * Returns:
 *   {
 *     status: 'free' | 'trial' | 'active' | 'past_due' | 'cancelled',
 *     isPaid: boolean,           // true if user has access to paid features
 *     tier: string | null,
 *     trialEndsAt: Date | null,
 *     periodEnd: Date | null,
 *     loading: boolean,
 *   }
 */

import { useState, useEffect } from 'react'
import { supabase } from './supabase'

export function useSubscription(user) {
  const [state, setState] = useState({
    status: 'free',
    isPaid: false,
    tier: null,
    trialEndsAt: null,
    periodEnd: null,
    loading: true,
  })

  useEffect(() => {
    if (!user?.id || user.id === 'admin') {
      // Admin bypass for testing
      setState({
        status: user?.id === 'admin' ? 'active' : 'free',
        isPaid: user?.id === 'admin',
        tier: user?.id === 'admin' ? 'admin' : null,
        trialEndsAt: null,
        periodEnd: null,
        loading: false,
      })
      return
    }

    let cancelled = false

    const load = async () => {
      const { data, error } = await supabase
        .from('user_profiles')
        .select('subscription_status, subscription_tier, trial_ends_at, subscription_current_period_end')
        .eq('id', user.id)
        .maybeSingle()

      if (cancelled) return

      if (error || !data) {
        setState({
          status: 'free',
          isPaid: false,
          tier: null,
          trialEndsAt: null,
          periodEnd: null,
          loading: false,
        })
        return
      }

      const status = data.subscription_status || 'free'
      const isPaid = ['active', 'trial'].includes(status)

      setState({
        status,
        isPaid,
        tier: data.subscription_tier,
        trialEndsAt: data.trial_ends_at ? new Date(data.trial_ends_at) : null,
        periodEnd: data.subscription_current_period_end ? new Date(data.subscription_current_period_end) : null,
        loading: false,
      })
    }

    load()

    // Re-fetch when user_profiles updates (Stripe webhook → Supabase realtime)
    const channel = supabase
      .channel(`profile_${user.id}`)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'user_profiles',
          filter: `id=eq.${user.id}`,
        },
        () => load()
      )
      .subscribe()

    return () => {
      cancelled = true
      supabase.removeChannel(channel)
    }
  }, [user?.id])

  return state
}

const API_URL = import.meta.env.VITE_API_URL ||
  (window.location.hostname.includes('railway.app')
    ? 'https://immigration-case-predictor-production.up.railway.app'
    : 'http://localhost:8000')

/**
 * Start a Stripe Checkout session and redirect to it.
 */
export async function startCheckout(user) {
  const response = await fetch(`${API_URL}/stripe/create-checkout-session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: user.id, email: user.email }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || 'Could not start checkout')
  }
  const { url } = await response.json()
  window.location.href = url
}

/**
 * Open the Stripe customer portal (cancel, update card, etc.)
 */
export async function openPortal(user) {
  const response = await fetch(`${API_URL}/stripe/create-portal-session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: user.id }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err.detail || 'Could not open portal')
  }
  const { url } = await response.json()
  window.location.href = url
}

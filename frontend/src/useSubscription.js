/**
 * Access state hook (90-day one-time payment model).
 *
 * Reads access status directly from Supabase (user_profiles table).
 *
 * Returns:
 *   {
 *     status: 'free' | 'active' | 'expired',
 *     isPaid: boolean,           // true if user has access right now
 *     tier: string | null,
 *     periodEnd: Date | null,    // when current access expires
 *     daysRemaining: number,     // days until expiration
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
    periodEnd: null,
    daysRemaining: 0,
    loading: true,
  })

  useEffect(() => {
    if (!user?.id || user.id === 'admin') {
      // Admin bypass for testing
      setState({
        status: user?.id === 'admin' ? 'active' : 'free',
        isPaid: user?.id === 'admin',
        tier: user?.id === 'admin' ? 'admin' : null,
        periodEnd: null,
        daysRemaining: user?.id === 'admin' ? 9999 : 0,
        loading: false,
      })
      return
    }

    let cancelled = false

    const computeState = (data) => {
      if (!data) return { status: 'free', isPaid: false, tier: null, periodEnd: null, daysRemaining: 0 }

      const status = data.subscription_status || 'free'
      const periodEnd = data.subscription_current_period_end ? new Date(data.subscription_current_period_end) : null
      const now = new Date()

      // Check expiration client-side too (in case server hasn't run the lazy expiry yet)
      let effectiveStatus = status
      if (status === 'active' && periodEnd && periodEnd < now) {
        effectiveStatus = 'expired'
      }

      const isPaid = effectiveStatus === 'active'
      const daysRemaining = periodEnd && periodEnd > now
        ? Math.ceil((periodEnd - now) / (1000 * 60 * 60 * 24))
        : 0

      return {
        status: effectiveStatus,
        isPaid,
        tier: data.subscription_tier,
        periodEnd,
        daysRemaining,
      }
    }

    const load = async () => {
      const { data, error } = await supabase
        .from('user_profiles')
        .select('subscription_status, subscription_tier, subscription_current_period_end')
        .eq('id', user.id)
        .maybeSingle()

      if (cancelled) return

      if (error) {
        setState({ status: 'free', isPaid: false, tier: null, periodEnd: null, daysRemaining: 0, loading: false })
        return
      }

      setState({ ...computeState(data), loading: false })
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

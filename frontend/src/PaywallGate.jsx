/**
 * PaywallGate — wraps premium features and shows a paywall when locked.
 *
 * Usage:
 *   <PaywallGate user={user} feature="work_permit_guide">
 *     <YourPremiumComponent />
 *   </PaywallGate>
 */

import { useState } from 'react'
import { useSubscription, startCheckout, openPortal } from './useSubscription'
import { useLang } from './i18n.jsx'

export default function PaywallGate({ user, feature, children }) {
  const { isPaid, status, loading, trialEndsAt } = useSubscription(user)
  const { lang } = useLang()
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="animate-pulse text-slate-400">
          {lang === 'es' ? 'Verificando suscripción...' : 'Checking subscription...'}
        </div>
      </div>
    )
  }

  // Past due — keep them locked but explain why
  if (status === 'past_due') {
    return (
      <div className="bg-amber-50 border-2 border-amber-300 rounded-xl p-8 text-center">
        <div className="text-5xl mb-4">⚠️</div>
        <h2 className="text-xl font-semibold text-amber-900 mb-2">
          {lang === 'es' ? 'Pago pendiente' : 'Payment past due'}
        </h2>
        <p className="text-amber-800 mb-6 max-w-md mx-auto">
          {lang === 'es'
            ? 'Su último pago no se procesó. Actualice su método de pago para continuar usando ImmigrationAI.'
            : 'Your last payment did not go through. Update your payment method to continue using ImmigrationAI.'}
        </p>
        <button
          onClick={async () => {
            setBusy(true)
            try { await openPortal(user) } catch (e) { setError(e.message) } finally { setBusy(false) }
          }}
          disabled={busy}
          className="px-6 py-3 bg-amber-600 hover:bg-amber-700 text-white rounded-lg font-medium disabled:bg-slate-300"
        >
          {busy ? '...' : (lang === 'es' ? 'Actualizar método de pago' : 'Update payment method')}
        </button>
        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
      </div>
    )
  }

  if (isPaid) {
    return children
  }

  // Free tier — show paywall
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-200 rounded-2xl overflow-hidden">
        <div className="p-8 md:p-12">
          <div className="max-w-2xl mx-auto text-center">
            <div className="text-5xl mb-4">🔒</div>
            <h2 className="text-3xl font-bold text-slate-800 mb-3">
              {lang === 'es'
                ? 'Función Premium'
                : 'Premium Feature'}
            </h2>
            <p className="text-lg text-slate-600 mb-2">
              {lang === 'es'
                ? 'Acceda a todas las guías paso a paso de IRCC en español'
                : 'Get full access to all IRCC step-by-step guides in Spanish'}
            </p>
            <p className="text-sm text-slate-500 mb-8">
              {lang === 'es' ? 'Pruebe gratis 7 días, luego $9.99/mes. Cancele cuando quiera.' : 'Free 7-day trial, then $9.99/mo. Cancel anytime.'}
            </p>

            <div className="bg-white rounded-xl p-6 mb-6 border border-slate-200 text-left">
              <h3 className="font-semibold text-slate-800 mb-4 text-center">
                {lang === 'es' ? 'Lo que incluye su suscripción:' : "What's included:"}
              </h3>
              <ul className="space-y-3 max-w-md mx-auto">
                {[
                  lang === 'es'
                    ? 'Guías paso a paso para todos los formularios principales de IRCC (Permiso de trabajo, Patrocinio, Visa de visitante, etc.)'
                    : 'Step-by-step guides for all major IRCC forms (Work Permit, Sponsorship, Visitor Visa, etc.)',
                  lang === 'es'
                    ? 'Explicación campo por campo en español, con ejemplos reales'
                    : 'Field-by-field Spanish explanations with real examples',
                  lang === 'es'
                    ? 'Foto al rescate: tome foto de cualquier carta de IRCC y obtenga explicación en español'
                    : 'Photo to the rescue: snap any IRCC letter, get a Spanish explanation',
                  lang === 'es'
                    ? 'Sus borradores siempre seguros — nunca pierde su progreso'
                    : 'Your drafts always saved — never lose your progress',
                  lang === 'es'
                    ? 'Acceso 24/7 desde cualquier dispositivo'
                    : '24/7 access from any device',
                ].map((feat, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <span className="text-green-600 font-bold mt-0.5">✓</span>
                    <span className="text-sm text-slate-700">{feat}</span>
                  </li>
                ))}
              </ul>
            </div>

            <button
              onClick={async () => {
                setBusy(true)
                setError(null)
                try { await startCheckout(user) } catch (e) { setError(e.message) } finally { setBusy(false) }
              }}
              disabled={busy}
              className="w-full md:w-auto px-8 py-4 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold text-lg disabled:bg-slate-300 transition-colors"
            >
              {busy
                ? (lang === 'es' ? 'Cargando...' : 'Loading...')
                : (lang === 'es' ? 'Comenzar prueba gratis de 7 días' : 'Start 7-day free trial')}
            </button>

            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
                {error}
              </div>
            )}

            <p className="mt-4 text-xs text-slate-400">
              {lang === 'es'
                ? 'No se cobra durante la prueba. Cancele en cualquier momento.'
                : 'Not charged during trial. Cancel anytime.'}
            </p>
          </div>
        </div>
      </div>

      {/* Legal disclaimer in Spanish — IRPA s.91 compliance */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-600">
        <p className="font-medium mb-1">
          {lang === 'es' ? 'Aviso importante:' : 'Important notice:'}
        </p>
        <p>
          {lang === 'es'
            ? 'ImmigrationAI es una herramienta de información y preparación de documentos. No proveemos asesoría legal personalizada. Para asesoría legal, consulte con un abogado autorizado o un consultor RCIC.'
            : 'ImmigrationAI is an information and document preparation tool. We do not provide personalized legal advice. For legal advice, consult a licensed lawyer or RCIC.'}
        </p>
      </div>
    </div>
  )
}

/**
 * SubscriptionBadge — small inline indicator showing user's plan status.
 * Place in the header.
 */
export function SubscriptionBadge({ user }) {
  const { status, isPaid, trialEndsAt } = useSubscription(user)
  const { lang } = useLang()
  const [busy, setBusy] = useState(false)

  if (user?.id === 'admin') {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
        ⚡ Admin
      </span>
    )
  }

  if (status === 'trial' && trialEndsAt) {
    const daysLeft = Math.max(0, Math.ceil((trialEndsAt - new Date()) / (1000 * 60 * 60 * 24)))
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
        🎁 {lang === 'es' ? `Prueba: ${daysLeft}d` : `Trial: ${daysLeft}d`}
      </span>
    )
  }

  if (status === 'active') {
    return (
      <button
        onClick={async () => {
          setBusy(true)
          try { await openPortal(user) } catch (e) { console.error(e) } finally { setBusy(false) }
        }}
        disabled={busy}
        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 hover:bg-green-200"
        title={lang === 'es' ? 'Administrar suscripción' : 'Manage subscription'}
      >
        ✓ Premium
      </button>
    )
  }

  if (status === 'past_due') {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
        ⚠️ {lang === 'es' ? 'Pago pendiente' : 'Past due'}
      </span>
    )
  }

  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600">
      {lang === 'es' ? 'Gratis' : 'Free'}
    </span>
  )
}

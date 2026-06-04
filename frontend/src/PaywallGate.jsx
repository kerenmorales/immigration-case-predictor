/**
 * PaywallGate — wraps premium features and shows a paywall when locked.
 *
 * Uses one-time $9.99 payment for 90 days of access.
 *
 * Usage:
 *   <PaywallGate user={user} feature="work_permit_guide">
 *     <YourPremiumComponent />
 *   </PaywallGate>
 */

import { useState } from 'react'
import { useSubscription, startCheckout } from './useSubscription'
import { useLang } from './i18n.jsx'

export default function PaywallGate({ user, feature, children }) {
  const { isPaid, status, loading, daysRemaining } = useSubscription(user)
  const { lang } = useLang()
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="animate-pulse text-slate-400">
          {lang === 'es' ? 'Verificando acceso...' : 'Checking access...'}
        </div>
      </div>
    )
  }

  if (isPaid) {
    return children
  }

  const isExpired = status === 'expired'

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-200 rounded-2xl overflow-hidden">
        <div className="p-8 md:p-12">
          <div className="max-w-2xl mx-auto text-center">
            <div className="text-5xl mb-4">{isExpired ? '⏰' : '🔒'}</div>
            <h2 className="text-3xl font-bold text-slate-800 mb-3">
              {isExpired
                ? (lang === 'es' ? 'Su acceso ha expirado' : 'Your access has expired')
                : (lang === 'es' ? 'Función Premium' : 'Premium Feature')}
            </h2>
            <p className="text-lg text-slate-600 mb-2">
              {isExpired
                ? (lang === 'es' ? 'Renueve por otros 90 días para continuar' : 'Renew for another 90 days to continue')
                : (lang === 'es'
                  ? 'Acceso completo a las guías paso a paso de IRCC en español'
                  : 'Full access to all IRCC step-by-step guides in Spanish')}
            </p>
            <div className="my-6">
              <div className="inline-block bg-white px-6 py-3 rounded-xl border border-red-200 shadow-sm">
                <div className="text-sm text-slate-500">
                  {lang === 'es' ? 'Pago único' : 'One-time payment'}
                </div>
                <div className="text-3xl font-bold text-slate-800">
                  $9.99 <span className="text-sm font-normal text-slate-500">CAD</span>
                </div>
                <div className="text-sm text-green-700 font-medium">
                  {lang === 'es' ? '90 días de acceso completo' : '90 days of full access'}
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 mb-6 border border-slate-200 text-left">
              <h3 className="font-semibold text-slate-800 mb-4 text-center">
                {lang === 'es' ? 'Lo que incluye:' : "What's included:"}
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
                    ? 'Sus borradores siempre seguros — nunca pierde su progreso'
                    : 'Your drafts always saved — never lose your progress',
                  lang === 'es'
                    ? '90 días completos para terminar su aplicación, sin renovación automática'
                    : 'Full 90 days to finish your application, no auto-renewal',
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
                : isExpired
                  ? (lang === 'es' ? 'Renovar por $9.99' : 'Renew for $9.99')
                  : (lang === 'es' ? 'Obtener acceso por $9.99' : 'Get access for $9.99')}
            </button>

            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
                {error}
              </div>
            )}

            <p className="mt-4 text-xs text-slate-400">
              {lang === 'es'
                ? 'Pago único, sin cargos recurrentes. Pago seguro vía Stripe.'
                : 'One-time payment, no recurring charges. Secure payment via Stripe.'}
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
 * SubscriptionBadge — small inline indicator showing user's access status.
 * Place in the header.
 */
export function SubscriptionBadge({ user }) {
  const { status, isPaid, daysRemaining } = useSubscription(user)
  const { lang } = useLang()

  if (user?.id === 'admin') {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
        ⚡ Admin
      </span>
    )
  }

  if (status === 'active') {
    // Color shifts based on how much time is left
    const isLow = daysRemaining <= 7
    const isMedium = daysRemaining <= 30 && daysRemaining > 7
    const colorClass = isLow
      ? 'bg-amber-100 text-amber-800'
      : isMedium
        ? 'bg-blue-100 text-blue-800'
        : 'bg-green-100 text-green-800'

    return (
      <span
        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium ${colorClass}`}
        title={lang === 'es' ? `${daysRemaining} días restantes` : `${daysRemaining} days remaining`}
      >
        ✓ {daysRemaining}d
      </span>
    )
  }

  if (status === 'expired') {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
        ⏰ {lang === 'es' ? 'Expirado' : 'Expired'}
      </span>
    )
  }

  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600">
      {lang === 'es' ? 'Gratis' : 'Free'}
    </span>
  )
}

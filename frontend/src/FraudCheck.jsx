/**
 * Fraud Check — $4.99 AI second opinion on suspicious documents.
 *
 * Flow:
 *   1. User clicks "Start fraud check" → pays $4.99 via Stripe
 *   2. Returns to app → uploads a document/screenshot (max 10MB)
 *   3. Optional: provides context ("got this on WhatsApp from someone claiming to be IRCC")
 *   4. Claude Vision analyzes for fraud patterns
 *   5. Result shown with confidence score + smart CTAs:
 *      - High score → strongly recommend $29 consultation
 *      - Low score → recommend verification with IRCC
 */

import { useState, useEffect, useRef } from 'react'
import { useLang } from './i18n.jsx'

const API_URL = import.meta.env.VITE_API_URL ||
  (window.location.hostname.includes('railway.app')
    ? 'https://immigration-case-predictor-production.up.railway.app'
    : 'http://localhost:8000')


export default function FraudCheck({ user, setActiveTab }) {
  const { lang } = useLang()
  const [view, setView] = useState('intro')
  // 'intro' | 'paying' | 'upload' | 'analyzing' | 'result'
  const [activeCheck, setActiveCheck] = useState(null)
  const [error, setError] = useState(null)
  const [pastChecks, setPastChecks] = useState([])

  // On mount, check for active or past fraud checks
  useEffect(() => {
    if (!user?.id || user.id === 'admin') return
    fetch(`${API_URL}/fraud/checks/${user.id}`)
      .then(r => r.json())
      .then(data => {
        const checks = data.checks || []
        setPastChecks(checks)
        // Look for a paid check that hasn't been completed
        const inProgress = checks.find(c => c.is_paid && c.status !== 'completed' && c.status !== 'failed')
        if (inProgress) {
          setActiveCheck(inProgress)
          setView(inProgress.status === 'analyzing' ? 'analyzing' : 'upload')
        }
      })
      .catch(e => console.error('Failed to load fraud checks', e))
  }, [user?.id])

  // Listen for ?fraud_paid=true after Stripe redirect
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (params.get('fraud_paid') === 'true') {
      const checkId = params.get('check_id')
      window.history.replaceState({}, '', window.location.pathname)
      if (user?.id) {
        // Re-fetch and find the paid one
        setTimeout(() => {
          fetch(`${API_URL}/fraud/checks/${user.id}`)
            .then(r => r.json())
            .then(data => {
              const check = (data.checks || []).find(c => c.id === checkId) || (data.checks || [])[0]
              if (check && check.is_paid && check.status !== 'completed') {
                setActiveCheck(check)
                setView('upload')
              }
            })
        }, 1000)
      }
    }
  }, [user?.id])

  const startCheckout = async () => {
    setError(null)
    setView('paying')
    try {
      const resp = await fetch(`${API_URL}/fraud/create-checkout-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: user.id, email: user.email }),
      })
      if (!resp.ok) {
        const e = await resp.json().catch(() => ({}))
        throw new Error(e.detail || 'Could not start checkout')
      }
      const { url } = await resp.json()
      window.location.href = url
    } catch (e) {
      setError(e.message)
      setView('intro')
    }
  }

  if (view === 'intro') {
    return <FraudIntro lang={lang} onStart={startCheckout} pastChecks={pastChecks} onViewPast={(c) => { setActiveCheck(c); setView('result') }} error={error} />
  }
  if (view === 'paying') {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="animate-pulse text-slate-500">
          {lang === 'es' ? 'Redirigiendo a pago seguro...' : 'Redirecting to secure checkout...'}
        </div>
      </div>
    )
  }
  if (view === 'upload' && activeCheck) {
    return <FraudUpload user={user} activeCheck={activeCheck} onAnalyzing={() => setView('analyzing')} onComplete={(updated) => { setActiveCheck(updated); setView('result') }} lang={lang} />
  }
  if (view === 'analyzing') {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="text-2xl mb-3 animate-pulse">🔍</div>
        <p className="text-slate-700 font-medium">
          {lang === 'es' ? 'Analizando su documento...' : 'Analyzing your document...'}
        </p>
        <p className="text-sm text-slate-500 mt-2">
          {lang === 'es' ? 'Esto toma 20-40 segundos.' : 'This takes 20-40 seconds.'}
        </p>
      </div>
    )
  }
  if (view === 'result' && activeCheck) {
    return <FraudResult user={user} check={activeCheck} setActiveTab={setActiveTab} lang={lang} />
  }
  return null
}


// ============================================================
// 1. INTRO / PAYWALL
// ============================================================
function FraudIntro({ lang, onStart, pastChecks, onViewPast, error }) {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-amber-200 rounded-2xl p-8">
        <div className="flex items-start gap-4">
          <div className="text-5xl">🔍</div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-slate-800 mb-2">
              {lang === 'es' ? 'Detector de Fraudes — Segunda Opinión' : 'Fraud Check — Second Opinion'}
            </h2>
            <p className="text-slate-700 mb-4">
              {lang === 'es'
                ? '¿Recibió un correo, mensaje, o carta sospechosa? Suba el documento y obtenga una segunda opinión analizada por IA con un puntaje de confianza.'
                : 'Got a suspicious email, message, or letter? Upload it and get an AI-analyzed second opinion with a confidence score.'}
            </p>
            <div className="bg-white rounded-lg p-4 mb-4 border border-amber-200">
              <p className="text-2xl font-bold text-slate-800">
                $4.99 <span className="text-sm font-normal text-slate-500">CAD — {lang === 'es' ? 'pago único' : 'one-time'}</span>
              </p>
              <p className="text-sm text-slate-600 mt-1">
                {lang === 'es' ? 'Por análisis. Sin suscripciones.' : 'Per check. No subscriptions.'}
              </p>
            </div>
            <button
              onClick={onStart}
              className="px-6 py-3 bg-amber-600 hover:bg-amber-700 text-white rounded-lg font-semibold transition-colors"
            >
              {lang === 'es' ? 'Comenzar análisis por $4.99' : 'Start check for $4.99'}
            </button>
            {error && (
              <p className="mt-3 text-sm text-red-600">{error}</p>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl p-6 border border-slate-200">
          <h3 className="font-semibold text-slate-800 mb-3">
            {lang === 'es' ? '✅ Lo que detectamos' : '✅ What we detect'}
          </h3>
          <ul className="space-y-2 text-sm text-slate-600">
            {(lang === 'es' ? [
              'Dominios de correo falsos que pretenden ser de IRCC',
              'Solicitudes de pago por transferencia bancaria, Western Union o Bitcoin',
              'Amenazas con plazos urgentes (24 horas, "deportación inmediata")',
              'Promesas de aprobación garantizada de visa',
              '"Oficiales" personales por WhatsApp o gmail',
              'Errores tipográficos, logos incorrectos, sellos borrosos',
            ] : [
              'Fake email domains pretending to be IRCC',
              'Bank wire / Western Union / Bitcoin payment requests',
              'Threats with urgent deadlines (24 hours, "immediate deportation")',
              'Guaranteed visa approval promises',
              'Personal "officers" via WhatsApp or gmail',
              'Typos, wrong logos, blurry seals',
            ]).map((item, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-green-600 font-bold">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-white rounded-xl p-6 border border-slate-200">
          <h3 className="font-semibold text-slate-800 mb-3">
            {lang === 'es' ? '⚠️ Lo que NO hacemos' : '⚠️ What we DON\'T do'}
          </h3>
          <ul className="space-y-2 text-sm text-slate-600">
            {(lang === 'es' ? [
              'NO acusamos a personas o negocios específicos',
              'NO presentamos reportes ante las autoridades por usted',
              'NO le diremos "esto es 100% seguro"',
              'NO reemplazamos la asesoría de un abogado licenciado',
              'NO podemos verificar números de archivo IRCC reales',
            ] : [
              'We do NOT accuse specific persons or businesses',
              'We do NOT file reports to authorities on your behalf',
              'We will NOT say "this is 100% safe"',
              'We do NOT replace advice from a licensed lawyer',
              'We cannot verify real IRCC file numbers',
            ]).map((item, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-amber-600 font-bold">!</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
        <h3 className="font-semibold text-blue-900 mb-2">
          {lang === 'es' ? '💡 ¿Cómo funciona?' : '💡 How it works'}
        </h3>
        <ol className="space-y-2 text-sm text-blue-800 list-decimal list-inside">
          {(lang === 'es' ? [
            'Pague $4.99 (única vez)',
            'Suba una imagen o PDF del documento sospechoso (máximo 10MB)',
            'Nuestra IA analiza el documento en 20-40 segundos',
            'Reciba un puntaje de confianza con detalles específicos',
            'Si la IA detecta señales fuertes, le recomendaremos consultar con Keren Morales',
          ] : [
            'Pay $4.99 (one-time)',
            'Upload an image or PDF of the suspicious document (max 10MB)',
            'Our AI analyzes the document in 20-40 seconds',
            'Receive a confidence score with specific details',
            'If AI detects strong signals, we recommend consulting Keren Morales',
          ]).map((step, i) => <li key={i}>{step}</li>)}
        </ol>
      </div>

      {pastChecks.filter(c => c.status === 'completed').length > 0 && (
        <div className="bg-white rounded-xl p-6 border border-slate-200">
          <h3 className="font-semibold text-slate-800 mb-3">
            {lang === 'es' ? 'Sus análisis anteriores' : 'Your past checks'}
          </h3>
          <div className="space-y-2">
            {pastChecks.filter(c => c.status === 'completed').slice(0, 5).map((c) => (
              <button
                key={c.id}
                onClick={() => onViewPast(c)}
                className="w-full text-left p-3 bg-slate-50 hover:bg-slate-100 rounded-lg flex items-center justify-between"
              >
                <div>
                  <p className="text-sm font-medium text-slate-700">{c.document_filename || 'Documento'}</p>
                  <p className="text-xs text-slate-500">{new Date(c.created_at).toLocaleDateString()}</p>
                </div>
                <ConfidenceBadge score={c.confidence_score} label={c.confidence_label} lang={lang} />
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-600">
        <p className="font-semibold text-slate-700 mb-1">
          ⚖️ {lang === 'es' ? 'Antes de continuar, entienda:' : 'Before continuing, understand:'}
        </p>
        <p>
          {lang === 'es'
            ? 'Este análisis es una opinión automatizada por IA basada en patrones comunes de fraude. NO es asesoría legal, NO crea una relación abogado-cliente, y NO puede determinar con certeza si una persona u organización específica está cometiendo fraude. La IA puede equivocarse. Siempre verifique con IRCC al 1-888-242-2100 o con un abogado autorizado / RCIC antes de tomar acción.'
            : 'This analysis is an automated AI opinion based on common fraud patterns. It is NOT legal advice, does NOT create an attorney-client relationship, and CANNOT determine with certainty whether any specific person or organization is committing fraud. AI can be wrong. Always verify with IRCC at 1-888-242-2100 or a licensed lawyer / RCIC before taking action.'}
        </p>
      </div>
    </div>
  )
}


// ============================================================
// 2. UPLOAD COMPONENT
// ============================================================
function FraudUpload({ user, activeCheck, onAnalyzing, onComplete, lang }) {
  const [file, setFile] = useState(null)
  const [context, setContext] = useState('')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleSubmit = async () => {
    if (!file) {
      setError(lang === 'es' ? 'Por favor seleccione un archivo' : 'Please select a file')
      return
    }
    setUploading(true)
    setError(null)
    onAnalyzing()

    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('fraud_check_id', activeCheck.id)
      fd.append('user_id', user.id)
      fd.append('user_context', context)
      fd.append('user_email', user.email || '')

      const resp = await fetch(`${API_URL}/fraud/upload`, { method: 'POST', body: fd })
      if (!resp.ok) {
        const e = await resp.json().catch(() => ({}))
        throw new Error(e.detail || 'Analysis failed')
      }
      const updatedCheck = await resp.json()
      onComplete(updatedCheck)
    } catch (e) {
      setError(e.message)
      setUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="bg-amber-50 px-6 py-4 border-b border-amber-200">
          <h3 className="font-semibold text-slate-800">
            {lang === 'es' ? 'Suba el documento sospechoso' : 'Upload the suspicious document'}
          </h3>
          <p className="text-sm text-slate-600">
            {lang === 'es' ? 'Imágenes JPG/PNG/WEBP o PDF, máximo 10MB' : 'JPG/PNG/WEBP images or PDF, max 10MB'}
          </p>
        </div>

        <div className="p-6 space-y-4">
          <div className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center hover:border-amber-400">
            {!file ? (
              <>
                <div className="text-4xl mb-3">📤</div>
                <p className="text-slate-600 mb-3">
                  {lang === 'es' ? 'Seleccione una imagen o PDF' : 'Select an image or PDF'}
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp,.gif,.pdf"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-5 py-2 bg-amber-600 hover:bg-amber-700 text-white rounded-lg font-medium"
                >
                  {lang === 'es' ? 'Elegir archivo' : 'Choose file'}
                </button>
              </>
            ) : (
              <div>
                <div className="text-4xl mb-2">📎</div>
                <p className="text-slate-700 font-medium">{file.name}</p>
                <p className="text-xs text-slate-500 mb-3">{(file.size / 1024).toFixed(0)} KB</p>
                <button
                  onClick={() => { setFile(null); if (fileInputRef.current) fileInputRef.current.value = '' }}
                  className="text-sm text-slate-500 underline hover:text-slate-700"
                >
                  {lang === 'es' ? 'Cambiar archivo' : 'Change file'}
                </button>
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              {lang === 'es' ? 'Contexto (opcional)' : 'Context (optional)'}
            </label>
            <textarea
              value={context}
              onChange={(e) => setContext(e.target.value)}
              rows={3}
              placeholder={lang === 'es'
                ? 'Ej: Recibí esto por WhatsApp. Dicen que son de IRCC. Me piden $2,500 en 24 horas.'
                : 'Ex: Got this on WhatsApp. They say they\'re from IRCC. Demanding $2,500 in 24 hours.'}
              className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-amber-500 focus:border-transparent resize-none text-sm"
            />
            <p className="text-xs text-slate-500 mt-1">
              {lang === 'es' ? 'Esto ayuda al análisis a ser más preciso.' : 'This helps make the analysis more accurate.'}
            </p>
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>
          )}

          <button
            onClick={handleSubmit}
            disabled={!file || uploading}
            className="w-full px-6 py-3 bg-amber-600 hover:bg-amber-700 text-white rounded-lg font-semibold disabled:bg-slate-300 disabled:cursor-not-allowed"
          >
            {uploading
              ? (lang === 'es' ? 'Analizando...' : 'Analyzing...')
              : (lang === 'es' ? 'Iniciar análisis' : 'Start analysis')}
          </button>
        </div>
      </div>
    </div>
  )
}


// ============================================================
// 3. RESULT COMPONENT — confidence-scored output
// ============================================================
function FraudResult({ user, check, setActiveTab, lang }) {
  const score = check.confidence_score || 0
  const label = check.confidence_label || 'unclear'
  const patterns = check.patterns_detected || []
  const legitimate = check.patterns_legitimate || []
  const couldNotVerify = check.could_not_verify || []
  const extracted = check.extracted_entities || {}
  const domainChecks = check.domain_age_checks || []

  // Determine color and CTA based on score
  const config = scoreConfig(score, lang)

  return (
    <div className="space-y-6">
      {/* Hero Result */}
      <div className={`rounded-2xl p-8 border-2 ${config.bgClass} ${config.borderClass}`}>
        <div className="flex items-start gap-4 mb-4">
          <div className="text-5xl">{config.emoji}</div>
          <div className="flex-1">
            <h2 className={`text-2xl font-bold ${config.titleClass}`}>{config.title}</h2>
            <p className={`text-sm ${config.subTitleClass} mt-1`}>{config.subtitle}</p>
          </div>
          <div className="text-right">
            <div className={`text-4xl font-bold ${config.titleClass}`}>{score}<span className="text-xl">%</span></div>
            <div className={`text-xs ${config.subTitleClass}`}>
              {lang === 'es' ? 'confianza fraude' : 'fraud confidence'}
            </div>
          </div>
        </div>
        <div className="bg-white/60 rounded-lg p-4">
          <p className={`text-sm ${config.titleClass}`}>{check.recommended_action}</p>
        </div>
      </div>

      {/* Score legend — helps users interpret the score */}
      <div className="bg-white border border-slate-200 rounded-xl p-5">
        <h3 className="font-semibold text-slate-800 mb-2 text-sm">
          {lang === 'es' ? '📊 Cómo leer su puntaje (menor = mejor)' : '📊 How to read your score (lower = better)'}
        </h3>
        <ul className="space-y-1.5 text-sm">
          <li className="flex items-start gap-2">
            <span className={`px-2 py-0.5 rounded-full text-xs font-bold w-16 text-center ${score < 20 ? 'bg-green-200 text-green-900 ring-2 ring-green-400' : 'bg-green-100 text-green-800'}`}>0–19%</span>
            <span className="text-slate-700">
              {lang === 'es' ? 'No se detectaron señales — el documento parece legítimo ✅' : 'No fraud signals detected — looks legitimate ✅'}
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className={`px-2 py-0.5 rounded-full text-xs font-bold w-16 text-center ${score >= 20 && score < 40 ? 'bg-yellow-200 text-yellow-900 ring-2 ring-yellow-400' : 'bg-yellow-100 text-yellow-800'}`}>20–39%</span>
            <span className="text-slate-700">
              {lang === 'es' ? 'Pocos marcadores — probablemente legítimo, pero verifique' : 'Few markers — probably legit, but verify'}
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className={`px-2 py-0.5 rounded-full text-xs font-bold w-16 text-center ${score >= 40 && score < 60 ? 'bg-amber-200 text-amber-900 ring-2 ring-amber-400' : 'bg-amber-100 text-amber-800'}`}>40–59%</span>
            <span className="text-slate-700">
              {lang === 'es' ? 'Señales mixtas — inconcluyente' : 'Mixed signals — inconclusive'}
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className={`px-2 py-0.5 rounded-full text-xs font-bold w-16 text-center ${score >= 60 && score < 80 ? 'bg-orange-200 text-orange-900 ring-2 ring-orange-400' : 'bg-orange-100 text-orange-800'}`}>60–79%</span>
            <span className="text-slate-700">
              {lang === 'es' ? 'Varios patrones preocupantes' : 'Several concerning patterns'}
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className={`px-2 py-0.5 rounded-full text-xs font-bold w-16 text-center ${score >= 80 ? 'bg-red-200 text-red-900 ring-2 ring-red-400' : 'bg-red-100 text-red-800'}`}>80–95%</span>
            <span className="text-slate-700">
              {lang === 'es' ? 'Señales fuertes de fraude 🚨' : 'Strong fraud signals 🚨'}
            </span>
          </li>
        </ul>
      </div>

      {/* Saved + emailed confirmation */}
      <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 flex items-start gap-3">
        <div className="text-2xl">💾</div>
        <div className="flex-1 text-sm text-emerald-900">
          <p className="font-medium">
            {lang === 'es' ? 'Resultado guardado' : 'Result saved'}
          </p>
          <p className="text-emerald-800 mt-1">
            {lang === 'es'
              ? 'Guardamos este análisis en su cuenta y le enviamos una copia a su correo electrónico para que la conserve.'
              : 'We saved this analysis to your account and sent a copy to your email for your records.'}
          </p>
        </div>
      </div>

      {/* Verification helper — for documents mentioning lawyers/RCICs/notarios */}
      <VerificationHelper extracted={extracted} lang={lang} />

      {/* Domain WHOIS — flag newly-registered domains */}
      <DomainAgeFlags domainChecks={domainChecks} lang={lang} />

      {/* High-risk CTA */}
      {score >= 50 && (
        <div className="bg-red-50 border-2 border-red-300 rounded-xl p-6">
          <div className="flex items-start gap-4">
            <div className="text-4xl">🆘</div>
            <div className="flex-1">
              <h3 className="font-bold text-red-900 text-lg mb-2">
                {lang === 'es' ? 'Recomendamos consultar con Keren ANTES de tomar acción' : 'We recommend consulting Keren BEFORE taking action'}
              </h3>
              <p className="text-sm text-red-800 mb-4">
                {lang === 'es'
                  ? 'Hemos detectado señales serias de fraude en este documento. NO envíe dinero. NO comparta más información personal. Una consulta de 30 minutos con Keren Morales le dará una respuesta definitiva sobre cómo proceder de forma segura.'
                  : 'We detected serious fraud signals in this document. Do NOT send money. Do NOT share more personal information. A 30-minute consultation with Keren Morales will give you a definitive answer on how to proceed safely.'}
              </p>
              <button
                onClick={() => setActiveTab('intake')}
                className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold"
              >
                {lang === 'es' ? 'Reservar consulta de $29 →' : 'Book $29 consultation →'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Medium-risk soft CTA */}
      {score >= 20 && score < 50 && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
          <h3 className="font-semibold text-blue-900 mb-2">
            {lang === 'es' ? '¿Quiere una respuesta definitiva?' : 'Want a definitive answer?'}
          </h3>
          <p className="text-sm text-blue-800 mb-3">
            {lang === 'es'
              ? 'Una consulta de 30 minutos con Keren Morales puede determinar con certeza si esto es un fraude.'
              : 'A 30-minute consultation with Keren Morales can determine with certainty whether this is fraud.'}
          </p>
          <button
            onClick={() => setActiveTab('intake')}
            className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium text-sm"
          >
            {lang === 'es' ? 'Ver consulta de $29' : 'See $29 consultation'}
          </button>
        </div>
      )}

      {/* Patterns Detected */}
      {patterns.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-800 mb-4">
            🚩 {lang === 'es' ? 'Patrones detectados' : 'Patterns detected'}
          </h3>
          <ul className="space-y-3">
            {patterns.map((p, i) => (
              <li key={i} className="border-l-4 pl-4 py-1" style={{
                borderColor: p.severity === 'high' ? '#dc2626' : p.severity === 'medium' ? '#d97706' : '#2563eb',
              }}>
                <div className="flex items-start gap-2">
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                    p.severity === 'high' ? 'bg-red-100 text-red-800' :
                    p.severity === 'medium' ? 'bg-amber-100 text-amber-800' :
                    'bg-blue-100 text-blue-800'
                  }`}>
                    {p.severity === 'high' ? (lang === 'es' ? 'Alto' : 'High') :
                     p.severity === 'medium' ? (lang === 'es' ? 'Medio' : 'Medium') :
                     (lang === 'es' ? 'Bajo' : 'Low')}
                  </span>
                  <p className="font-medium text-slate-800">{p.pattern}</p>
                </div>
                <p className="text-sm text-slate-600 mt-1">{p.explanation}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Legitimate Patterns */}
      {legitimate.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-800 mb-3">
            ✓ {lang === 'es' ? 'Aspectos que parecen legítimos' : 'Aspects that look legitimate'}
          </h3>
          <ul className="space-y-2">
            {legitimate.map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                <span className="text-green-600 font-bold mt-0.5">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Could not verify */}
      {couldNotVerify.length > 0 && (
        <div className="bg-slate-50 rounded-xl border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-800 mb-3">
            ❓ {lang === 'es' ? 'Lo que no pudimos verificar' : 'What we couldn\'t verify'}
          </h3>
          <ul className="space-y-2">
            {couldNotVerify.map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                <span className="text-slate-500 mt-0.5">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Educational notes */}
      {check.educational_notes && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-6">
          <h3 className="font-semibold text-emerald-900 mb-2">
            💡 {lang === 'es' ? 'Aprenda a identificar IRCC legítimo' : 'Learn to identify legitimate IRCC'}
          </h3>
          <p className="text-sm text-emerald-800">{check.educational_notes}</p>
        </div>
      )}

      {/* Verification options */}
      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <h3 className="font-semibold text-slate-800 mb-3">
          {lang === 'es' ? 'Cómo verificar oficialmente' : 'How to verify officially'}
        </h3>
        <div className="space-y-3 text-sm">
          <div className="flex items-start gap-3">
            <span className="text-2xl">📞</span>
            <div>
              <p className="font-medium text-slate-700">
                {lang === 'es' ? 'Llame a IRCC' : 'Call IRCC'}
              </p>
              <p className="text-slate-600">1-888-242-2100</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="text-2xl">🌐</span>
            <div>
              <p className="font-medium text-slate-700">
                {lang === 'es' ? 'Sitio oficial de IRCC' : 'Official IRCC website'}
              </p>
              <a href="https://www.canada.ca/en/services/immigration-citizenship.html" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                canada.ca/immigration
              </a>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <span className="text-2xl">⚖️</span>
            <div>
              <p className="font-medium text-slate-700">
                {lang === 'es' ? 'Verificar consultor (CICC)' : 'Verify consultant (CICC)'}
              </p>
              <a href="https://college-ic.ca/protecting-the-public/find-an-immigration-consultant" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                college-ic.ca
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer — NOT legal advice */}
      <div className="bg-red-50 border-2 border-red-200 rounded-xl p-5">
        <h3 className="font-bold text-red-900 mb-2 flex items-center gap-2">
          <span>⚖️</span>
          <span>{lang === 'es' ? 'AVISO IMPORTANTE — Esto NO es asesoría legal' : 'IMPORTANT — This is NOT legal advice'}</span>
        </h3>
        <p className="text-sm text-red-800 leading-relaxed">
          {lang === 'es'
            ? 'Este análisis es una opinión automatizada generada por inteligencia artificial, basada en patrones comunes de fraude. NO constituye asesoría legal vinculante, NO crea una relación abogado-cliente, y NO reemplaza la consulta con un abogado autorizado o un consultor de inmigración (RCIC) registrado. Para certeza legal sobre su caso específico, consulte con un profesional licenciado o llame directamente a IRCC al 1-888-242-2100. La inteligencia artificial puede equivocarse — siempre verifique información crítica con una fuente oficial antes de tomar decisiones importantes.'
            : 'This analysis is an automated opinion generated by artificial intelligence, based on common fraud patterns. It does NOT constitute binding legal advice, does NOT create an attorney-client relationship, and does NOT replace consultation with a licensed lawyer or registered immigration consultant (RCIC). For legal certainty on your specific case, consult a licensed professional or call IRCC directly at 1-888-242-2100. Artificial intelligence can be wrong — always verify critical information with an official source before making important decisions.'}
        </p>
      </div>

      {/* Run another check */}
      <div className="text-center">
        <button
          onClick={() => window.location.reload()}
          className="px-5 py-2 border border-slate-300 hover:bg-slate-50 rounded-lg text-sm text-slate-600"
        >
          {lang === 'es' ? '🔍 Hacer otro análisis ($4.99)' : '🔍 Run another check ($4.99)'}
        </button>
      </div>
    </div>
  )
}


// ============================================================
// Helpers
// ============================================================
function scoreConfig(score, lang) {
  if (score >= 80) {
    return {
      bgClass: 'bg-red-50',
      borderClass: 'border-red-300',
      titleClass: 'text-red-900',
      subTitleClass: 'text-red-700',
      emoji: '🚨',
      title: lang === 'es' ? 'Señales fuertes de fraude' : 'Strong fraud signals',
      subtitle: lang === 'es' ? 'Múltiples patrones de fraude conocidos detectados' : 'Multiple known fraud patterns detected',
    }
  }
  if (score >= 50) {
    return {
      bgClass: 'bg-orange-50',
      borderClass: 'border-orange-300',
      titleClass: 'text-orange-900',
      subTitleClass: 'text-orange-700',
      emoji: '⚠️',
      title: lang === 'es' ? 'Probable fraude' : 'Likely fraud',
      subtitle: lang === 'es' ? 'Varios patrones preocupantes presentes' : 'Several concerning patterns present',
    }
  }
  if (score >= 20) {
    return {
      bgClass: 'bg-yellow-50',
      borderClass: 'border-yellow-300',
      titleClass: 'text-yellow-900',
      subTitleClass: 'text-yellow-700',
      emoji: '❓',
      title: lang === 'es' ? 'Señales mixtas' : 'Mixed signals',
      subtitle: lang === 'es' ? 'Algunos elementos inusuales pero inconcluyentes' : 'Some unusual elements but inconclusive',
    }
  }
  return {
    bgClass: 'bg-green-50',
    borderClass: 'border-green-300',
    titleClass: 'text-green-900',
    subTitleClass: 'text-green-700',
    emoji: '✅',
    title: lang === 'es' ? 'No se detectaron señales de fraude' : 'No fraud signals detected',
    subtitle: lang === 'es' ? 'Aún así, recomendamos verificar con IRCC' : 'Still, we recommend verifying with IRCC',
  }
}


function ConfidenceBadge({ score, label, lang }) {
  const config = scoreConfig(score || 0, lang)
  return (
    <div className={`px-3 py-1 rounded-full text-xs font-medium ${
      score >= 80 ? 'bg-red-100 text-red-800' :
      score >= 50 ? 'bg-orange-100 text-orange-800' :
      score >= 20 ? 'bg-yellow-100 text-yellow-800' :
      'bg-green-100 text-green-800'
    }`}>
      {score}% {config.emoji}
    </div>
  )
}


// ============================================================
// Verification Helper — direct links to CICC + provincial law societies
// ============================================================
//
// In Canada, only RCICs (registered with CICC), lawyers (registered with their
// provincial law society), and Quebec notaries can legally charge for immigration
// services. We don't auto-scrape registries — we hand the user the right link
// and let them verify in 60 seconds.

const PROVINCE_LINKS = {
  ON: { label: 'Law Society of Ontario', url: 'https://lso.ca/public-resources/finding-a-lawyer-or-paralegal/lawyer-and-paralegal-directory' },
  BC: { label: 'Law Society of BC', url: 'https://www.lawsociety.bc.ca/lsbc/apps/lkup/mbr-search.cfm' },
  QC: { label: 'Barreau du Québec', url: 'https://www.barreau.qc.ca/en/public/find-lawyer-notary/' },
  AB: { label: 'Law Society of Alberta', url: 'https://www.lawsociety.ab.ca/lawyer-directory/' },
  MB: { label: 'Law Society of Manitoba', url: 'https://lawsociety.mb.ca/for-the-public/find-a-lawyer/' },
  SK: { label: 'Law Society of Saskatchewan', url: 'https://www.lawsociety.sk.ca/lawyer-look-up/' },
  NS: { label: 'Nova Scotia Barristers Society', url: 'https://nsbs.org/find-a-lawyer/' },
  NB: { label: 'Law Society of New Brunswick', url: 'https://lawsociety-barreau.nb.ca/en/public/find-a-lawyer/' },
  PE: { label: 'Law Society of PEI', url: 'https://lspei.pe.ca/public/find-a-lawyer/' },
  NL: { label: 'Law Society of NL', url: 'https://lawsociety.nf.ca/for-the-public/find-a-lawyer/' },
}

function VerificationHelper({ extracted, lang }) {
  const names = extracted?.claimed_names || []
  const titles = extracted?.claimed_titles || []
  const licenses = extracted?.claimed_license_numbers || []
  const province = extracted?.claimed_province || ''

  // Don't render if nothing to verify
  if (names.length === 0 && titles.length === 0 && licenses.length === 0) {
    return null
  }

  // Show 4 most relevant law society links: claimed province first, then top 3 by population
  const defaultProvinces = ['ON', 'BC', 'QC', 'AB']
  const orderedProvinces = province && PROVINCE_LINKS[province]
    ? [province, ...defaultProvinces.filter(p => p !== province)]
    : defaultProvinces
  const visibleProvinces = orderedProvinces.slice(0, 4)

  return (
    <div className="bg-blue-50 border-2 border-blue-200 rounded-xl p-6">
      <div className="flex items-start gap-3 mb-3">
        <div className="text-3xl">🔍</div>
        <div className="flex-1">
          <h3 className="font-bold text-blue-900 text-lg">
            {lang === 'es' ? 'Verifique a esta persona' : 'Verify this person'}
          </h3>
          <p className="text-sm text-blue-800 mt-1">
            {lang === 'es'
              ? 'El documento menciona a alguien que dice tener autoridad. Verifíquelo gratis en los registros oficiales.'
              : 'The document mentions someone claiming authority. Verify them in official registries (free).'}
          </p>
        </div>
      </div>

      {/* What we extracted */}
      <div className="bg-white border border-blue-200 rounded-lg p-4 mb-4 text-sm">
        {names.length > 0 && (
          <div className="mb-1">
            <span className="font-semibold text-slate-700">
              {lang === 'es' ? 'Nombre(s):' : 'Name(s):'}
            </span>{' '}
            <span className="text-slate-800">{names.join(', ')}</span>
          </div>
        )}
        {titles.length > 0 && (
          <div className="mb-1">
            <span className="font-semibold text-slate-700">
              {lang === 'es' ? 'Título reclamado:' : 'Claimed title:'}
            </span>{' '}
            <span className="text-slate-800">{titles.join(', ')}</span>
          </div>
        )}
        {licenses.length > 0 && (
          <div className="mb-1">
            <span className="font-semibold text-slate-700">
              {lang === 'es' ? 'Licencia/Reg. #:' : 'License/Reg #:'}
            </span>{' '}
            <span className="font-mono text-slate-800">{licenses.join(', ')}</span>
          </div>
        )}
        {province && PROVINCE_LINKS[province] && (
          <div>
            <span className="font-semibold text-slate-700">
              {lang === 'es' ? 'Provincia:' : 'Province:'}
            </span>{' '}
            <span className="text-slate-800">{province}</span>
          </div>
        )}
      </div>

      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-4 text-xs text-amber-900">
        <strong>
          {lang === 'es' ? '⚖️ Ley canadiense:' : '⚖️ Canadian law:'}
        </strong>{' '}
        {lang === 'es'
          ? 'Solo abogados licenciados, RCIC registrados, o notarios de Quebec pueden cobrar por servicios de inmigración. Cualquier otra persona que cobre comete un delito federal (Sección 91 IRPA).'
          : 'Only licensed lawyers, registered RCICs, or Quebec notaries can charge for immigration services. Anyone else charging is committing a federal offense (Section 91 IRPA).'}
      </div>

      <p className="font-semibold text-blue-900 text-sm mb-2">
        {lang === 'es' ? 'Verifique aquí (gratis, 60 segundos):' : 'Verify here (free, 60 seconds):'}
      </p>

      <div className="space-y-3">
        {/* CICC — RCIC consultants */}
        <a
          href="https://college-ic.ca/protecting-the-public/find-an-immigration-consultant"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-start gap-3 p-3 bg-white border border-blue-200 hover:border-blue-400 hover:bg-blue-50 rounded-lg transition-colors"
        >
          <div className="text-2xl">🇨🇦</div>
          <div className="flex-1">
            <div className="font-semibold text-blue-900 text-sm">
              {lang === 'es' ? 'CICC — Consultor de inmigración (RCIC)' : 'CICC — Immigration consultant (RCIC)'}
            </div>
            <div className="text-xs text-slate-600 mt-0.5">
              {lang === 'es'
                ? 'Si dice ser "RCIC" o "consultor de inmigración registrado", busque por nombre o número R.'
                : 'If they claim to be "RCIC" or registered immigration consultant, search by name or R-number.'}
            </div>
          </div>
          <div className="text-blue-600 text-sm">→</div>
        </a>

        {/* Provincial law societies */}
        {visibleProvinces.map(prov => {
          const info = PROVINCE_LINKS[prov]
          if (!info) return null
          return (
            <a
              key={prov}
              href={info.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-start gap-3 p-3 bg-white border border-blue-200 hover:border-blue-400 hover:bg-blue-50 rounded-lg transition-colors"
            >
              <div className="text-2xl">⚖️</div>
              <div className="flex-1">
                <div className="font-semibold text-blue-900 text-sm">
                  {info.label}{province === prov ? ` ⭐ (${lang === 'es' ? 'provincia mencionada' : 'mentioned province'})` : ''}
                </div>
                <div className="text-xs text-slate-600 mt-0.5">
                  {lang === 'es' ? 'Si dice ser abogado en esta provincia.' : 'If they claim to be a lawyer in this province.'}
                </div>
              </div>
              <div className="text-blue-600 text-sm">→</div>
            </a>
          )
        })}

        {/* Quebec notaries */}
        <a
          href="https://www.cnq.org/en/find-a-notary/"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-start gap-3 p-3 bg-white border border-blue-200 hover:border-blue-400 hover:bg-blue-50 rounded-lg transition-colors"
        >
          <div className="text-2xl">📜</div>
          <div className="flex-1">
            <div className="font-semibold text-blue-900 text-sm">
              {lang === 'es' ? 'Chambre des notaires du Québec' : 'Chambre des notaires du Québec'}
            </div>
            <div className="text-xs text-slate-600 mt-0.5">
              {lang === 'es'
                ? 'Solo si dice ser notario de Quebec. ⚠️ "Notarios" en otras provincias NO pueden hacer trámites de inmigración.'
                : 'Only if they claim to be a Quebec notary. ⚠️ "Notarios" in other provinces CANNOT do immigration work.'}
            </div>
          </div>
          <div className="text-blue-600 text-sm">→</div>
        </a>
      </div>

      <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-800">
        <strong>
          {lang === 'es' ? '🚨 Si NO aparece en ningún registro:' : '🚨 If NOT found in any registry:'}
        </strong>{' '}
        {lang === 'es'
          ? 'Esa persona no puede cobrar legalmente por servicios de inmigración en Canadá. Es una señal muy fuerte de fraude. Considere reportar al Canadian Anti-Fraud Centre (1-888-495-8501).'
          : 'That person cannot legally charge for Canadian immigration services. This is a very strong fraud signal. Consider reporting to the Canadian Anti-Fraud Centre (1-888-495-8501).'}
      </div>
    </div>
  )
}


// ============================================================
// DomainAgeFlags — surfaces newly-registered domains as a strong fraud signal
// ============================================================
function DomainAgeFlags({ domainChecks, lang }) {
  const newDomains = (domainChecks || []).filter(d => d?.is_new)
  if (newDomains.length === 0) return null

  return (
    <div className="bg-orange-50 border-2 border-orange-200 rounded-xl p-5">
      <div className="flex items-start gap-3 mb-2">
        <div className="text-3xl">🌐</div>
        <div className="flex-1">
          <h3 className="font-bold text-orange-900">
            {lang === 'es' ? 'Dominios sospechosamente nuevos' : 'Suspiciously new domains'}
          </h3>
          <p className="text-sm text-orange-800 mt-1">
            {lang === 'es'
              ? 'Los dominios oficiales de IRCC tienen más de 20 años. Estos dominios son recientes:'
              : 'Official IRCC domains are 20+ years old. These domains are recent:'}
          </p>
        </div>
      </div>
      <ul className="space-y-2 mt-3">
        {newDomains.map((d, i) => (
          <li key={i} className="bg-white border border-orange-200 rounded-lg p-3 flex items-center justify-between">
            <div>
              <span className="font-mono text-sm text-orange-900 font-semibold">{d.domain}</span>
              <span className="text-xs text-orange-700 ml-2">
                {lang === 'es' ? 'registrado hace' : 'registered'} {d.age_days} {lang === 'es' ? 'días' : 'days ago'}
              </span>
            </div>
            <span className="px-2 py-1 bg-orange-200 text-orange-900 rounded-full text-xs font-bold">
              {lang === 'es' ? 'Sospechoso' : 'Suspicious'}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/**
 * AI Intake — $29 conversational intake interview powered by Claude.
 *
 * Flow:
 *   1. User picks application type (Sponsorship / Work Permit / Open-ended)
 *   2. User pays $29 via Stripe Checkout
 *   3. After payment, AI conversational intake begins
 *   4. User uploads supporting documents (10 files, 50MB max)
 *   5. AI generates dual summaries (lawyer + user) and emails both
 *   6. User books 30-min Calendly consultation
 */

import { useState, useEffect, useRef } from 'react'
import { supabase } from './supabase'
import { useLang } from './i18n.jsx'

const API_URL = import.meta.env.VITE_API_URL ||
  (window.location.hostname.includes('railway.app')
    ? 'https://immigration-case-predictor-production.up.railway.app'
    : 'http://localhost:8000')

const APPLICATION_TYPES = [
  {
    id: 'spousal_sponsorship',
    icon: '💑',
    name_es: 'Patrocinio Conyugal',
    name_en: 'Spousal Sponsorship',
    desc_es: 'Patrocinar a su esposo/a o pareja',
    desc_en: 'Sponsor your spouse or partner',
  },
  {
    id: 'work_permit',
    icon: '💼',
    name_es: 'Permiso de Trabajo',
    name_en: 'Work Permit',
    desc_es: 'Para trabajar en Canadá',
    desc_en: 'To work in Canada',
  },
  {
    id: 'open_ended',
    icon: '💬',
    name_es: 'Cuéntame tu situación',
    name_en: 'Tell me your situation',
    desc_es: 'No estoy seguro qué tipo necesito',
    desc_en: "I'm not sure which type I need",
  },
]

export default function AIIntake({ user }) {
  const { lang } = useLang()
  const [view, setView] = useState('select')
  // 'select' | 'paying' | 'chat' | 'documents' | 'finalizing' | 'completed'
  const [selectedType, setSelectedType] = useState(null)
  const [intakeSession, setIntakeSession] = useState(null)
  const [error, setError] = useState(null)

  // On mount, check if user has an in-progress intake
  useEffect(() => {
    if (!user?.id || user.id === 'admin') return
    fetch(`${API_URL}/intake/sessions/${user.id}`)
      .then(r => r.json())
      .then(data => {
        const sessions = data.sessions || []
        const inProgress = sessions.find(s => s.is_paid && s.status !== 'completed' && s.status !== 'abandoned')
        if (inProgress) {
          setIntakeSession(inProgress)
          setSelectedType(inProgress.application_type)
          setView(inProgress.status === 'awaiting_documents' ? 'documents' : 'chat')
        }
      })
      .catch(e => console.error('Failed to load intake sessions', e))
  }, [user?.id])

  // Listen for ?intake_paid=true in URL (after Stripe redirect)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (params.get('intake_paid') === 'true') {
      // Refresh to pick up newly-paid session
      const sessionId = params.get('session_id')
      window.history.replaceState({}, '', window.location.pathname)
      // The user_profiles realtime channel will trigger a re-fetch via parent
      setTimeout(() => {
        if (user?.id) {
          fetch(`${API_URL}/intake/sessions/${user.id}`)
            .then(r => r.json())
            .then(data => {
              const session = (data.sessions || []).find(s => s.id === sessionId) || (data.sessions || [])[0]
              if (session && session.is_paid) {
                setIntakeSession(session)
                setSelectedType(session.application_type)
                setView('chat')
              }
            })
        }
      }, 1000)
    }
  }, [user?.id])

  const startCheckout = async (appType) => {
    setError(null)
    setView('paying')
    try {
      const resp = await fetch(`${API_URL}/intake/create-checkout-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user.id,
          email: user.email,
          application_type: appType,
        }),
      })
      if (!resp.ok) {
        const e = await resp.json().catch(() => ({}))
        throw new Error(e.detail || 'Could not start checkout')
      }
      const { url } = await resp.json()
      window.location.href = url
    } catch (e) {
      setError(e.message)
      setView('select')
    }
  }

  // ============================================================
  // RENDER
  // ============================================================

  if (view === 'select') {
    return <IntakeTypeSelector lang={lang} appTypes={APPLICATION_TYPES} onSelect={(t) => { setSelectedType(t); startCheckout(t) }} error={error} />
  }

  if (view === 'paying') {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="animate-pulse text-slate-500 mb-2">
          {lang === 'es' ? 'Redirigiendo a pago seguro...' : 'Redirecting to secure checkout...'}
        </div>
        <p className="text-xs text-slate-400">Stripe</p>
      </div>
    )
  }

  if (view === 'chat' && intakeSession) {
    return (
      <IntakeChat
        user={user}
        intakeSession={intakeSession}
        onReadyForDocs={() => setView('documents')}
        lang={lang}
      />
    )
  }

  if (view === 'documents' && intakeSession) {
    return (
      <IntakeDocuments
        user={user}
        intakeSession={intakeSession}
        onFinalize={() => setView('finalizing')}
        lang={lang}
      />
    )
  }

  if (view === 'finalizing' && intakeSession) {
    return (
      <IntakeFinalize
        user={user}
        intakeSession={intakeSession}
        onComplete={(updated) => { setIntakeSession(updated); setView('completed') }}
        lang={lang}
      />
    )
  }

  if (view === 'completed' && intakeSession) {
    return <IntakeCompleted intakeSession={intakeSession} lang={lang} />
  }

  return null
}

// ============================================================
// 1. TYPE SELECTOR
// ============================================================
function IntakeTypeSelector({ lang, appTypes, onSelect, error }) {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-8 text-white">
        <h2 className="text-3xl font-bold mb-3">
          {lang === 'es' ? 'Consulta Personalizada de Inmigración' : 'Personalized Immigration Intake'}
        </h2>
        <p className="text-lg text-slate-300 mb-2">
          {lang === 'es'
            ? 'Una entrevista guiada por IA en español + 30 minutos de consulta con Keren Morales'
            : 'AI-guided interview in Spanish + 30-minute consultation with Keren Morales'}
        </p>
        <p className="text-2xl font-bold text-white mt-4">
          $29 CAD <span className="text-sm font-normal text-slate-400">{lang === 'es' ? '(pago único)' : '(one-time)'}</span>
        </p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h3 className="font-semibold text-slate-800 mb-4">
          {lang === 'es' ? '¿Qué tipo de caso necesita?' : 'What kind of case do you need help with?'}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {appTypes.map((t) => (
            <button
              key={t.id}
              onClick={() => onSelect(t.id)}
              className="p-6 border-2 border-slate-200 rounded-xl text-left hover:border-red-500 hover:shadow-md transition-all"
            >
              <div className="text-4xl mb-3">{t.icon}</div>
              <h4 className="font-semibold text-slate-800 mb-1">
                {lang === 'es' ? t.name_es : t.name_en}
              </h4>
              <p className="text-sm text-slate-500">
                {lang === 'es' ? t.desc_es : t.desc_en}
              </p>
            </button>
          ))}
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>
        )}
      </div>

      <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-6">
        <h4 className="font-semibold text-emerald-800 mb-2">
          {lang === 'es' ? '✨ Lo que incluye su consulta de $29:' : '✨ What your $29 includes:'}
        </h4>
        <ul className="space-y-2 text-sm text-emerald-800">
          {[
            lang === 'es'
              ? 'Entrevista de IA en español adaptada a su caso (15-20 minutos)'
              : 'Spanish AI interview adapted to your case (15-20 min)',
            lang === 'es'
              ? 'Subir documentos importantes (hasta 10 archivos, 50MB)'
              : 'Upload supporting documents (up to 10 files, 50MB)',
            lang === 'es'
              ? 'Resumen profesional de su caso, en español'
              : 'Professional case summary, in Spanish',
            lang === 'es'
              ? 'Lista personalizada de "Sus próximos 5 pasos"'
              : 'Personalized "Your next 5 steps" checklist',
            lang === 'es'
              ? 'Consulta de 30 minutos con Keren Morales (videollamada o teléfono)'
              : '30-minute consultation with Keren Morales (video or phone)',
            lang === 'es'
              ? 'Sus respuestas son confidenciales — nunca se comparten'
              : 'Your answers are confidential — never shared',
          ].map((item, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">✓</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-600">
        <p>
          {lang === 'es'
            ? 'Aviso: Esta herramienta provee información, no asesoría legal personalizada. La asesoría se entrega durante la consulta de 30 minutos con Keren Morales.'
            : 'Disclaimer: This tool provides information, not personalized legal advice. Advice is given during the 30-min consultation with Keren Morales.'}
        </p>
      </div>
    </div>
  )
}

// ============================================================
// 2. CHAT COMPONENT
// ============================================================
function IntakeChat({ user, intakeSession, onReadyForDocs, lang }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [readyForDocs, setReadyForDocs] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    fetch(`${API_URL}/intake/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: user.id, application_type: intakeSession.application_type }),
    })
      .then(r => r.json())
      .then(data => {
        setMessages(data.history || [])
        setLoading(false)
        if (data.status === 'awaiting_documents') {
          setReadyForDocs(true)
        }
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [intakeSession.id])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const sendMessage = async (e) => {
    e?.preventDefault()
    if (!input.trim() || loading) return
    const userMessage = input.trim()
    setInput('')
    setLoading(true)
    setError(null)

    const updatedMessages = [...messages, { role: 'user', content: userMessage }]
    setMessages(updatedMessages)

    try {
      const resp = await fetch(`${API_URL}/intake/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: intakeSession.id, user_id: user.id, message: userMessage }),
      })
      if (!resp.ok) throw new Error('AI request failed')
      const data = await resp.json()
      setMessages([...updatedMessages, { role: 'assistant', content: data.message }])
      if (data.ready_for_documents) {
        setReadyForDocs(true)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden flex flex-col" style={{ height: '600px' }}>
        <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-slate-800">
              {lang === 'es' ? 'Entrevista de Admisión' : 'Intake Interview'}
            </h3>
            <p className="text-xs text-slate-500">
              {lang === 'es' ? 'Confidencial · Powered by AI' : 'Confidential · Powered by AI'}
            </p>
          </div>
          {readyForDocs && (
            <button
              onClick={onReadyForDocs}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-sm font-medium"
            >
              {lang === 'es' ? 'Subir Documentos →' : 'Upload Documents →'}
            </button>
          )}
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-3">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                m.role === 'user' ? 'bg-red-600 text-white' : 'bg-slate-100 text-slate-800'
              }`}>
                <p className="text-sm whitespace-pre-wrap leading-relaxed">{m.content}</p>
              </div>
            </div>
          ))}
          {loading && messages.length > 0 && (
            <div className="flex justify-start">
              <div className="bg-slate-100 rounded-2xl px-4 py-3">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          )}
        </div>

        <form onSubmit={sendMessage} className="p-4 border-t border-slate-200 flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={lang === 'es' ? 'Escriba su respuesta...' : 'Type your answer...'}
            className="flex-1 border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
            disabled={loading || readyForDocs}
          />
          <button
            type="submit"
            disabled={loading || readyForDocs || !input.trim()}
            className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
          >
            {lang === 'es' ? 'Enviar' : 'Send'}
          </button>
        </form>
      </div>

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>
      )}
    </div>
  )
}

// ============================================================
// 3. DOCUMENT UPLOADER
// ============================================================
function IntakeDocuments({ user, intakeSession, onFinalize, lang }) {
  const [docs, setDocs] = useState([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    fetch(`${API_URL}/intake/documents/${intakeSession.id}?user_id=${user.id}`)
      .then(r => r.json())
      .then(data => setDocs(data.documents || []))
      .catch(e => console.error('Failed to load docs', e))
  }, [intakeSession.id])

  const handleUpload = async (event) => {
    const files = Array.from(event.target.files || [])
    if (!files.length) return
    setUploading(true)
    setError(null)

    for (const file of files) {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('session_id', intakeSession.id)
      fd.append('user_id', user.id)
      fd.append('document_category', 'other')

      try {
        const resp = await fetch(`${API_URL}/intake/upload`, { method: 'POST', body: fd })
        if (!resp.ok) {
          const e = await resp.json().catch(() => ({}))
          throw new Error(e.detail || `Upload failed: ${file.name}`)
        }
        const data = await resp.json()
        setDocs(prev => [...prev, data])
      } catch (e) {
        setError(e.message)
        break
      }
    }

    setUploading(false)
    event.target.value = ''
  }

  const totalBytes = docs.reduce((sum, d) => sum + (d.size_bytes || 0), 0)
  const totalMB = (totalBytes / (1024 * 1024)).toFixed(1)

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
          <h3 className="font-semibold text-slate-800">
            {lang === 'es' ? 'Suba sus documentos' : 'Upload your documents'}
          </h3>
          <p className="text-sm text-slate-500">
            {lang === 'es'
              ? 'Cualquier documento relacionado con su caso. Máximo 10 archivos, 50MB total.'
              : 'Any documents related to your case. Max 10 files, 50MB total.'}
          </p>
        </div>

        <div className="p-6">
          <div className="border-2 border-dashed border-slate-300 rounded-xl p-12 text-center hover:border-red-400 transition-colors">
            <div className="text-5xl mb-3">📄</div>
            <p className="text-slate-600 mb-3">
              {lang === 'es' ? 'Arrastre archivos aquí o haga clic para seleccionar' : 'Drag files here or click to select'}
            </p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png,.webp,.docx,.doc,.zip"
              onChange={handleUpload}
              className="hidden"
              disabled={uploading || docs.length >= 10}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading || docs.length >= 10}
              className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300"
            >
              {uploading
                ? (lang === 'es' ? 'Subiendo...' : 'Uploading...')
                : (lang === 'es' ? 'Seleccionar Archivos' : 'Select Files')}
            </button>
            <p className="text-xs text-slate-400 mt-3">
              PDF, JPG, PNG, DOCX, ZIP · {docs.length}/10 archivos · {totalMB}MB / 50MB
            </p>
          </div>

          {docs.length > 0 && (
            <ul className="mt-6 space-y-2">
              {docs.map((d) => (
                <li key={d.id || d.document_id || d.filename} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <span>📎</span>
                    <span className="text-sm font-medium text-slate-700">{d.filename}</span>
                    <span className="text-xs text-slate-400">
                      {((d.size_bytes || 0) / 1024).toFixed(0)} KB
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}

          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>
          )}
        </div>
      </div>

      <div className="flex justify-end gap-3">
        <button
          onClick={onFinalize}
          className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-semibold"
        >
          {docs.length > 0
            ? (lang === 'es' ? `Enviar caso (${docs.length} documentos) →` : `Submit case (${docs.length} documents) →`)
            : (lang === 'es' ? 'Enviar sin documentos →' : 'Submit without documents →')}
        </button>
      </div>
    </div>
  )
}

// ============================================================
// 4. FINALIZE (calls Claude to generate summaries)
// ============================================================
function IntakeFinalize({ user, intakeSession, onComplete, lang }) {
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API_URL}/intake/finalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: intakeSession.id, user_id: user.id }),
    })
      .then(r => r.json())
      .then(data => onComplete({ ...intakeSession, ...data }))
      .catch(e => setError(e.message))
  }, [])

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-8 text-center">
        <p className="text-red-700">{error}</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
      <div className="animate-pulse text-2xl mb-3">⏳</div>
      <p className="text-slate-700 font-medium">
        {lang === 'es' ? 'Procesando su caso...' : 'Processing your case...'}
      </p>
      <p className="text-sm text-slate-500 mt-2">
        {lang === 'es'
          ? 'Generando resumen y enviando notificaciones. Esto toma 30-60 segundos.'
          : 'Generating summary and sending notifications. Takes 30-60 seconds.'}
      </p>
    </div>
  )
}

// ============================================================
// 5. COMPLETED — show summary + Calendly button
// ============================================================
function IntakeCompleted({ intakeSession, lang }) {
  const calendlyLink = intakeSession.calendly_link || intakeSession.calendly_link_used || 'https://calendly.com/'
  const summary = intakeSession.user_summary_es || intakeSession.user_summary_en || ''
  const nextSteps = intakeSession.next_steps || []

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-emerald-50 to-green-50 border border-emerald-200 rounded-xl p-8 text-center">
        <div className="text-5xl mb-3">✅</div>
        <h2 className="text-2xl font-bold text-emerald-900 mb-2">
          {lang === 'es' ? '¡Su caso fue enviado!' : 'Your case was submitted!'}
        </h2>
        <p className="text-emerald-800">
          {lang === 'es'
            ? 'Hemos enviado un resumen a su correo y a Keren Morales. Reserve su consulta de 30 minutos abajo.'
            : "We've sent a summary to your email and to Keren Morales. Book your 30-minute consultation below."}
        </p>
      </div>

      <div className="bg-red-50 border-2 border-red-300 rounded-xl p-8 text-center">
        <h3 className="text-xl font-bold text-slate-800 mb-2">
          {lang === 'es' ? 'Reserve su consulta de 30 minutos' : 'Book your 30-minute consultation'}
        </h3>
        <p className="text-slate-600 mb-4 text-sm">
          {lang === 'es'
            ? 'Incluida en su pago. Sin costo adicional.'
            : 'Included in your payment. No additional charge.'}
        </p>
        <a
          href={calendlyLink}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block px-8 py-4 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold text-lg"
        >
          📅 {lang === 'es' ? 'Reservar Ahora' : 'Book Now'}
        </a>
      </div>

      {summary && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-800 mb-3">
            {lang === 'es' ? 'Su resumen' : 'Your summary'}
          </h3>
          <div className="whitespace-pre-wrap text-slate-700 text-sm leading-relaxed">{summary}</div>
        </div>
      )}

      {nextSteps.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-800 mb-4">
            {lang === 'es' ? 'Sus próximos 5 pasos' : 'Your next 5 steps'}
          </h3>
          <ol className="space-y-3">
            {nextSteps.map((s, i) => (
              <li key={i} className="flex gap-3">
                <span className="flex-shrink-0 w-7 h-7 bg-red-100 text-red-700 rounded-full flex items-center justify-center text-sm font-bold">
                  {i + 1}
                </span>
                <div>
                  <p className="font-medium text-slate-800">{s.step}</p>
                  {s.reason && <p className="text-sm text-slate-500 mt-1">{s.reason}</p>}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}

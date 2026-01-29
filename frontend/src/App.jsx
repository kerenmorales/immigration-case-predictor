import { useState, useEffect } from 'react'
import { supabase } from './supabase'

// Use environment variable, or detect Railway production, or fallback to localhost
const API_URL = import.meta.env.VITE_API_URL || 
  (window.location.hostname.includes('railway.app') 
    ? 'https://immigration-case-predictor-production.up.railway.app' 
    : 'http://localhost:8000')

function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('predictor')
  const [sponsorshipData, setSponsorshipData] = useState({})

  useEffect(() => {
    // Check current session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null)
      setLoading(false)
    })

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })

    return () => subscription.unsubscribe()
  }, [])

  const handleSignOut = async () => {
    await supabase.auth.signOut()
    setSponsorshipData({})
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return <AuthPage />
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-6">
        <div className="max-w-5xl mx-auto px-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">🇨🇦 Immigration Law Assistant</h1>
            <p className="text-indigo-100 mt-1">AI-powered tools for immigration lawyers</p>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-indigo-200">{user.email}</span>
            <button onClick={handleSignOut} className="text-sm bg-white/20 hover:bg-white/30 px-3 py-1 rounded">
              Sign Out
            </button>
          </div>
        </div>
      </header>

      <div className="bg-white border-b">
        <div className="max-w-5xl mx-auto px-4">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('predictor')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'predictor' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              Case Outcome Predictor
            </button>
            <button
              onClick={() => setActiveTab('sponsorship')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'sponsorship' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              Sponsorship Form Assistant
            </button>
            <button
              onClick={() => setActiveTab('reports')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'reports' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              Form Reports
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'history' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
            >
              My History
            </button>
          </nav>
        </div>
      </div>

      <main className="max-w-5xl mx-auto py-8 px-4">
        {activeTab === 'predictor' && <CasePredictor user={user} />}
        {activeTab === 'sponsorship' && <SponsorshipAssistant formData={sponsorshipData} setFormData={setSponsorshipData} user={user} />}
        {activeTab === 'reports' && <FormReports formData={sponsorshipData} />}
        {activeTab === 'history' && <UserHistory user={user} />}
      </main>
    </div>
  )
}

function AuthPage() {
  const [isLogin, setIsLogin] = useState(true)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [message, setMessage] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setMessage(null)

    try {
      if (isLogin) {
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
      } else {
        const { error } = await supabase.auth.signUp({ email, password })
        if (error) throw error
        setMessage('Check your email for a confirmation link!')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-600 to-purple-700 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">🇨🇦 Immigration Law Assistant</h1>
          <p className="text-gray-500 mt-2">AI-powered tools for immigration lawyers</p>
        </div>

        <div className="flex mb-6 bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setIsLogin(true)}
            className={`flex-1 py-2 rounded-md text-sm font-medium transition ${isLogin ? 'bg-white shadow text-indigo-600' : 'text-gray-500'}`}
          >
            Sign In
          </button>
          <button
            onClick={() => setIsLogin(false)}
            className={`flex-1 py-2 rounded-md text-sm font-medium transition ${!isLogin ? 'bg-white shadow text-indigo-600' : 'text-gray-500'}`}
          >
            Sign Up
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              placeholder="you@example.com"
              required
            />
          </div>
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              placeholder="••••••••"
              required
              minLength={6}
            />
          </div>

          {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>}
          {message && <div className="mb-4 p-3 bg-green-50 border border-green-200 text-green-700 rounded-lg text-sm">{message}</div>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 text-white py-3 rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-400 transition"
          >
            {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <p className="mt-6 text-center text-sm text-gray-500">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button onClick={() => setIsLogin(!isLogin)} className="text-indigo-600 font-medium hover:underline">
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </p>
      </div>
    </div>
  )
}

function CasePredictor({ user }) {
  const [caseText, setCaseText] = useState('')
  const [country, setCountry] = useState('')
  const [claimType, setClaimType] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: caseText, country_of_origin: country || null, claim_type: claimType || null })
      })
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || 'Prediction failed')
      }
      const result = await response.json()
      setPrediction(result)

      // Save to Supabase
      await supabase.from('predictions').insert({
        user_id: user.id,
        case_text: caseText,
        country_of_origin: country || null,
        claim_type: claimType || null,
        prediction: result.prediction,
        confidence: result.confidence,
        risk_level: result.risk_level,
        factors: result.factors
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Predict Case Outcome</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">Case Description / Facts</label>
            <textarea value={caseText} onChange={(e) => setCaseText(e.target.value)} rows={6} className="w-full border border-gray-300 rounded-md p-3 focus:ring-2 focus:ring-indigo-500" placeholder="Describe the case facts, including details about the refugee claim, persecution, or immigration application..." required />
          </div>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Country of Origin</label>
              <input type="text" value={country} onChange={(e) => setCountry(e.target.value)} className="w-full border border-gray-300 rounded-md p-2" placeholder="e.g., Iran" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Claim Type</label>
              <select value={claimType} onChange={(e) => setClaimType(e.target.value)} className="w-full border border-gray-300 rounded-md p-2">
                <option value="">Select type...</option>
                <option value="political">Political persecution</option>
                <option value="religious">Religious persecution</option>
                <option value="gender">Gender-based persecution</option>
                <option value="ethnic">Ethnic persecution</option>
              </select>
            </div>
          </div>
          <button type="submit" disabled={loading || !caseText.trim()} className="w-full bg-indigo-600 text-white py-3 rounded-md font-medium hover:bg-indigo-700 disabled:bg-gray-400">
            {loading ? 'Analyzing...' : 'Predict Outcome'}
          </button>
        </form>
      </div>
      {error && <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md mb-8">{error}</div>}
      {prediction && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
          
          <div className="flex items-center gap-4 mb-6">
            <div className={`px-4 py-2 rounded-full font-semibold text-lg ${prediction.prediction === 'Allowed' ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50'}`}>
              {prediction.prediction}
            </div>
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              prediction.risk_level === 'High' ? 'bg-blue-100 text-blue-700' : 
              prediction.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-700' : 
              'bg-gray-100 text-gray-700'
            }`}>
              {prediction.risk_level} Confidence
            </div>
          </div>

          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700">{prediction.risk_description}</p>
          </div>

          <div className="mb-6">
            <p className="text-sm font-medium text-gray-700 mb-2">Model Confidence</p>
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div className={`h-4 rounded-full ${prediction.prediction === 'Allowed' ? 'bg-green-500' : 'bg-red-500'}`} style={{ width: `${prediction.confidence * 100}%` }} />
            </div>
            <p className="text-sm text-gray-600 mt-1">{(prediction.confidence * 100).toFixed(1)}%</p>
          </div>

          {prediction.factors && prediction.factors.length > 0 && (
            <div className="mb-6">
              <p className="text-sm font-medium text-gray-700 mb-3">Key Legal Factors Detected</p>
              <div className="flex flex-wrap gap-2">
                {prediction.factors.map((f, i) => (
                  <span key={i} className={`px-3 py-1 rounded-full text-xs font-medium ${
                    f.impact === 'positive' ? 'bg-green-100 text-green-700' :
                    f.impact === 'negative' ? 'bg-red-100 text-red-700' :
                    'bg-gray-100 text-gray-600'
                  }`}>
                    {f.factor}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="mb-6 p-4 border border-indigo-100 bg-indigo-50 rounded-lg">
            <p className="text-sm font-medium text-indigo-800 mb-1">📊 Historical Context</p>
            <p className="text-sm text-indigo-700">{prediction.historical_context}</p>
          </div>

          <div className="mb-6 text-sm text-gray-500">
            <p className="font-medium">Data Source:</p>
            <p>{prediction.data_source?.name} • {prediction.data_source?.cases?.toLocaleString()} cases ({prediction.data_source?.period})</p>
          </div>

          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-md">
            <p className="text-sm text-yellow-800"><strong>⚠️ Disclaimer:</strong> This prediction is for informational purposes only. Always consult with a qualified immigration lawyer.</p>
          </div>
        </div>
      )}
    </div>
  )
}


function SponsorshipAssistant({ formData, setFormData, user }) {
  const [step, setStep] = useState(0)
  const [downloading, setDownloading] = useState(false)
  const [saving, setSaving] = useState(false)

  const updateField = (field, value) => setFormData(prev => ({ ...prev, [field]: value }))

  const isStep1Valid = () => {
    const required = ['sponsor_family_name', 'sponsor_given_name', 'sponsor_dob', 'sponsor_sex', 
      'sponsor_country_birth', 'sponsor_citizenship', 'sponsor_phone', 'sponsor_email',
      'sponsor_street', 'sponsor_city', 'sponsor_province', 'sponsor_postal']
    return required.every(field => formData[field]?.trim())
  }

  const isStep2Valid = () => {
    const required = ['applicant_family_name', 'applicant_given_name', 'applicant_dob', 'applicant_sex',
      'applicant_country_birth', 'applicant_citizenship', 'applicant_passport', 'applicant_passport_expiry',
      'applicant_marital', 'applicant_phone', 'applicant_email', 'applicant_address']
    return required.every(field => formData[field]?.trim())
  }

  const isStep3Valid = () => {
    const required = ['marriage_date', 'marriage_location', 'first_met_date', 'first_met_location',
      'relationship_start', 'living_together']
    return required.every(field => formData[field]?.trim())
  }

  const saveForm = async () => {
    setSaving(true)
    try {
      await supabase.from('sponsorship_forms').insert({
        user_id: user.id,
        form_data: formData,
        status: 'completed'
      })
    } catch (err) {
      console.error('Error saving form:', err)
    } finally {
      setSaving(false)
    }
  }

  const downloadFilledPDFs = async () => {
    setDownloading(true)
    try {
      const response = await fetch(`${API_URL}/api/fill-forms`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      if (!response.ok) throw new Error('Failed to generate PDF')
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'sponsorship_summary.pdf'
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      alert('Error downloading PDF: ' + err.message)
    } finally {
      setDownloading(false)
    }
  }

  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(formData, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'sponsorship_data.json'
    a.click()
  }

  if (step === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8 text-center">
        <h2 className="text-2xl font-bold mb-4">🇨🇦 Spousal Sponsorship Assistant</h2>
        <p className="text-gray-600 mb-6">Fill out your IRCC spousal sponsorship forms step by step.</p>
        <ul className="text-left inline-block mb-6 text-gray-600">
          <li className="mb-2">• IMM 1344 - Application to Sponsor</li>
          <li className="mb-2">• IMM 0008 - Generic Application Form</li>
          <li className="mb-2">• IMM 5532 - Relationship Information</li>
        </ul>
        <div>
          <button onClick={() => setStep(1)} className="bg-indigo-600 text-white px-8 py-3 rounded-full font-medium hover:bg-indigo-700">
            Start Application
          </button>
        </div>
      </div>
    )
  }

  if (step === 4) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8">
        <h2 className="text-2xl font-bold mb-6 text-center text-green-600">✓ Application Complete!</h2>
        
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-indigo-600 mb-2">Sponsor</h4>
              <p className="text-sm">{formData.sponsor_family_name}, {formData.sponsor_given_name}</p>
              <p className="text-sm text-gray-500">{formData.sponsor_email}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-indigo-600 mb-2">Applicant</h4>
              <p className="text-sm">{formData.applicant_family_name}, {formData.applicant_given_name}</p>
              <p className="text-sm text-gray-500">{formData.applicant_citizenship}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-indigo-600 mb-2">Relationship</h4>
              <p className="text-sm">Married: {formData.marriage_date}</p>
              <p className="text-sm text-gray-500">{formData.marriage_location}</p>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-4 justify-center">
          <button onClick={() => { saveForm(); downloadFilledPDFs(); }} disabled={downloading || saving} className="bg-green-600 text-white px-6 py-3 rounded-full font-medium hover:bg-green-700 disabled:bg-gray-400">
            {downloading ? 'Generating...' : '📄 Download Summary PDF'}
          </button>
          <button onClick={downloadJSON} className="bg-indigo-600 text-white px-6 py-3 rounded-full font-medium hover:bg-indigo-700">
            📋 Download JSON Data
          </button>
          <button onClick={() => { setStep(0); setFormData({}) }} className="bg-gray-500 text-white px-6 py-3 rounded-full font-medium hover:bg-gray-600">
            Start Over
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-500 mb-2">
          <span className={step >= 1 ? 'text-indigo-600 font-medium' : ''}>1. Sponsor</span>
          <span className={step >= 2 ? 'text-indigo-600 font-medium' : ''}>2. Applicant</span>
          <span className={step >= 3 ? 'text-indigo-600 font-medium' : ''}>3. Relationship</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div className="bg-indigo-600 h-2 rounded-full transition-all" style={{ width: `${(step / 3) * 100}%` }} />
        </div>
      </div>

      {step === 1 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Sponsor Information (IMM 1344)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Family Name *</label>
              <input type="text" value={formData.sponsor_family_name || ''} onChange={(e) => updateField('sponsor_family_name', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Given Name(s) *</label>
              <input type="text" value={formData.sponsor_given_name || ''} onChange={(e) => updateField('sponsor_given_name', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date of Birth *</label>
              <input type="date" value={formData.sponsor_dob || ''} onChange={(e) => updateField('sponsor_dob', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Sex *</label>
              <select value={formData.sponsor_sex || ''} onChange={(e) => updateField('sponsor_sex', e.target.value)} className="w-full border rounded-md p-2">
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="X">Another gender (X)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Country of Birth *</label>
              <input type="text" value={formData.sponsor_country_birth || ''} onChange={(e) => updateField('sponsor_country_birth', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Citizenship Status *</label>
              <select value={formData.sponsor_citizenship || ''} onChange={(e) => updateField('sponsor_citizenship', e.target.value)} className="w-full border rounded-md p-2">
                <option value="">Select...</option>
                <option value="Canadian Citizen">Canadian Citizen</option>
                <option value="Permanent Resident">Permanent Resident</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Phone *</label>
              <input type="tel" value={formData.sponsor_phone || ''} onChange={(e) => updateField('sponsor_phone', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email *</label>
              <input type="email" value={formData.sponsor_email || ''} onChange={(e) => updateField('sponsor_email', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Street Address *</label>
              <input type="text" value={formData.sponsor_street || ''} onChange={(e) => updateField('sponsor_street', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">City *</label>
              <input type="text" value={formData.sponsor_city || ''} onChange={(e) => updateField('sponsor_city', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Province *</label>
              <select value={formData.sponsor_province || ''} onChange={(e) => updateField('sponsor_province', e.target.value)} className="w-full border rounded-md p-2">
                <option value="">Select...</option>
                <option value="AB">Alberta</option>
                <option value="BC">British Columbia</option>
                <option value="MB">Manitoba</option>
                <option value="NB">New Brunswick</option>
                <option value="NL">Newfoundland</option>
                <option value="NS">Nova Scotia</option>
                <option value="ON">Ontario</option>
                <option value="PE">PEI</option>
                <option value="QC">Quebec</option>
                <option value="SK">Saskatchewan</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Postal Code *</label>
              <input type="text" value={formData.sponsor_postal || ''} onChange={(e) => updateField('sponsor_postal', e.target.value.toUpperCase())} className="w-full border rounded-md p-2" maxLength={7} />
            </div>
          </div>
          <div className="mt-6 flex justify-between items-center">
            <span className="text-sm text-gray-500">{isStep1Valid() ? '✓ Complete' : 'Fill all fields'}</span>
            <button onClick={() => setStep(2)} disabled={!isStep1Valid()} className="bg-indigo-600 text-white px-6 py-2 rounded-full hover:bg-indigo-700 disabled:bg-gray-300">
              Next →
            </button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Applicant Information (IMM 0008)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Family Name *</label>
              <input type="text" value={formData.applicant_family_name || ''} onChange={(e) => updateField('applicant_family_name', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Given Name(s) *</label>
              <input type="text" value={formData.applicant_given_name || ''} onChange={(e) => updateField('applicant_given_name', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date of Birth *</label>
              <input type="date" value={formData.applicant_dob || ''} onChange={(e) => updateField('applicant_dob', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Sex *</label>
              <select value={formData.applicant_sex || ''} onChange={(e) => updateField('applicant_sex', e.target.value)} className="w-full border rounded-md p-2">
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="X">Another gender (X)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Country of Birth *</label>
              <input type="text" value={formData.applicant_country_birth || ''} onChange={(e) => updateField('applicant_country_birth', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Citizenship *</label>
              <input type="text" value={formData.applicant_citizenship || ''} onChange={(e) => updateField('applicant_citizenship', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Passport Number *</label>
              <input type="text" value={formData.applicant_passport || ''} onChange={(e) => updateField('applicant_passport', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Passport Expiry *</label>
              <input type="date" value={formData.applicant_passport_expiry || ''} onChange={(e) => updateField('applicant_passport_expiry', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Marital Status *</label>
              <select value={formData.applicant_marital || ''} onChange={(e) => updateField('applicant_marital', e.target.value)} className="w-full border rounded-md p-2">
                <option value="">Select...</option>
                <option value="Married">Married</option>
                <option value="Common-Law">Common-Law</option>
                <option value="Single">Single</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Phone *</label>
              <input type="tel" value={formData.applicant_phone || ''} onChange={(e) => updateField('applicant_phone', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Email *</label>
              <input type="email" value={formData.applicant_email || ''} onChange={(e) => updateField('applicant_email', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Current Address *</label>
              <input type="text" value={formData.applicant_address || ''} onChange={(e) => updateField('applicant_address', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
          </div>
          <div className="mt-6 flex justify-between items-center">
            <button onClick={() => setStep(1)} className="text-gray-600 hover:text-gray-800">← Back</button>
            <span className="text-sm text-gray-500">{isStep2Valid() ? '✓ Complete' : 'Fill all fields'}</span>
            <button onClick={() => setStep(3)} disabled={!isStep2Valid()} className="bg-indigo-600 text-white px-6 py-2 rounded-full hover:bg-indigo-700 disabled:bg-gray-300">
              Next →
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Relationship Information (IMM 5532)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Marriage Date *</label>
              <input type="date" value={formData.marriage_date || ''} onChange={(e) => updateField('marriage_date', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Marriage Location *</label>
              <input type="text" value={formData.marriage_location || ''} onChange={(e) => updateField('marriage_location', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date First Met *</label>
              <input type="date" value={formData.first_met_date || ''} onChange={(e) => updateField('first_met_date', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Where First Met *</label>
              <input type="text" value={formData.first_met_location || ''} onChange={(e) => updateField('first_met_location', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Relationship Start *</label>
              <input type="date" value={formData.relationship_start || ''} onChange={(e) => updateField('relationship_start', e.target.value)} className="w-full border rounded-md p-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Living Together? *</label>
              <select value={formData.living_together || ''} onChange={(e) => updateField('living_together', e.target.value)} className="w-full border rounded-md p-2">
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>
          <div className="mt-6 flex justify-between items-center">
            <button onClick={() => setStep(2)} className="text-gray-600 hover:text-gray-800">← Back</button>
            <span className="text-sm text-gray-500">{isStep3Valid() ? '✓ Complete' : 'Fill all fields'}</span>
            <button onClick={() => setStep(4)} disabled={!isStep3Valid()} className="bg-green-600 text-white px-6 py-2 rounded-full hover:bg-green-700 disabled:bg-gray-300">
              Complete ✓
            </button>
          </div>
        </div>
      )}
    </div>
  )
}


function FormReports({ formData }) {
  const [activeForm, setActiveForm] = useState('imm1344')
  const hasData = Object.keys(formData).length > 0

  const Field = ({ label, value }) => (
    <div className="py-2 border-b border-gray-100">
      <span className="text-gray-500 text-sm">{label}</span>
      <p className="font-medium">{value || '—'}</p>
    </div>
  )

  if (!hasData) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8 text-center">
        <div className="text-6xl mb-4">📋</div>
        <h2 className="text-xl font-semibold mb-2">No Form Data Yet</h2>
        <p className="text-gray-600">Complete the Sponsorship Form Assistant to view your form reports here.</p>
      </div>
    )
  }

  return (
    <div>
      <div className="bg-white rounded-lg shadow-md mb-6">
        <div className="flex border-b">
          <button onClick={() => setActiveForm('imm1344')} className={`flex-1 py-4 text-center font-medium ${activeForm === 'imm1344' ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50' : 'text-gray-500'}`}>
            IMM 1344<br /><span className="text-xs font-normal">Sponsor</span>
          </button>
          <button onClick={() => setActiveForm('imm0008')} className={`flex-1 py-4 text-center font-medium ${activeForm === 'imm0008' ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50' : 'text-gray-500'}`}>
            IMM 0008<br /><span className="text-xs font-normal">Applicant</span>
          </button>
          <button onClick={() => setActiveForm('imm5532')} className={`flex-1 py-4 text-center font-medium ${activeForm === 'imm5532' ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50' : 'text-gray-500'}`}>
            IMM 5532<br /><span className="text-xs font-normal">Relationship</span>
          </button>
        </div>
      </div>

      {activeForm === 'imm1344' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">IMM 1344 - Application to Sponsor</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <Field label="Family Name" value={formData.sponsor_family_name} />
              <Field label="Given Name(s)" value={formData.sponsor_given_name} />
              <Field label="Date of Birth" value={formData.sponsor_dob} />
              <Field label="Sex" value={formData.sponsor_sex} />
              <Field label="Country of Birth" value={formData.sponsor_country_birth} />
              <Field label="Citizenship" value={formData.sponsor_citizenship} />
            </div>
            <div>
              <Field label="Phone" value={formData.sponsor_phone} />
              <Field label="Email" value={formData.sponsor_email} />
              <Field label="Address" value={formData.sponsor_street} />
              <Field label="City" value={formData.sponsor_city} />
              <Field label="Province" value={formData.sponsor_province} />
              <Field label="Postal Code" value={formData.sponsor_postal} />
            </div>
          </div>
        </div>
      )}

      {activeForm === 'imm0008' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">IMM 0008 - Generic Application</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <Field label="Family Name" value={formData.applicant_family_name} />
              <Field label="Given Name(s)" value={formData.applicant_given_name} />
              <Field label="Date of Birth" value={formData.applicant_dob} />
              <Field label="Sex" value={formData.applicant_sex} />
              <Field label="Country of Birth" value={formData.applicant_country_birth} />
              <Field label="Citizenship" value={formData.applicant_citizenship} />
            </div>
            <div>
              <Field label="Passport Number" value={formData.applicant_passport} />
              <Field label="Passport Expiry" value={formData.applicant_passport_expiry} />
              <Field label="Marital Status" value={formData.applicant_marital} />
              <Field label="Phone" value={formData.applicant_phone} />
              <Field label="Email" value={formData.applicant_email} />
              <Field label="Address" value={formData.applicant_address} />
            </div>
          </div>
        </div>
      )}

      {activeForm === 'imm5532' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">IMM 5532 - Relationship Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <Field label="Marriage Date" value={formData.marriage_date} />
              <Field label="Marriage Location" value={formData.marriage_location} />
              <Field label="Living Together" value={formData.living_together} />
            </div>
            <div>
              <Field label="Date First Met" value={formData.first_met_date} />
              <Field label="Where First Met" value={formData.first_met_location} />
              <Field label="Relationship Start" value={formData.relationship_start} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function UserHistory({ user }) {
  const [predictions, setPredictions] = useState([])
  const [forms, setForms] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('predictions')

  useEffect(() => {
    loadHistory()
  }, [user])

  const loadHistory = async () => {
    setLoading(true)
    try {
      const [predRes, formRes] = await Promise.all([
        supabase.from('predictions').select('*').eq('user_id', user.id).order('created_at', { ascending: false }).limit(20),
        supabase.from('sponsorship_forms').select('*').eq('user_id', user.id).order('created_at', { ascending: false }).limit(10)
      ])
      setPredictions(predRes.data || [])
      setForms(formRes.data || [])
    } catch (err) {
      console.error('Error loading history:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="bg-white rounded-lg shadow-md p-8 text-center text-gray-500">Loading history...</div>
  }

  return (
    <div>
      <div className="bg-white rounded-lg shadow-md mb-6">
        <div className="flex border-b">
          <button onClick={() => setActiveTab('predictions')} className={`flex-1 py-4 text-center font-medium ${activeTab === 'predictions' ? 'text-indigo-600 border-b-2 border-indigo-600' : 'text-gray-500'}`}>
            Predictions ({predictions.length})
          </button>
          <button onClick={() => setActiveTab('forms')} className={`flex-1 py-4 text-center font-medium ${activeTab === 'forms' ? 'text-indigo-600 border-b-2 border-indigo-600' : 'text-gray-500'}`}>
            Saved Forms ({forms.length})
          </button>
        </div>
      </div>

      {activeTab === 'predictions' && (
        <div className="space-y-4">
          {predictions.length === 0 ? (
            <div className="bg-white rounded-lg shadow-md p-8 text-center text-gray-500">
              No predictions yet. Try the Case Outcome Predictor!
            </div>
          ) : (
            predictions.map((p) => (
              <div key={p.id} className="bg-white rounded-lg shadow-md p-4">
                <div className="flex justify-between items-start mb-2">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${p.prediction === 'Allowed' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                    {p.prediction}
                  </span>
                  <span className="text-sm text-gray-400">{new Date(p.created_at).toLocaleDateString()}</span>
                </div>
                <p className="text-sm text-gray-600 line-clamp-2">{p.case_text?.substring(0, 150)}...</p>
                <div className="mt-2 text-sm text-gray-500">
                  Confidence: {(p.confidence * 100).toFixed(1)}% • {p.risk_level} confidence
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'forms' && (
        <div className="space-y-4">
          {forms.length === 0 ? (
            <div className="bg-white rounded-lg shadow-md p-8 text-center text-gray-500">
              No saved forms yet. Complete a sponsorship application!
            </div>
          ) : (
            forms.map((f) => (
              <div key={f.id} className="bg-white rounded-lg shadow-md p-4">
                <div className="flex justify-between items-start mb-2">
                  <span className="px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 text-indigo-700">
                    {f.status}
                  </span>
                  <span className="text-sm text-gray-400">{new Date(f.created_at).toLocaleDateString()}</span>
                </div>
                <p className="text-sm font-medium">
                  {f.form_data?.sponsor_family_name}, {f.form_data?.sponsor_given_name} → {f.form_data?.applicant_family_name}, {f.form_data?.applicant_given_name}
                </p>
                <p className="text-sm text-gray-500">Marriage: {f.form_data?.marriage_date} in {f.form_data?.marriage_location}</p>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}

export default App

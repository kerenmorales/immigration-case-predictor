import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [activeTab, setActiveTab] = useState('predictor')
  const [sponsorshipData, setSponsorshipData] = useState({})

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-6">
        <div className="max-w-5xl mx-auto px-4">
          <h1 className="text-2xl font-bold">🇨🇦 Immigration Law Assistant</h1>
          <p className="text-indigo-100 mt-1">AI-powered tools for immigration lawyers</p>
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
          </nav>
        </div>
      </div>

      <main className="max-w-5xl mx-auto py-8 px-4">
        {activeTab === 'predictor' && <CasePredictor />}
        {activeTab === 'sponsorship' && <SponsorshipAssistant formData={sponsorshipData} setFormData={setSponsorshipData} />}
        {activeTab === 'reports' && <FormReports formData={sponsorshipData} />}
      </main>
    </div>
  )
}

function CasePredictor() {
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
      if (!response.ok) throw new Error('Prediction failed')
      setPrediction(await response.json())
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
            <textarea value={caseText} onChange={(e) => setCaseText(e.target.value)} rows={6} className="w-full border border-gray-300 rounded-md p-3 focus:ring-2 focus:ring-indigo-500" placeholder="Describe the case facts..." required />
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
          <div className={`inline-block px-4 py-2 rounded-full font-semibold text-lg mb-6 ${prediction.prediction === 'Allowed' ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50'}`}>{prediction.prediction}</div>
          <div className="mb-6">
            <p className="text-sm font-medium text-gray-700 mb-2">Confidence</p>
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div className="bg-indigo-600 h-4 rounded-full" style={{ width: `${prediction.confidence * 100}%` }} />
            </div>
            <p className="text-sm text-gray-600 mt-1">{(prediction.confidence * 100).toFixed(1)}%</p>
          </div>
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-md">
            <p className="text-sm text-yellow-800"><strong>Disclaimer:</strong> For informational purposes only.</p>
          </div>
        </div>
      )}
    </div>
  )
}

function SponsorshipAssistant({ formData, setFormData }) {
  const [step, setStep] = useState(0) // 0=start, 1=sponsor, 2=applicant, 3=relationship, 4=complete
  const [downloading, setDownloading] = useState(false)

  const updateField = (field, value) => setFormData(prev => ({ ...prev, [field]: value }))

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
    } catch (err) {
      alert('Error: ' + err.message)
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

  // Start screen
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

  // Complete screen
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
              <p className="text-sm text-gray-500">{formData.sponsor_city}, {formData.sponsor_province}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-indigo-600 mb-2">Applicant</h4>
              <p className="text-sm">{formData.applicant_family_name}, {formData.applicant_given_name}</p>
              <p className="text-sm text-gray-500">{formData.applicant_citizenship}</p>
              <p className="text-sm text-gray-500">Passport: {formData.applicant_passport}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-indigo-600 mb-2">Relationship</h4>
              <p className="text-sm">Married: {formData.marriage_date}</p>
              <p className="text-sm text-gray-500">{formData.marriage_location}</p>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-4 justify-center">
          <button onClick={downloadFilledPDFs} disabled={downloading} className="bg-green-600 text-white px-6 py-3 rounded-full font-medium hover:bg-green-700 disabled:bg-gray-400">
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

  // Form steps
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Progress */}
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

      {/* Step 1: Sponsor Info */}
      {step === 1 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Sponsor Information (IMM 1344)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Family Name (Surname) *</label>
              <input type="text" value={formData.sponsor_family_name || ''} onChange={(e) => updateField('sponsor_family_name', e.target.value)} className="w-full border rounded-md p-2" placeholder="e.g., Smith" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Given Name(s) *</label>
              <input type="text" value={formData.sponsor_given_name || ''} onChange={(e) => updateField('sponsor_given_name', e.target.value)} className="w-full border rounded-md p-2" placeholder="e.g., John Michael" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date of Birth *</label>
              <input type="date" value={formData.sponsor_dob || ''} onChange={(e) => updateField('sponsor_dob', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Sex *</label>
              <select value={formData.sponsor_sex || ''} onChange={(e) => updateField('sponsor_sex', e.target.value)} className="w-full border rounded-md p-2" required>
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="X">Another gender (X)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Country of Birth *</label>
              <input type="text" value={formData.sponsor_country_birth || ''} onChange={(e) => updateField('sponsor_country_birth', e.target.value)} className="w-full border rounded-md p-2" placeholder="e.g., Canada" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Citizenship Status *</label>
              <select value={formData.sponsor_citizenship || ''} onChange={(e) => updateField('sponsor_citizenship', e.target.value)} className="w-full border rounded-md p-2" required>
                <option value="">Select...</option>
                <option value="Canadian Citizen">Canadian Citizen</option>
                <option value="Permanent Resident">Permanent Resident</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Phone Number *</label>
              <input type="tel" value={formData.sponsor_phone || ''} onChange={(e) => updateField('sponsor_phone', e.target.value)} className="w-full border rounded-md p-2" placeholder="(123) 456-7890" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email Address *</label>
              <input type="email" value={formData.sponsor_email || ''} onChange={(e) => updateField('sponsor_email', e.target.value)} className="w-full border rounded-md p-2" placeholder="email@example.com" required />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Street Address *</label>
              <input type="text" value={formData.sponsor_street || ''} onChange={(e) => updateField('sponsor_street', e.target.value)} className="w-full border rounded-md p-2" placeholder="123 Main Street, Apt 4" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">City *</label>
              <input type="text" value={formData.sponsor_city || ''} onChange={(e) => updateField('sponsor_city', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Province *</label>
              <select value={formData.sponsor_province || ''} onChange={(e) => updateField('sponsor_province', e.target.value)} className="w-full border rounded-md p-2" required>
                <option value="">Select...</option>
                <option value="AB">Alberta</option>
                <option value="BC">British Columbia</option>
                <option value="MB">Manitoba</option>
                <option value="NB">New Brunswick</option>
                <option value="NL">Newfoundland and Labrador</option>
                <option value="NS">Nova Scotia</option>
                <option value="NT">Northwest Territories</option>
                <option value="NU">Nunavut</option>
                <option value="ON">Ontario</option>
                <option value="PE">Prince Edward Island</option>
                <option value="QC">Quebec</option>
                <option value="SK">Saskatchewan</option>
                <option value="YT">Yukon</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Postal Code *</label>
              <input type="text" value={formData.sponsor_postal || ''} onChange={(e) => updateField('sponsor_postal', e.target.value.toUpperCase())} className="w-full border rounded-md p-2" placeholder="A1A 1A1" maxLength={7} required />
            </div>
          </div>
          <div className="mt-6 flex justify-end">
            <button onClick={() => setStep(2)} className="bg-indigo-600 text-white px-6 py-2 rounded-full hover:bg-indigo-700">
              Next: Applicant Info →
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Applicant Info */}
      {step === 2 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Applicant Information (IMM 0008)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Family Name (Surname) *</label>
              <input type="text" value={formData.applicant_family_name || ''} onChange={(e) => updateField('applicant_family_name', e.target.value)} className="w-full border rounded-md p-2" placeholder="As shown on passport" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Given Name(s) *</label>
              <input type="text" value={formData.applicant_given_name || ''} onChange={(e) => updateField('applicant_given_name', e.target.value)} className="w-full border rounded-md p-2" placeholder="As shown on passport" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date of Birth *</label>
              <input type="date" value={formData.applicant_dob || ''} onChange={(e) => updateField('applicant_dob', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Sex *</label>
              <select value={formData.applicant_sex || ''} onChange={(e) => updateField('applicant_sex', e.target.value)} className="w-full border rounded-md p-2" required>
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="X">Another gender (X)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Country of Birth *</label>
              <input type="text" value={formData.applicant_country_birth || ''} onChange={(e) => updateField('applicant_country_birth', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Country of Citizenship *</label>
              <input type="text" value={formData.applicant_citizenship || ''} onChange={(e) => updateField('applicant_citizenship', e.target.value)} className="w-full border rounded-md p-2" placeholder="e.g., India, Philippines" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Passport Number *</label>
              <input type="text" value={formData.applicant_passport || ''} onChange={(e) => updateField('applicant_passport', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Passport Expiry Date *</label>
              <input type="date" value={formData.applicant_passport_expiry || ''} onChange={(e) => updateField('applicant_passport_expiry', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Marital Status *</label>
              <select value={formData.applicant_marital || ''} onChange={(e) => updateField('applicant_marital', e.target.value)} className="w-full border rounded-md p-2" required>
                <option value="">Select...</option>
                <option value="Married">Married</option>
                <option value="Common-Law">Common-Law</option>
                <option value="Single">Single</option>
                <option value="Divorced">Divorced</option>
                <option value="Widowed">Widowed</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Phone Number *</label>
              <input type="tel" value={formData.applicant_phone || ''} onChange={(e) => updateField('applicant_phone', e.target.value)} className="w-full border rounded-md p-2" placeholder="Include country code" required />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Email Address *</label>
              <input type="email" value={formData.applicant_email || ''} onChange={(e) => updateField('applicant_email', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Current Address *</label>
              <input type="text" value={formData.applicant_address || ''} onChange={(e) => updateField('applicant_address', e.target.value)} className="w-full border rounded-md p-2" placeholder="Full address including country" required />
            </div>
          </div>
          <div className="mt-6 flex justify-between">
            <button onClick={() => setStep(1)} className="text-gray-600 px-6 py-2 hover:text-gray-800">← Back</button>
            <button onClick={() => setStep(3)} className="bg-indigo-600 text-white px-6 py-2 rounded-full hover:bg-indigo-700">
              Next: Relationship →
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Relationship Info */}
      {step === 3 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Relationship Information (IMM 5532)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Marriage Date *</label>
              <input type="date" value={formData.marriage_date || ''} onChange={(e) => updateField('marriage_date', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Marriage Location *</label>
              <input type="text" value={formData.marriage_location || ''} onChange={(e) => updateField('marriage_location', e.target.value)} className="w-full border rounded-md p-2" placeholder="City, Country" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Date You First Met *</label>
              <input type="date" value={formData.first_met_date || ''} onChange={(e) => updateField('first_met_date', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Where You First Met *</label>
              <input type="text" value={formData.first_met_location || ''} onChange={(e) => updateField('first_met_location', e.target.value)} className="w-full border rounded-md p-2" placeholder="City, Country or Online" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Relationship Start Date *</label>
              <input type="date" value={formData.relationship_start || ''} onChange={(e) => updateField('relationship_start', e.target.value)} className="w-full border rounded-md p-2" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Currently Living Together? *</label>
              <select value={formData.living_together || ''} onChange={(e) => updateField('living_together', e.target.value)} className="w-full border rounded-md p-2" required>
                <option value="">Select...</option>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>
          <div className="mt-6 flex justify-between">
            <button onClick={() => setStep(2)} className="text-gray-600 px-6 py-2 hover:text-gray-800">← Back</button>
            <button onClick={() => setStep(4)} className="bg-green-600 text-white px-6 py-2 rounded-full hover:bg-green-700">
              Complete Application ✓
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
      {/* Form Tabs */}
      <div className="bg-white rounded-lg shadow-md mb-6">
        <div className="flex border-b">
          <button
            onClick={() => setActiveForm('imm1344')}
            className={`flex-1 py-4 px-4 text-center font-medium ${activeForm === 'imm1344' ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50' : 'text-gray-500 hover:bg-gray-50'}`}
          >
            IMM 1344<br /><span className="text-xs font-normal">Application to Sponsor</span>
          </button>
          <button
            onClick={() => setActiveForm('imm0008')}
            className={`flex-1 py-4 px-4 text-center font-medium ${activeForm === 'imm0008' ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50' : 'text-gray-500 hover:bg-gray-50'}`}
          >
            IMM 0008<br /><span className="text-xs font-normal">Generic Application</span>
          </button>
          <button
            onClick={() => setActiveForm('imm5532')}
            className={`flex-1 py-4 px-4 text-center font-medium ${activeForm === 'imm5532' ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50' : 'text-gray-500 hover:bg-gray-50'}`}
          >
            IMM 5532<br /><span className="text-xs font-normal">Relationship Info</span>
          </button>
        </div>
      </div>

      {/* IMM 1344 - Sponsor */}
      {activeForm === 'imm1344' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-bold text-gray-800">IMM 1344 - Application to Sponsor</h2>
              <p className="text-sm text-gray-500">Sponsorship Agreement and Undertaking</p>
            </div>
            <span className="bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm font-medium">🍁 IRCC</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <h3 className="font-semibold text-indigo-600 mb-3 mt-4">Personal Information</h3>
              <Field label="Family Name (Surname)" value={formData.sponsor_family_name} />
              <Field label="Given Name(s)" value={formData.sponsor_given_name} />
              <Field label="Date of Birth" value={formData.sponsor_dob} />
              <Field label="Sex" value={formData.sponsor_sex} />
              <Field label="Country of Birth" value={formData.sponsor_country_birth} />
              <Field label="Citizenship Status" value={formData.sponsor_citizenship} />
            </div>
            <div>
              <h3 className="font-semibold text-indigo-600 mb-3 mt-4">Contact Information</h3>
              <Field label="Phone Number" value={formData.sponsor_phone} />
              <Field label="Email Address" value={formData.sponsor_email} />
              <Field label="Street Address" value={formData.sponsor_street} />
              <Field label="City" value={formData.sponsor_city} />
              <Field label="Province/Territory" value={formData.sponsor_province} />
              <Field label="Postal Code" value={formData.sponsor_postal} />
            </div>
          </div>
        </div>
      )}

      {/* IMM 0008 - Applicant */}
      {activeForm === 'imm0008' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-bold text-gray-800">IMM 0008 - Generic Application Form</h2>
              <p className="text-sm text-gray-500">Principal Applicant Information</p>
            </div>
            <span className="bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm font-medium">🍁 IRCC</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <h3 className="font-semibold text-indigo-600 mb-3 mt-4">Personal Details</h3>
              <Field label="Family Name (Surname)" value={formData.applicant_family_name} />
              <Field label="Given Name(s)" value={formData.applicant_given_name} />
              <Field label="Date of Birth" value={formData.applicant_dob} />
              <Field label="Sex" value={formData.applicant_sex} />
              <Field label="Country of Birth" value={formData.applicant_country_birth} />
              <Field label="Country of Citizenship" value={formData.applicant_citizenship} />
              <Field label="Marital Status" value={formData.applicant_marital} />
            </div>
            <div>
              <h3 className="font-semibold text-indigo-600 mb-3 mt-4">Passport & Contact</h3>
              <Field label="Passport Number" value={formData.applicant_passport} />
              <Field label="Passport Expiry Date" value={formData.applicant_passport_expiry} />
              <Field label="Phone Number" value={formData.applicant_phone} />
              <Field label="Email Address" value={formData.applicant_email} />
              <Field label="Current Address" value={formData.applicant_address} />
            </div>
          </div>
        </div>
      )}

      {/* IMM 5532 - Relationship */}
      {activeForm === 'imm5532' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-bold text-gray-800">IMM 5532 - Relationship Information</h2>
              <p className="text-sm text-gray-500">Spouse/Common-Law Partner Questionnaire</p>
            </div>
            <span className="bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm font-medium">🍁 IRCC</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            <div>
              <h3 className="font-semibold text-indigo-600 mb-3 mt-4">Marriage Details</h3>
              <Field label="Date of Marriage" value={formData.marriage_date} />
              <Field label="Place of Marriage" value={formData.marriage_location} />
              <Field label="Currently Living Together" value={formData.living_together} />
            </div>
            <div>
              <h3 className="font-semibold text-indigo-600 mb-3 mt-4">How You Met</h3>
              <Field label="Date You First Met" value={formData.first_met_date} />
              <Field label="Where You First Met" value={formData.first_met_location} />
              <Field label="Relationship Start Date" value={formData.relationship_start} />
            </div>
          </div>
        </div>
      )}

      {/* Print/Download Note */}
      <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-sm text-yellow-800">
          <strong>Note:</strong> Use the "Download Summary PDF" button in the Sponsorship Form Assistant tab to get a printable version of all forms.
        </p>
      </div>
    </div>
  )
}

export default App

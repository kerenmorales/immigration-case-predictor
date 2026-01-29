import { useState, useEffect } from 'react'
import { supabase } from './supabase'

const API_URL = import.meta.env.VITE_API_URL || 
  (window.location.hostname.includes('railway.app') 
    ? 'https://immigration-case-predictor-production.up.railway.app' 
    : 'http://localhost:8000')

function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('home')
  const [sponsorshipData, setSponsorshipData] = useState({})

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null)
      setLoading(false)
    })
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })
    return () => subscription.unsubscribe()
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="animate-pulse text-slate-400">Loading...</div>
      </div>
    )
  }

  if (!user) return <AuthPage />

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Professional Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-red-600 to-red-700 rounded-lg flex items-center justify-center">
              <span className="text-white text-lg">🍁</span>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-slate-800">ImmigrationAI</h1>
              <p className="text-xs text-slate-500">Legal Intelligence Platform</p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <span className="text-sm text-slate-600">{user.email}</span>
            <button onClick={() => supabase.auth.signOut()} className="text-sm text-slate-500 hover:text-slate-700">
              Sign Out
            </button>
          </div>
        </div>
        
        {/* Navigation */}
        <div className="max-w-6xl mx-auto px-6">
          <nav className="flex gap-1 overflow-x-auto">
            {[
              { id: 'home', label: 'Overview' },
              { id: 'eligibility', label: 'Eligibility Check' },
              { id: 'visaforms', label: 'Visa Forms' },
              { id: 'sponsorship', label: 'Sponsorship Forms' },
              { id: 'predictor', label: 'Case Predictor' },
              { id: 'history', label: 'My Cases' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id 
                    ? 'border-red-600 text-red-600' 
                    : 'border-transparent text-slate-600 hover:text-slate-800'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {activeTab === 'home' && <HomePage setActiveTab={setActiveTab} />}
        {activeTab === 'eligibility' && <EligibilityCheck setActiveTab={setActiveTab} />}
        {activeTab === 'visaforms' && <VisaForms user={user} />}
        {activeTab === 'predictor' && <CasePredictor user={user} />}
        {activeTab === 'sponsorship' && <SponsorshipAssistant formData={sponsorshipData} setFormData={setSponsorshipData} user={user} />}
        {activeTab === 'history' && <UserHistory user={user} />}
      </main>

      {/* Footer */}
      <footer className="bg-slate-800 text-slate-400 py-12 mt-16">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-white font-semibold mb-3">ImmigrationAI</h3>
              <p className="text-sm">AI-powered legal intelligence for Canadian immigration professionals.</p>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-3">Data Sources</h3>
              <p className="text-sm">Trained on 7,093 Federal Court decisions from the Refugee Law Lab dataset (1996-2022).</p>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-3">Disclaimer</h3>
              <p className="text-sm">This tool provides informational analysis only and does not constitute legal advice.</p>
            </div>
          </div>
          <div className="border-t border-slate-700 mt-8 pt-8 text-sm text-center">
            © 2026 ImmigrationAI. For professional use only.
          </div>
        </div>
      </footer>
    </div>
  )
}

function HomePage({ setActiveTab }) {
  return (
    <div>
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-12 text-white mb-12">
        <div className="max-w-3xl">
          <h1 className="text-4xl font-bold mb-4">AI-Powered Immigration Case Analysis</h1>
          <p className="text-xl text-slate-300 mb-8">
            Leverage machine learning trained on thousands of Federal Court decisions to gain insights into case outcomes and streamline your practice.
          </p>
          <div className="flex gap-4">
            <button onClick={() => setActiveTab('predictor')} className="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg font-medium transition-colors">
              Analyze a Case
            </button>
            <button onClick={() => setActiveTab('sponsorship')} className="bg-white/10 hover:bg-white/20 text-white px-6 py-3 rounded-lg font-medium transition-colors">
              Sponsorship Forms
            </button>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
        <div className="bg-white rounded-xl p-8 border border-slate-200">
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">⚖️</span>
          </div>
          <h2 className="text-xl font-semibold text-slate-800 mb-3">Case Outcome Predictor</h2>
          <p className="text-slate-600 mb-4">
            Our AI model analyzes case facts and predicts likely outcomes based on patterns from 7,093 Federal Court judicial review decisions.
          </p>
          <ul className="text-sm text-slate-500 space-y-2">
            <li>✓ Trained on real Federal Court decisions</li>
            <li>✓ Identifies key legal factors</li>
            <li>✓ Provides confidence scoring</li>
            <li>✓ Historical context comparison</li>
          </ul>
        </div>

        <div className="bg-white rounded-xl p-8 border border-slate-200">
          <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">📋</span>
          </div>
          <h2 className="text-xl font-semibold text-slate-800 mb-3">Sponsorship Form Assistant</h2>
          <p className="text-slate-600 mb-4">
            Streamline spousal sponsorship applications with our guided form wizard that helps you complete IMM 1344, IMM 0008, and IMM 5532.
          </p>
          <ul className="text-sm text-slate-500 space-y-2">
            <li>✓ Step-by-step guided process</li>
            <li>✓ IRCC-compliant field formats</li>
            <li>✓ PDF summary generation</li>
            <li>✓ Save and resume applications</li>
          </ul>
        </div>
      </div>

      {/* Model Methodology Section */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden mb-12">
        <div className="bg-slate-50 px-8 py-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-800">About Our AI Model</h2>
        </div>
        <div className="p-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-slate-800 mb-3">Training Data</h3>
              <p className="text-slate-600 text-sm mb-4">
                Our model was trained on the Refugee Law Lab dataset, containing 7,093 Federal Court of Canada judicial review decisions spanning from 1996 to 2022. This comprehensive dataset includes both allowed and dismissed cases across various claim types.
              </p>
              <div className="bg-slate-50 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-slate-500">Total Cases</p>
                    <p className="text-2xl font-semibold text-slate-800">7,093</p>
                  </div>
                  <div>
                    <p className="text-slate-500">Time Period</p>
                    <p className="text-2xl font-semibold text-slate-800">26 years</p>
                  </div>
                  <div>
                    <p className="text-slate-500">Allowed Rate</p>
                    <p className="text-2xl font-semibold text-green-600">30.1%</p>
                  </div>
                  <div>
                    <p className="text-slate-500">Dismissed Rate</p>
                    <p className="text-2xl font-semibold text-red-600">69.9%</p>
                  </div>
                </div>
              </div>
            </div>
            <div>
              <h3 className="font-semibold text-slate-800 mb-3">Model Architecture</h3>
              <p className="text-slate-600 text-sm mb-4">
                We use DistilBERT, a state-of-the-art transformer model optimized for text classification. The model was fine-tuned specifically on immigration case law to understand legal language and identify patterns that correlate with case outcomes.
              </p>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center text-blue-600 text-sm font-medium">1</div>
                  <p className="text-sm text-slate-600">Text tokenization and encoding</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center text-blue-600 text-sm font-medium">2</div>
                  <p className="text-sm text-slate-600">Transformer attention analysis</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center text-blue-600 text-sm font-medium">3</div>
                  <p className="text-sm text-slate-600">Binary classification (Allowed/Dismissed)</p>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center text-blue-600 text-sm font-medium">4</div>
                  <p className="text-sm text-slate-600">Confidence scoring and factor extraction</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Trust & Limitations */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-8">
        <h3 className="font-semibold text-amber-800 mb-3">Important Limitations</h3>
        <div className="text-sm text-amber-700 space-y-2">
          <p>• This tool is designed to assist legal professionals, not replace legal judgment.</p>
          <p>• Predictions are based on historical patterns and may not account for recent legal developments.</p>
          <p>• Each case has unique circumstances that may not be fully captured by text analysis.</p>
          <p>• Always conduct independent legal research and analysis for your clients.</p>
        </div>
      </div>
    </div>
  )
}

function EligibilityCheck({ setActiveTab }) {
  const [appType, setAppType] = useState(null)
  const [questions, setQuestions] = useState([])
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState({})
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedLanguage, setSelectedLanguage] = useState('en')

  const appTypes = [
    { id: 'visitor_visa', name: 'Visitor Visa', icon: '✈️', desc: 'Tourism, visiting family, or business' },
    { id: 'work_permit', name: 'Work Permit', icon: '💼', desc: 'Employment in Canada' },
    { id: 'super_visa', name: 'Super Visa', icon: '👨‍👩‍👧', desc: 'Parents & grandparents (up to 5 years)' }
  ]

  const startAssessment = async (type) => {
    setAppType(type)
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/eligibility/questions/${type}?lang=${selectedLanguage}`)
      const data = await response.json()
      setQuestions(data.questions)
      setCurrentQuestion(0)
      setAnswers({})
      setResult(null)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Refetch questions when language changes
  const handleLanguageChange = async (lang) => {
    const langCode = lang === 'English' ? 'en' : lang === 'Español (Spanish)' ? 'es' : 'fr'
    setSelectedLanguage(langCode)
    setAnswers(prev => ({ ...prev, language: lang }))
    
    if (appType) {
      setLoading(true)
      try {
        const response = await fetch(`${API_URL}/eligibility/questions/${appType}?lang=${langCode}`)
        const data = await response.json()
        setQuestions(data.questions)
      } catch (err) {
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
  }

  const handleAnswer = (questionId, value) => {
    // If this is the language question, also update the language
    if (questionId === 'language') {
      handleLanguageChange(value)
    } else {
      setAnswers(prev => ({ ...prev, [questionId]: value }))
    }
  }

  const nextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(prev => prev + 1)
    }
  }

  const prevQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1)
    }
  }

  const submitAssessment = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/eligibility/assess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ application_type: appType, answers })
      })
      const data = await response.json()
      setResult(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const resetAssessment = () => {
    setAppType(null)
    setQuestions([])
    setCurrentQuestion(0)
    setAnswers({})
    setResult(null)
  }

  // Application type selection
  if (!appType) {
    return (
      <div>
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-800 mb-3">Eligibility Pre-Assessment</h1>
          <p className="text-slate-600 max-w-2xl mx-auto">
            Answer a few questions to check if you meet the basic requirements for your Canadian immigration application.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {appTypes.map(type => (
            <button
              key={type.id}
              onClick={() => startAssessment(type.id)}
              className="bg-white rounded-xl border border-slate-200 p-8 text-left hover:border-red-300 hover:shadow-lg transition-all group"
            >
              <div className="text-4xl mb-4">{type.icon}</div>
              <h3 className="text-xl font-semibold text-slate-800 mb-2 group-hover:text-red-600">{type.name}</h3>
              <p className="text-slate-500 text-sm">{type.desc}</p>
            </button>
          ))}
        </div>
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
          <h3 className="font-semibold text-blue-800 mb-2">How it works</h3>
          <p className="text-sm text-blue-700">
            This tool asks you questions based on IRCC eligibility criteria. Based on your answers, 
            we'll tell you if you likely qualify, and if not, explain exactly why and what you can do about it.
          </p>
        </div>
      </div>
    )
  }

  // Show results
  if (result) {
    return (
      <div>
        <button onClick={resetAssessment} className="mb-6 text-slate-600 hover:text-slate-800 flex items-center gap-2">
          ← Start New Assessment
        </button>
        
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className={`px-8 py-6 ${
            result.eligibility === 'likely_eligible' ? 'bg-green-50 border-b border-green-200' :
            result.eligibility === 'possibly_eligible' ? 'bg-amber-50 border-b border-amber-200' :
            'bg-red-50 border-b border-red-200'
          }`}>
            <div className="flex items-center gap-4">
              <div className={`text-5xl ${
                result.eligibility === 'likely_eligible' ? '' :
                result.eligibility === 'possibly_eligible' ? '' : ''
              }`}>
                {result.eligibility === 'likely_eligible' ? '✅' :
                 result.eligibility === 'possibly_eligible' ? '⚠️' : '❌'}
              </div>
              <div>
                <h2 className={`text-2xl font-bold ${
                  result.eligibility === 'likely_eligible' ? 'text-green-800' :
                  result.eligibility === 'possibly_eligible' ? 'text-amber-800' :
                  'text-red-800'
                }`}>
                  {result.eligibility === 'likely_eligible' ? 'You Likely Qualify!' :
                   result.eligibility === 'possibly_eligible' ? 'You May Qualify' :
                   'Eligibility Concerns'}
                </h2>
                <p className={`text-sm ${
                  result.eligibility === 'likely_eligible' ? 'text-green-700' :
                  result.eligibility === 'possibly_eligible' ? 'text-amber-700' :
                  'text-red-700'
                }`}>
                  Eligibility Score: {result.score}/100
                </p>
              </div>
            </div>
          </div>
          
          <div className="p-8">
            <p className="text-slate-700 mb-6">{result.summary}</p>
            
            {result.issues.length > 0 && (
              <div className="mb-6">
                <h3 className="font-semibold text-red-800 mb-3">❌ Issues Found</h3>
                <div className="space-y-3">
                  {result.issues.map((issue, i) => (
                    <div key={i} className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-800">
                      {issue}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {result.warnings.length > 0 && (
              <div className="mb-6">
                <h3 className="font-semibold text-amber-800 mb-3">⚠️ Notes</h3>
                <div className="space-y-3">
                  {result.warnings.map((warning, i) => (
                    <div key={i} className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-800">
                      {warning}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {result.budget_estimate && (
              <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-6">
                <h3 className="font-semibold text-blue-800 mb-4">💰 Estimated Trip Budget ({result.budget_estimate.trip_days} days)</h3>
                <div className="space-y-2 mb-4">
                  {result.budget_estimate.breakdown.map((item, i) => (
                    <div key={i} className="flex justify-between text-sm">
                      <span className="text-blue-700">{item.item}</span>
                      <span className="font-medium text-blue-800">${item.amount.toLocaleString()} CAD</span>
                    </div>
                  ))}
                  <div className="border-t border-blue-300 pt-2 mt-2 flex justify-between">
                    <span className="font-semibold text-blue-800">Total Estimate</span>
                    <span className="font-bold text-blue-900 text-lg">${result.budget_estimate.total.toLocaleString()} CAD</span>
                  </div>
                </div>
                <p className="text-xs text-blue-600 mb-2">{result.budget_estimate.note}</p>
                <a 
                  href={result.budget_estimate.exchange_link} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-sm text-blue-700 hover:text-blue-900 underline"
                >
                  🔗 Check exchange rates to your currency →
                </a>
              </div>
            )}
            
            {result.income_requirement && (
              <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="font-semibold text-blue-800 mb-2">💰 Income Requirement (LICO+30%)</h3>
                <p className="text-blue-700">
                  Minimum required: <span className="font-bold">${result.income_requirement.toLocaleString()} CAD</span> per year
                </p>
              </div>
            )}
            
            {result.action_plan && result.action_plan.length > 0 && (
              <div>
                <h3 className="font-semibold text-slate-800 mb-3">📋 Your Action Plan</h3>
                <div className="space-y-3">
                  {result.action_plan.map((item, i) => (
                    <div key={i} className={`p-4 rounded-lg border ${
                      item.priority === 'high' ? 'bg-red-50 border-red-200' :
                      item.priority === 'medium' ? 'bg-amber-50 border-amber-200' :
                      'bg-slate-50 border-slate-200'
                    }`}>
                      <div className="flex items-start gap-3">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          item.priority === 'high' ? 'bg-red-200 text-red-800' :
                          item.priority === 'medium' ? 'bg-amber-200 text-amber-800' :
                          'bg-slate-200 text-slate-700'
                        }`}>
                          {item.priority === 'high' ? 'HIGH' : item.priority === 'medium' ? 'MEDIUM' : 'LOW'}
                        </span>
                        <p className="text-sm text-slate-700 whitespace-pre-line flex-1">{item.action}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Navigate to Forms Button */}
            {(appType === 'visitor_visa' || appType === 'super_visa') && result.eligibility !== 'unlikely_eligible' && (
              <div className="mt-8 pt-6 border-t border-slate-200">
                <div className="bg-gradient-to-r from-red-50 to-amber-50 border border-red-200 rounded-xl p-6">
                  <h3 className="font-semibold text-slate-800 mb-2">📝 Ready to Start Your Application?</h3>
                  <p className="text-sm text-slate-600 mb-4">
                    Use our guided form wizard to prepare your {appType === 'visitor_visa' ? 'Visitor Visa' : 'Super Visa'} application. 
                    We'll help you organize all the information you need.
                  </p>
                  <button
                    onClick={() => setActiveTab('visaforms')}
                    className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors"
                  >
                    Go to Visa Forms →
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  // Question flow
  const q = questions[currentQuestion]
  if (!q) return <div className="text-center py-12">Loading questions...</div>

  return (
    <div>
      <button onClick={resetAssessment} className="mb-6 text-slate-600 hover:text-slate-800 flex items-center gap-2">
        ← Choose Different Application
      </button>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Progress */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h3 className="font-semibold text-slate-800 mb-4">
              {appTypes.find(t => t.id === appType)?.name}
            </h3>
            <div className="space-y-2">
              {questions.map((question, i) => (
                <div
                  key={question.id}
                  className={`flex items-center gap-2 text-sm ${
                    i === currentQuestion ? 'text-red-600 font-medium' :
                    answers[question.id] !== undefined ? 'text-green-600' : 'text-slate-400'
                  }`}
                >
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                    i === currentQuestion ? 'bg-red-100 text-red-600' :
                    answers[question.id] !== undefined ? 'bg-green-100 text-green-600' : 'bg-slate-100'
                  }`}>
                    {answers[question.id] !== undefined ? '✓' : i + 1}
                  </div>
                  <span className="truncate">Q{i + 1}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 pt-4 border-t border-slate-200">
              <p className="text-xs text-slate-500">
                Question {currentQuestion + 1} of {questions.length}
              </p>
              <div className="mt-2 w-full bg-slate-200 rounded-full h-2">
                <div
                  className="bg-red-600 h-2 rounded-full transition-all"
                  style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Question */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
              <p className="text-sm text-slate-500">Question {currentQuestion + 1}</p>
            </div>
            <div className="p-8">
              <h2 className="text-xl font-semibold text-slate-800 mb-4">{q.question}</h2>
              
              {q.help && (
                <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-sm text-blue-800 whitespace-pre-line">{q.help}</p>
                </div>
              )}
              
              {q.type === 'text' && (
                <div>
                  <input
                    type="text"
                    value={answers[q.id] || ''}
                    onChange={(e) => handleAnswer(q.id, e.target.value)}
                    className="w-full border border-slate-300 rounded-lg px-4 py-3 text-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                    placeholder={q.id === 'country' ? 'e.g., India, Philippines, Nigeria...' : 'Enter your answer'}
                  />
                </div>
              )}
              
              {q.type === 'boolean' && (
                <div className="flex gap-4">
                  <button
                    onClick={() => handleAnswer(q.id, true)}
                    className={`flex-1 py-4 px-6 rounded-lg border-2 font-medium transition-all ${
                      answers[q.id] === true
                        ? 'border-green-500 bg-green-50 text-green-700'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => handleAnswer(q.id, false)}
                    className={`flex-1 py-4 px-6 rounded-lg border-2 font-medium transition-all ${
                      answers[q.id] === false
                        ? 'border-red-500 bg-red-50 text-red-700'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    No
                  </button>
                </div>
              )}
              
              {q.type === 'select' && (
                <div className="space-y-3">
                  {q.options.map(option => (
                    <button
                      key={option}
                      onClick={() => handleAnswer(q.id, option)}
                      className={`w-full py-3 px-4 rounded-lg border-2 text-left transition-all ${
                        answers[q.id] === option
                          ? 'border-red-500 bg-red-50 text-red-700'
                          : 'border-slate-200 hover:border-slate-300'
                      }`}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              )}
              
              {q.type === 'number' && (
                <div>
                  <input
                    type="number"
                    value={answers[q.id] || ''}
                    onChange={(e) => handleAnswer(q.id, parseInt(e.target.value) || 0)}
                    min={q.min}
                    max={q.max}
                    className="w-full border border-slate-300 rounded-lg px-4 py-3 text-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                    placeholder={q.id === 'host_income' ? 'e.g., 65000' : 'Enter number'}
                  />
                </div>
              )}

              <div className="flex justify-between mt-8 pt-6 border-t border-slate-200">
                <button
                  onClick={prevQuestion}
                  disabled={currentQuestion === 0}
                  className="px-6 py-2.5 border border-slate-300 rounded-lg text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                {currentQuestion < questions.length - 1 ? (
                  <button
                    onClick={nextQuestion}
                    disabled={answers[q.id] === undefined || (q.type === 'text' && !answers[q.id]?.trim())}
                    className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
                  >
                    Next Question
                  </button>
                ) : (
                  <button
                    onClick={submitAssessment}
                    disabled={loading || answers[q.id] === undefined || (q.type === 'text' && !answers[q.id]?.trim())}
                    className="px-6 py-2.5 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Analyzing...' : 'Get Results'}
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function CopyForIRCC({ formData, formType }) {
  const [copiedField, setCopiedField] = useState(null)
  const [copiedAll, setCopiedAll] = useState(false)

  const copyToClipboard = async (value, fieldName) => {
    if (!value) return
    try {
      await navigator.clipboard.writeText(value)
      setCopiedField(fieldName)
      setTimeout(() => setCopiedField(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const visitorVisaFields = [
    { section: 'Personal Information', fields: [
      { key: 'family_name', label: 'Family Name (Surname)' },
      { key: 'given_name', label: 'Given Name(s)' },
      { key: 'dob', label: 'Date of Birth' },
      { key: 'country', label: 'Country of Citizenship' },
      { key: 'passport_number', label: 'Passport Number' },
      { key: 'passport_expiry', label: 'Passport Expiry Date' },
      { key: 'email', label: 'Email Address' },
      { key: 'phone', label: 'Phone Number' },
      { key: 'address', label: 'Current Address' },
    ]},
    { section: 'Travel Details', fields: [
      { key: 'purpose', label: 'Purpose of Visit', transform: (v) => ({ tourism: 'Tourism/Vacation', family: 'Visiting family or friends', business: 'Business meetings', medical: 'Medical treatment', other: 'Other' }[v] || v) },
      { key: 'trip_duration', label: 'Duration of Stay', transform: (v) => v ? `${v} days` : '' },
      { key: 'arrival_date', label: 'Planned Arrival Date' },
      { key: 'departure_date', label: 'Planned Departure Date' },
      { key: 'accommodation', label: 'Accommodation' },
      { key: 'canada_contact', label: 'Contact in Canada' },
    ]},
    { section: 'Financial & Ties', fields: [
      { key: 'occupation', label: 'Current Occupation' },
      { key: 'employer', label: 'Employer Name' },
      { key: 'monthly_income', label: 'Monthly Income' },
      { key: 'savings', label: 'Savings/Bank Balance' },
      { key: 'ties', label: 'Ties to Home Country' },
      { key: 'travel_history', label: 'Previous Travel History' },
    ]},
  ]

  const superVisaFields = [
    { section: 'Visitor Information (Parent/Grandparent)', fields: [
      { key: 'visitor_family_name', label: 'Family Name (Surname)' },
      { key: 'visitor_given_name', label: 'Given Name(s)' },
      { key: 'visitor_dob', label: 'Date of Birth' },
      { key: 'visitor_country', label: 'Country of Citizenship' },
      { key: 'visitor_passport', label: 'Passport Number' },
      { key: 'visitor_passport_expiry', label: 'Passport Expiry Date' },
      { key: 'relationship', label: 'Relationship to Host', transform: (v) => v ? v.charAt(0).toUpperCase() + v.slice(1) : '' },
      { key: 'visitor_address', label: 'Current Address' },
    ]},
    { section: 'Host Information (Child/Grandchild in Canada)', fields: [
      { key: 'host_name', label: 'Host Full Name' },
      { key: 'host_status', label: 'Immigration Status', transform: (v) => ({ citizen: 'Canadian Citizen', pr: 'Permanent Resident' }[v] || v) },
      { key: 'host_phone', label: 'Host Phone Number' },
      { key: 'host_email', label: 'Host Email' },
      { key: 'host_address', label: 'Host Address in Canada' },
    ]},
    { section: 'Income & Insurance', fields: [
      { key: 'family_size', label: 'Number of People in Household' },
      { key: 'host_income', label: 'Annual Household Income', transform: (v) => v ? `$${Number(v).toLocaleString()} CAD` : '' },
      { key: 'insurance_provider', label: 'Medical Insurance Provider' },
      { key: 'insurance_amount', label: 'Insurance Coverage Amount' },
      { key: 'notes', label: 'Additional Notes' },
    ]},
  ]

  const fieldSections = formType === 'visitor_visa' ? visitorVisaFields : superVisaFields

  const copyAllFields = async () => {
    const lines = []
    fieldSections.forEach(section => {
      lines.push(`=== ${section.section} ===`)
      section.fields.forEach(field => {
        let value = formData[field.key] || ''
        if (field.transform && value) value = field.transform(value)
        if (value) lines.push(`${field.label}: ${value}`)
      })
      lines.push('')
    })
    
    try {
      await navigator.clipboard.writeText(lines.join('\n'))
      setCopiedAll(true)
      setTimeout(() => setCopiedAll(false), 3000)
    } catch (err) {
      console.error('Failed to copy all:', err)
    }
  }

  const hasData = Object.values(formData).some(v => v)

  if (!hasData) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="text-4xl mb-4">📋</div>
        <h3 className="text-lg font-semibold text-slate-800 mb-2">No Data to Copy</h3>
        <p className="text-slate-500">Fill out the form using the Chat Assistant or Form Wizard first, then come back here to copy your information for the IRCC portal.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-semibold text-blue-800 mb-2">📋 Copy Your Information for IRCC</h3>
            <p className="text-sm text-blue-700">
              Click the copy button next to each field to copy it to your clipboard, then paste into the IRCC portal.
              Open the IRCC website in another tab and copy fields one by one.
            </p>
          </div>
          <button
            onClick={copyAllFields}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              copiedAll 
                ? 'bg-green-600 text-white' 
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {copiedAll ? '✓ Copied All!' : '📋 Copy All Fields'}
          </button>
        </div>
      </div>

      {fieldSections.map((section, sIdx) => (
        <div key={sIdx} className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
            <h3 className="font-semibold text-slate-800">{section.section}</h3>
          </div>
          <div className="p-6">
            <div className="space-y-3">
              {section.fields.map((field, fIdx) => {
                let value = formData[field.key] || ''
                if (field.transform && value) value = field.transform(value)
                const isCopied = copiedField === field.key
                
                return (
                  <div key={fIdx} className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
                    <div className="flex-1">
                      <span className="text-sm text-slate-500">{field.label}</span>
                      <p className={`font-medium ${value ? 'text-slate-800' : 'text-slate-300'}`}>
                        {value || '—'}
                      </p>
                    </div>
                    {value && (
                      <button
                        onClick={() => copyToClipboard(value, field.key)}
                        className={`ml-4 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          isCopied 
                            ? 'bg-green-100 text-green-700' 
                            : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                        }`}
                      >
                        {isCopied ? '✓ Copied!' : '📋 Copy'}
                      </button>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      ))}

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
        <h3 className="font-semibold text-amber-800 mb-2">🔗 IRCC Portal Links</h3>
        <div className="space-y-2 text-sm">
          {formType === 'visitor_visa' ? (
            <>
              <a href="https://www.canada.ca/en/immigration-refugees-citizenship/services/visit-canada/apply-visitor-visa.html" target="_blank" rel="noopener noreferrer" className="block text-amber-700 hover:text-amber-900 underline">
                → Apply for Visitor Visa Online
              </a>
              <a href="https://www.canada.ca/en/immigration-refugees-citizenship/services/application/account.html" target="_blank" rel="noopener noreferrer" className="block text-amber-700 hover:text-amber-900 underline">
                → IRCC Online Account (Sign In/Create)
              </a>
            </>
          ) : (
            <>
              <a href="https://www.canada.ca/en/immigration-refugees-citizenship/services/visit-canada/parent-grandparent-super-visa/apply.html" target="_blank" rel="noopener noreferrer" className="block text-amber-700 hover:text-amber-900 underline">
                → Apply for Super Visa Online
              </a>
              <a href="https://www.canada.ca/en/immigration-refugees-citizenship/services/application/account.html" target="_blank" rel="noopener noreferrer" className="block text-amber-700 hover:text-amber-900 underline">
                → IRCC Online Account (Sign In/Create)
              </a>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function VisaForms({ user }) {
  const [formType, setFormType] = useState(null)
  const [step, setStep] = useState(1)
  const [formData, setFormData] = useState({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [activeView, setActiveView] = useState('wizard')
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)

  const formTypes = [
    { id: 'visitor_visa', name: 'Visitor Visa', icon: '✈️', desc: 'Tourism, visiting family, or business' },
    { id: 'super_visa', name: 'Super Visa', icon: '👨‍👩‍👧', desc: 'Parents & grandparents (up to 5 years)' }
  ]

  const updateField = (field, value) => setFormData(prev => ({ ...prev, [field]: value }))

  const startForm = (type) => {
    setFormType(type)
    setStep(1)
    setFormData({})
    setChatMessages([
      { role: 'assistant', content: type === 'visitor_visa' 
        ? 'Hello! I\'ll help you prepare your Visitor Visa application. Tell me about yourself and your trip.\n\nFor example:\n"My name is Maria Garcia, born January 5, 1990 in Mexico City. I want to visit my sister in Toronto for 2 weeks."'
        : 'Hello! I\'ll help you prepare your Super Visa application. Tell me about the parent/grandparent who wants to visit.\n\nFor example:\n"My mother is Rosa Martinez, born March 15, 1955 in Colombia. She wants to visit me in Vancouver."'
      }
    ])
  }

  const handleChatSubmit = async (e) => {
    e.preventDefault()
    if (!chatInput.trim()) return
    
    const userMessage = chatInput.trim()
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setChatInput('')
    setChatLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat-visa-form`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, form_type: formType, current_data: formData })
      })
      if (!response.ok) throw new Error('Failed to get response')
      const data = await response.json()
      
      if (data.extracted_fields && Object.keys(data.extracted_fields).length > 0) {
        setFormData(prev => ({ ...prev, ...data.extracted_fields }))
      }
      
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again or use the Form Wizard.' }])
    } finally {
      setChatLoading(false)
    }
  }

  const handleDownloadPDF = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/generate-visa-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ form_type: formType, ...formData })
      })
      if (!response.ok) throw new Error('Failed to generate PDF')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${formType}_application_summary.pdf`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setLoading(true)
    setError(null)
    try {
      await supabase.from('visa_forms').insert({ user_id: user.id, form_type: formType, form_data: formData, status: 'draft' })
      setSuccess('Form saved successfully!')
      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const FormField = ({ label, field, type = 'text', required = false, placeholder = '', options = null, help = null }) => (
    <div className="mb-4">
      <label className="block text-sm font-medium text-slate-700 mb-1.5">
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      {options ? (
        <select
          value={formData[field] || ''}
          onChange={(e) => updateField(field, e.target.value)}
          className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
        >
          <option value="">Select...</option>
          {options.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
        </select>
      ) : type === 'textarea' ? (
        <textarea
          value={formData[field] || ''}
          onChange={(e) => updateField(field, e.target.value)}
          rows={3}
          className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent resize-none"
          placeholder={placeholder}
        />
      ) : (
        <input
          type={type}
          value={formData[field] || ''}
          onChange={(e) => updateField(field, e.target.value)}
          className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
          placeholder={placeholder}
        />
      )}
      {help && <p className="mt-1 text-xs text-slate-500">{help}</p>}
    </div>
  )

  // Form type selection
  if (!formType) {
    return (
      <div>
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-800 mb-3">Visa Application Forms</h1>
          <p className="text-slate-600 max-w-2xl mx-auto">
            Use our guided form wizard to prepare your visa application. We'll help you organize all the information you need.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl mx-auto">
          {formTypes.map(type => (
            <button
              key={type.id}
              onClick={() => startForm(type.id)}
              className="bg-white rounded-xl border border-slate-200 p-8 text-left hover:border-red-300 hover:shadow-lg transition-all group"
            >
              <div className="text-4xl mb-4">{type.icon}</div>
              <h3 className="text-xl font-semibold text-slate-800 mb-2 group-hover:text-red-600">{type.name}</h3>
              <p className="text-slate-500 text-sm">{type.desc}</p>
            </button>
          ))}
        </div>
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6 max-w-3xl mx-auto">
          <h3 className="font-semibold text-blue-800 mb-2">💡 Tip</h3>
          <p className="text-sm text-blue-700">
            Not sure if you qualify? Use the <span className="font-medium">Eligibility Check</span> tab first to assess your eligibility before filling out forms.
          </p>
        </div>
      </div>
    )
  }

  const visitorSteps = [
    { num: 1, label: 'Personal Info' },
    { num: 2, label: 'Travel Details' },
    { num: 3, label: 'Ties & Finances' }
  ]

  const superVisaSteps = [
    { num: 1, label: 'Visitor Info' },
    { num: 2, label: 'Host Info' },
    { num: 3, label: 'Income & Insurance' }
  ]

  const steps = formType === 'visitor_visa' ? visitorSteps : superVisaSteps
  const totalSteps = steps.length

  const canProceed = () => {
    if (formType === 'visitor_visa') {
      if (step === 1) return formData.family_name && formData.given_name && formData.dob && formData.country
      if (step === 2) return formData.purpose && formData.trip_duration
      if (step === 3) return true
    } else {
      if (step === 1) return formData.visitor_family_name && formData.visitor_given_name && formData.visitor_dob && formData.visitor_country
      if (step === 2) return formData.host_name && formData.host_status && formData.host_address
      if (step === 3) return formData.host_income && formData.family_size
    }
    return true
  }

  return (
    <div>
      <button onClick={() => setFormType(null)} className="mb-6 text-slate-600 hover:text-slate-800 flex items-center gap-2">
        ← Choose Different Form
      </button>

      {/* View Toggle */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveView('chat')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'chat' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          💬 Chat Assistant
        </button>
        <button
          onClick={() => setActiveView('wizard')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'wizard' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          📝 Form Wizard
        </button>
        <button
          onClick={() => setActiveView('copy')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'copy' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          📋 Copy for IRCC
        </button>
      </div>

      {activeView === 'copy' ? (
        <CopyForIRCC formData={formData} formType={formType} />
      ) : activeView === 'chat' ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden flex flex-col" style={{ height: '600px' }}>
              <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
                <h2 className="text-lg font-semibold text-slate-800">
                  {formType === 'visitor_visa' ? 'Visitor Visa' : 'Super Visa'} Assistant
                </h2>
                <p className="text-sm text-slate-500">Tell me about your application in your own words</p>
              </div>
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] rounded-lg px-4 py-3 ${msg.role === 'user' ? 'bg-red-600 text-white' : 'bg-slate-100 text-slate-800'}`}>
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="bg-slate-100 rounded-lg px-4 py-3">
                      <p className="text-sm text-slate-500">Thinking...</p>
                    </div>
                  </div>
                )}
              </div>
              <form onSubmit={handleChatSubmit} className="p-4 border-t border-slate-200">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Tell me about yourself and your trip..."
                    className="flex-1 border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  />
                  <button
                    type="submit"
                    disabled={chatLoading || !chatInput.trim()}
                    className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
                  >
                    Send
                  </button>
                </div>
              </form>
            </div>
          </div>
          <div className="space-y-6">
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-800 mb-3">Example Inputs</h3>
              <div className="space-y-2">
                {(formType === 'visitor_visa' ? [
                  'My name is Maria Garcia, born Jan 5 1990 in Mexico',
                  'I want to visit my sister in Toronto for 2 weeks',
                  'I work as a teacher and earn $3000/month',
                  'I own a house and have 2 children in school'
                ] : [
                  'My mother Rosa Martinez, born March 15 1955 in Colombia',
                  'I am a Canadian citizen living in Vancouver',
                  'My household income is $85,000 per year',
                  'We are 4 people in the household'
                ]).map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setChatInput(q)}
                    className="w-full text-left text-sm text-slate-600 hover:text-red-600 hover:bg-slate-50 px-3 py-2 rounded-lg transition-colors"
                  >
                    "{q}"
                  </button>
                ))}
              </div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
              <h3 className="font-semibold text-amber-800 mb-2">IRCC Format</h3>
              <p className="text-sm text-amber-700">
                Names: Family name in CAPS, given names separate<br/>
                Dates: YYYY-MM-DD format<br/>
                Phone: +1 (XXX) XXX-XXXX
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Progress Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-800 mb-4">
                {formType === 'visitor_visa' ? '✈️ Visitor Visa' : '👨‍👩‍👧 Super Visa'}
              </h3>
              <div className="space-y-3">
                {steps.map(s => (
                  <div key={s.num} className={`flex items-center gap-3 p-3 rounded-lg ${step === s.num ? 'bg-red-50 border border-red-200' : step > s.num ? 'bg-green-50' : 'bg-slate-50'}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${step === s.num ? 'bg-red-600 text-white' : step > s.num ? 'bg-green-500 text-white' : 'bg-slate-300 text-slate-600'}`}>
                      {step > s.num ? '✓' : s.num}
                    </div>
                    <span className={`text-sm ${step === s.num ? 'text-red-700 font-medium' : 'text-slate-600'}`}>{s.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Form Content */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
              <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
                <h2 className="text-lg font-semibold text-slate-800">
                  {steps[step - 1]?.label}
                </h2>
                <p className="text-sm text-slate-500">Step {step} of {totalSteps}</p>
              </div>
              <div className="p-6">
                {formType === 'visitor_visa' ? (
                  <>
                    {step === 1 && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                        <FormField label="Family Name (Surname)" field="family_name" required placeholder="As shown on passport" />
                        <FormField label="Given Name(s)" field="given_name" required placeholder="First and middle names" />
                        <FormField label="Date of Birth" field="dob" type="date" required />
                        <FormField label="Country of Citizenship" field="country" required />
                        <FormField label="Passport Number" field="passport_number" placeholder="As shown on passport" />
                        <FormField label="Passport Expiry Date" field="passport_expiry" type="date" />
                        <FormField label="Email Address" field="email" type="email" />
                        <FormField label="Phone Number" field="phone" placeholder="+1 (XXX) XXX-XXXX" />
                        <div className="md:col-span-2">
                          <FormField label="Current Address" field="address" placeholder="Street, City, Country" />
                        </div>
                      </div>
                    )}
                    {step === 2 && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                        <FormField label="Purpose of Visit" field="purpose" required options={[
                          { value: 'tourism', label: 'Tourism/Vacation' },
                          { value: 'family', label: 'Visiting family or friends' },
                          { value: 'business', label: 'Business meetings' },
                          { value: 'medical', label: 'Medical treatment' },
                          { value: 'other', label: 'Other' }
                        ]} />
                        <FormField label="Duration of Stay (days)" field="trip_duration" type="number" required placeholder="e.g., 14" />
                        <FormField label="Planned Arrival Date" field="arrival_date" type="date" />
                        <FormField label="Planned Departure Date" field="departure_date" type="date" />
                        <div className="md:col-span-2">
                          <FormField label="Where will you stay?" field="accommodation" placeholder="Hotel name, or friend/family address" />
                        </div>
                        <div className="md:col-span-2">
                          <FormField label="Contact in Canada (if any)" field="canada_contact" placeholder="Name, relationship, phone number" />
                        </div>
                      </div>
                    )}
                    {step === 3 && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                        <FormField label="Current Occupation" field="occupation" placeholder="e.g., Teacher, Engineer, Business Owner" />
                        <FormField label="Employer Name" field="employer" placeholder="Company or organization name" />
                        <FormField label="Monthly Income" field="monthly_income" placeholder="In your local currency" />
                        <FormField label="Savings/Bank Balance" field="savings" placeholder="Approximate amount" />
                        <div className="md:col-span-2">
                          <FormField label="Ties to Home Country" field="ties" type="textarea" placeholder="Describe your reasons to return: job, property, family, business, etc." help="This is very important - explain why you will return home after your visit" />
                        </div>
                        <div className="md:col-span-2">
                          <FormField label="Previous Travel History" field="travel_history" type="textarea" placeholder="List countries you've visited (especially US, UK, Europe, Australia)" />
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    {step === 1 && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                        <div className="md:col-span-2 mb-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <p className="text-sm text-blue-800">Enter information about the parent or grandparent who wants to visit Canada.</p>
                        </div>
                        <FormField label="Family Name (Surname)" field="visitor_family_name" required placeholder="As shown on passport" />
                        <FormField label="Given Name(s)" field="visitor_given_name" required placeholder="First and middle names" />
                        <FormField label="Date of Birth" field="visitor_dob" type="date" required />
                        <FormField label="Country of Citizenship" field="visitor_country" required />
                        <FormField label="Passport Number" field="visitor_passport" placeholder="As shown on passport" />
                        <FormField label="Passport Expiry Date" field="visitor_passport_expiry" type="date" />
                        <FormField label="Relationship to Host" field="relationship" required options={[
                          { value: 'parent', label: 'Parent' },
                          { value: 'grandparent', label: 'Grandparent' }
                        ]} />
                        <div className="md:col-span-2">
                          <FormField label="Current Address" field="visitor_address" placeholder="Street, City, Country" />
                        </div>
                      </div>
                    )}
                    {step === 2 && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                        <div className="md:col-span-2 mb-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <p className="text-sm text-blue-800">Enter information about the child or grandchild in Canada who will host the visitor.</p>
                        </div>
                        <FormField label="Host's Full Name" field="host_name" required placeholder="Full legal name" />
                        <FormField label="Immigration Status" field="host_status" required options={[
                          { value: 'citizen', label: 'Canadian Citizen' },
                          { value: 'pr', label: 'Permanent Resident' }
                        ]} />
                        <FormField label="Host's Phone Number" field="host_phone" placeholder="+1 (XXX) XXX-XXXX" />
                        <FormField label="Host's Email" field="host_email" type="email" />
                        <div className="md:col-span-2">
                          <FormField label="Host's Address in Canada" field="host_address" required placeholder="Street, City, Province, Postal Code" />
                        </div>
                      </div>
                    )}
                    {step === 3 && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                        <FormField label="Number of People in Household" field="family_size" type="number" required placeholder="Including host, spouse, children" help="This determines the minimum income requirement" />
                        <FormField label="Annual Household Income (CAD)" field="host_income" type="number" required placeholder="Before taxes" help="Combined income of host and spouse" />
                        <div className="md:col-span-2 mb-4 bg-amber-50 border border-amber-200 rounded-lg p-4">
                          <p className="text-sm text-amber-800">
                            <strong>Income Requirement:</strong> The host must meet LICO+30% based on family size. 
                            For a family of 4, this is approximately $71,000 CAD/year.
                          </p>
                        </div>
                        <FormField label="Medical Insurance Provider" field="insurance_provider" placeholder="e.g., Manulife, Blue Cross, TuGo" />
                        <FormField label="Insurance Coverage Amount" field="insurance_amount" placeholder="Minimum $100,000 CAD" />
                        <div className="md:col-span-2">
                          <FormField label="Additional Notes" field="notes" type="textarea" placeholder="Any additional information about the application" />
                        </div>
                      </div>
                    )}
                  </>
                )}

                {error && <div className="mt-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>}
                {success && <div className="mt-4 p-4 bg-green-50 border border-green-200 text-green-700 rounded-lg text-sm">{success}</div>}

                <div className="flex justify-between mt-8 pt-6 border-t border-slate-200">
                  <button
                    onClick={() => setStep(s => Math.max(1, s - 1))}
                    disabled={step === 1}
                    className="px-6 py-2.5 border border-slate-300 rounded-lg text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <div className="flex gap-3">
                    <button onClick={handleSave} disabled={loading} className="px-6 py-2.5 border border-slate-300 rounded-lg text-slate-600 hover:bg-slate-50">
                      {loading ? 'Saving...' : 'Save Draft'}
                    </button>
                    {step < totalSteps ? (
                      <button
                        onClick={() => setStep(s => s + 1)}
                        disabled={!canProceed()}
                        className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
                      >
                        Continue
                      </button>
                    ) : (
                      <button
                        onClick={handleDownloadPDF}
                        disabled={loading || !canProceed()}
                        className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
                      >
                        {loading ? 'Generating...' : 'Download PDF Summary'}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
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
    <div className="min-h-screen bg-slate-100 flex">
      {/* Left Panel - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-slate-800 to-slate-900 p-12 flex-col justify-between">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <div className="w-12 h-12 bg-red-600 rounded-lg flex items-center justify-center">
              <span className="text-white text-2xl">🍁</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">ImmigrationAI</h1>
              <p className="text-slate-400 text-sm">Legal Intelligence Platform</p>
            </div>
          </div>
          <h2 className="text-4xl font-bold text-white mb-6">
            AI-Powered Immigration Case Analysis
          </h2>
          <p className="text-xl text-slate-300 mb-8">
            Leverage machine learning trained on 7,093 Federal Court decisions to gain insights into case outcomes.
          </p>
          <div className="space-y-4">
            <div className="flex items-center gap-3 text-slate-300">
              <span className="text-green-400">✓</span>
              <span>Case outcome prediction with confidence scoring</span>
            </div>
            <div className="flex items-center gap-3 text-slate-300">
              <span className="text-green-400">✓</span>
              <span>Key legal factor identification</span>
            </div>
            <div className="flex items-center gap-3 text-slate-300">
              <span className="text-green-400">✓</span>
              <span>Sponsorship form assistance</span>
            </div>
            <div className="flex items-center gap-3 text-slate-300">
              <span className="text-green-400">✓</span>
              <span>Secure case history storage</span>
            </div>
          </div>
        </div>
        <p className="text-slate-500 text-sm">
          Trusted by immigration professionals across Canada
        </p>
      </div>

      {/* Right Panel - Auth Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          <div className="lg:hidden mb-8 text-center">
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="w-10 h-10 bg-red-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-lg">🍁</span>
              </div>
              <h1 className="text-xl font-bold text-slate-800">ImmigrationAI</h1>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
            <h2 className="text-2xl font-semibold text-slate-800 mb-2">
              {isLogin ? 'Welcome back' : 'Create your account'}
            </h2>
            <p className="text-slate-500 mb-6">
              {isLogin ? 'Sign in to access your dashboard' : 'Start analyzing cases today'}
            </p>

            <form onSubmit={handleSubmit}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">Email address</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-shadow"
                  placeholder="you@lawfirm.com"
                  required
                />
              </div>
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-red-500 focus:border-transparent transition-shadow"
                  placeholder="••••••••"
                  required
                  minLength={6}
                />
              </div>

              {error && <div className="mb-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>}
              {message && <div className="mb-4 p-4 bg-green-50 border border-green-200 text-green-700 rounded-lg text-sm">{message}</div>}

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-red-600 hover:bg-red-700 text-white py-3 rounded-lg font-medium transition-colors disabled:bg-slate-300"
              >
                {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Create Account')}
              </button>
            </form>

            <p className="mt-6 text-center text-sm text-slate-500">
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <button onClick={() => setIsLogin(!isLogin)} className="text-red-600 font-medium hover:underline">
                {isLogin ? 'Sign up' : 'Sign in'}
              </button>
            </p>
          </div>

          <p className="mt-6 text-center text-xs text-slate-400">
            By signing up, you agree to our Terms of Service and Privacy Policy
          </p>
        </div>
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
        throw new Error(errData.detail || 'Analysis failed')
      }
      const result = await response.json()
      setPrediction(result)
      await supabase.from('predictions').insert({
        user_id: user.id, case_text: caseText, country_of_origin: country || null,
        claim_type: claimType || null, prediction: result.prediction,
        confidence: result.confidence, risk_level: result.risk_level, factors: result.factors
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Input Panel */}
      <div className="lg:col-span-2">
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
            <h2 className="text-lg font-semibold text-slate-800">Case Analysis</h2>
            <p className="text-sm text-slate-500">Enter case details for AI-powered outcome prediction</p>
          </div>
          <form onSubmit={handleSubmit} className="p-6">
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-700 mb-2">Case Facts & Description</label>
              <textarea
                value={caseText}
                onChange={(e) => setCaseText(e.target.value)}
                rows={8}
                className="w-full border border-slate-300 rounded-lg p-4 focus:ring-2 focus:ring-red-500 focus:border-transparent resize-none"
                placeholder="Describe the case facts, including details about the refugee claim, grounds for persecution, evidence presented, and any procedural history..."
                required
              />
              <p className="mt-2 text-xs text-slate-400">Minimum 50 characters. Include immigration-related terms for accurate analysis.</p>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Country of Origin</label>
                <input
                  type="text"
                  value={country}
                  onChange={(e) => setCountry(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  placeholder="e.g., Iran, China, Nigeria"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Claim Type</label>
                <select
                  value={claimType}
                  onChange={(e) => setClaimType(e.target.value)}
                  className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
                >
                  <option value="">Select type...</option>
                  <option value="political">Political persecution</option>
                  <option value="religious">Religious persecution</option>
                  <option value="gender">Gender-based persecution</option>
                  <option value="ethnic">Ethnic persecution</option>
                  <option value="sexual_orientation">Sexual orientation</option>
                </select>
              </div>
            </div>
            <button
              type="submit"
              disabled={loading || caseText.length < 50}
              className="w-full bg-red-600 hover:bg-red-700 text-white py-3 rounded-lg font-medium transition-colors disabled:bg-slate-300 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing Case...' : 'Analyze Case'}
            </button>
          </form>
        </div>
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">{error}</div>
        )}
      </div>

      {/* Info Panel */}
      <div className="space-y-6">
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h3 className="font-semibold text-slate-800 mb-3">How It Works</h3>
          <div className="space-y-3 text-sm text-slate-600">
            <p>1. Enter the case facts and relevant details</p>
            <p>2. Our AI analyzes patterns from 7,093 Federal Court decisions</p>
            <p>3. Receive a prediction with confidence score and key factors</p>
          </div>
        </div>
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
          <h3 className="font-semibold text-blue-800 mb-2">Model Accuracy</h3>
          <p className="text-sm text-blue-700">
            Our DistilBERT model was trained on real Federal Court judicial review decisions with validated outcomes.
          </p>
        </div>
      </div>

      {/* Results */}
      {prediction && (
        <div className="lg:col-span-3 bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex justify-between items-center">
            <div>
              <h2 className="text-lg font-semibold text-slate-800">Analysis Results</h2>
              <p className="text-sm text-slate-500">AI-generated prediction based on case patterns</p>
            </div>
            <div className={`px-4 py-2 rounded-full font-semibold ${
              prediction.prediction === 'Allowed' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            }`}>
              {prediction.prediction}
            </div>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="bg-slate-50 rounded-lg p-4">
                <p className="text-sm text-slate-500 mb-1">Confidence Level</p>
                <p className="text-2xl font-semibold text-slate-800">{(prediction.confidence * 100).toFixed(1)}%</p>
                <div className="mt-2 w-full bg-slate-200 rounded-full h-2">
                  <div className={`h-2 rounded-full ${prediction.prediction === 'Allowed' ? 'bg-green-500' : 'bg-red-500'}`} style={{ width: `${prediction.confidence * 100}%` }} />
                </div>
              </div>
              <div className="bg-slate-50 rounded-lg p-4">
                <p className="text-sm text-slate-500 mb-1">Risk Assessment</p>
                <p className={`text-2xl font-semibold ${
                  prediction.risk_level === 'High' ? 'text-blue-600' : 
                  prediction.risk_level === 'Medium' ? 'text-amber-600' : 'text-slate-600'
                }`}>{prediction.risk_level}</p>
                <p className="text-xs text-slate-500 mt-1">Confidence indicator</p>
              </div>
              <div className="bg-slate-50 rounded-lg p-4">
                <p className="text-sm text-slate-500 mb-1">Historical Baseline</p>
                <p className="text-2xl font-semibold text-slate-800">30.1%</p>
                <p className="text-xs text-slate-500 mt-1">Cases allowed historically</p>
              </div>
            </div>

            {prediction.risk_description && (
              <div className="bg-slate-50 rounded-lg p-4 mb-6">
                <p className="text-sm text-slate-700">{prediction.risk_description}</p>
              </div>
            )}

            {prediction.factors?.length > 0 && (
              <div className="mb-6">
                <h3 className="font-semibold text-slate-800 mb-3">Key Legal Factors Identified</h3>
                <div className="flex flex-wrap gap-2">
                  {prediction.factors.map((f, i) => (
                    <span key={i} className={`px-3 py-1.5 rounded-full text-sm font-medium ${
                      f.impact === 'positive' ? 'bg-green-100 text-green-700' :
                      f.impact === 'negative' ? 'bg-red-100 text-red-700' :
                      'bg-slate-100 text-slate-600'
                    }`}>
                      {f.factor}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {prediction.historical_context && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <p className="text-sm font-medium text-blue-800 mb-1">📊 Historical Context</p>
                <p className="text-sm text-blue-700">{prediction.historical_context}</p>
              </div>
            )}

            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <p className="text-sm text-amber-800">
                <strong>Disclaimer:</strong> This analysis is for informational purposes only and does not constitute legal advice. 
                Actual case outcomes depend on many factors not captured by this model. Always conduct independent legal research.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


function DocumentChecklist() {
  const [applicationType, setApplicationType] = useState('spouse_only')
  const [checkedItems, setCheckedItems] = useState({})

  const toggleItem = (id) => {
    setCheckedItems(prev => ({ ...prev, [id]: !prev[id] }))
  }

  const spouseOnlyDocs = [
    {
      category: 'Sponsor Documents (IMM 1344)',
      items: [
        { id: 's1', doc: 'IMM 1344 - Application to Sponsor', required: true },
        { id: 's2', doc: 'Proof of Canadian citizenship or PR status', required: true, help: 'Canadian passport, citizenship certificate, or PR card (both sides)' },
        { id: 's3', doc: 'Proof of income (if applicable)', required: false, help: 'Notice of Assessment (NOA), T4 slips, employment letter' },
        { id: 's4', doc: 'Copy of valid photo ID', required: true, help: "Driver's license, provincial ID, or passport photo page" },
      ]
    },
    {
      category: 'Principal Applicant Documents (IMM 0008)',
      items: [
        { id: 'a1', doc: 'IMM 0008 - Generic Application Form', required: true },
        { id: 'a2', doc: 'IMM 5669 - Schedule A (Background/Declaration)', required: true },
        { id: 'a3', doc: 'Valid passport (all pages with stamps/visas)', required: true },
        { id: 'a4', doc: 'Two passport-size photos', required: true, help: 'Must meet IRCC photo specifications' },
        { id: 'a5', doc: 'Police certificates', required: true, help: 'From every country lived in 6+ months since age 18' },
        { id: 'a6', doc: 'Medical exam results (IMM 1017)', required: true, help: 'Done by IRCC panel physician' },
        { id: 'a7', doc: 'Birth certificate', required: true },
        { id: 'a8', doc: 'Proof of language ability (if applicable)', required: false },
      ]
    },
    {
      category: 'Relationship Documents (IMM 5532)',
      items: [
        { id: 'r1', doc: 'IMM 5532 - Relationship Information and Sponsorship Evaluation', required: true },
        { id: 'r2', doc: 'Marriage certificate (certified copy)', required: true, help: 'Must be officially translated if not in English/French' },
        { id: 'r3', doc: 'Proof of relationship genuineness (10+ pages)', required: true, help: 'Photos, messages, call logs, travel records, etc.' },
        { id: 'r4', doc: 'Photos together (with dates and descriptions)', required: true, help: 'Include photos from different occasions: wedding, trips, family events' },
        { id: 'r5', doc: 'Communication evidence', required: true, help: 'Screenshots of texts, emails, video call logs, social media' },
        { id: 'r6', doc: 'Travel records showing visits', required: false, help: 'Flight tickets, boarding passes, passport stamps' },
        { id: 'r7', doc: 'Joint financial documents (if any)', required: false, help: 'Joint bank accounts, shared bills, money transfers' },
        { id: 'r8', doc: 'Letters from family/friends', required: false, help: 'Statutory declarations from people who know your relationship' },
      ]
    },
    {
      category: 'Additional Documents',
      items: [
        { id: 'x1', doc: 'IMM 5476 - Use of Representative (if using one)', required: false },
        { id: 'x2', doc: 'IMM 5409 - Statutory Declaration of Common-law Union', required: false, help: 'Only if common-law relationship' },
        { id: 'x3', doc: 'Divorce/annulment certificates (if previously married)', required: false },
        { id: 'x4', doc: 'Death certificate of former spouse (if widowed)', required: false },
      ]
    }
  ]

  const spouseWithDependentsDocs = [
    ...spouseOnlyDocs,
    {
      category: 'Dependent Children Documents',
      items: [
        { id: 'd1', doc: 'IMM 0008DEP - Additional Dependants/Declaration', required: true },
        { id: 'd2', doc: "Child's birth certificate", required: true, help: 'Showing both parents names' },
        { id: 'd3', doc: "Child's passport (all pages)", required: true },
        { id: 'd4', doc: 'Two passport-size photos per child', required: true },
        { id: 'd5', doc: "Child's police certificate (if 18+)", required: false },
        { id: 'd6', doc: "Child's medical exam results", required: true },
        { id: 'd7', doc: 'Custody documents (if applicable)', required: false, help: 'Court orders, custody agreements' },
        { id: 'd8', doc: 'Consent letter from other parent (if applicable)', required: false, help: 'If child has another legal parent not traveling' },
        { id: 'd9', doc: 'Adoption papers (if adopted)', required: false },
        { id: 'd10', doc: 'Proof of full-time student status (if 19-22)', required: false, help: 'School enrollment letter, transcripts' },
      ]
    }
  ]

  const docs = applicationType === 'spouse_only' ? spouseOnlyDocs : spouseWithDependentsDocs
  
  const totalItems = docs.reduce((acc, cat) => acc + cat.items.length, 0)
  const checkedCount = Object.values(checkedItems).filter(Boolean).length
  const progress = Math.round((checkedCount / totalItems) * 100)

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h2 className="text-xl font-semibold text-slate-800 mb-4">📋 Spousal Sponsorship Document Checklist</h2>
        <p className="text-slate-600 mb-4">Based on IMM 5532 requirements. Check off documents as you gather them.</p>
        
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setApplicationType('spouse_only')}
            className={`flex-1 py-3 px-4 rounded-lg border-2 font-medium transition-all ${
              applicationType === 'spouse_only'
                ? 'border-red-500 bg-red-50 text-red-700'
                : 'border-slate-200 hover:border-slate-300'
            }`}
          >
            👫 Spouse Only
          </button>
          <button
            onClick={() => setApplicationType('spouse_with_dependents')}
            className={`flex-1 py-3 px-4 rounded-lg border-2 font-medium transition-all ${
              applicationType === 'spouse_with_dependents'
                ? 'border-red-500 bg-red-50 text-red-700'
                : 'border-slate-200 hover:border-slate-300'
            }`}
          >
            👨‍👩‍👧 Spouse + Dependent Children
          </button>
        </div>

        <div className="bg-slate-50 rounded-lg p-4 mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-slate-700">Progress</span>
            <span className="text-sm text-slate-600">{checkedCount} of {totalItems} documents</span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-3">
            <div
              className="bg-green-500 h-3 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {docs.map((category, cIdx) => (
        <div key={cIdx} className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
            <h3 className="font-semibold text-slate-800">{category.category}</h3>
          </div>
          <div className="p-4">
            {category.items.map((item) => (
              <div
                key={item.id}
                onClick={() => toggleItem(item.id)}
                className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-colors ${
                  checkedItems[item.id] ? 'bg-green-50' : 'hover:bg-slate-50'
                }`}
              >
                <div className={`w-6 h-6 rounded border-2 flex items-center justify-center flex-shrink-0 mt-0.5 ${
                  checkedItems[item.id] 
                    ? 'bg-green-500 border-green-500 text-white' 
                    : 'border-slate-300'
                }`}>
                  {checkedItems[item.id] && '✓'}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className={`font-medium ${checkedItems[item.id] ? 'text-green-700 line-through' : 'text-slate-800'}`}>
                      {item.doc}
                    </span>
                    {item.required && (
                      <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded font-medium">Required</span>
                    )}
                  </div>
                  {item.help && (
                    <p className="text-sm text-slate-500 mt-1">{item.help}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
        <h3 className="font-semibold text-blue-800 mb-2">💡 Tips</h3>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• All documents not in English or French must be translated by a certified translator</li>
          <li>• Keep original documents - submit copies unless otherwise specified</li>
          <li>• Proof of relationship should be at least 10 pages - use the "Proof of Relationship" tab to organize</li>
          <li>• Photos should have dates and brief descriptions on the back</li>
        </ul>
      </div>
    </div>
  )
}

function ProofOfRelationship({ user }) {
  const [entries, setEntries] = useState([])
  const [newEntry, setNewEntry] = useState({ type: 'text_message', date: '', content: '', description: '' })
  const [loading, setLoading] = useState(false)

  const entryTypes = [
    { value: 'text_message', label: '💬 Text Message', icon: '💬' },
    { value: 'email', label: '📧 Email', icon: '📧' },
    { value: 'social_media', label: '📱 Social Media', icon: '📱' },
    { value: 'letter', label: '✉️ Letter', icon: '✉️' },
    { value: 'call_log', label: '📞 Call Log', icon: '📞' },
    { value: 'photo', label: '📷 Photo Description', icon: '📷' },
    { value: 'other', label: '📄 Other', icon: '📄' },
  ]

  const addEntry = () => {
    if (!newEntry.date || !newEntry.content) return
    setEntries(prev => [...prev, { ...newEntry, id: Date.now() }].sort((a, b) => new Date(a.date) - new Date(b.date)))
    setNewEntry({ type: 'text_message', date: '', content: '', description: '' })
  }

  const removeEntry = (id) => {
    setEntries(prev => prev.filter(e => e.id !== id))
  }

  // Estimate pages (roughly 3000 characters per page)
  const totalChars = entries.reduce((acc, e) => acc + e.content.length + (e.description?.length || 0) + 100, 0)
  const estimatedPages = Math.max(1, Math.ceil(totalChars / 3000))
  const pageProgress = Math.min(100, (estimatedPages / 10) * 100)

  const downloadPDF = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/generate-proof-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entries })
      })
      if (!response.ok) throw new Error('Failed to generate PDF')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'proof_of_relationship.pdf'
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      console.error(err)
      alert('Failed to generate PDF. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-pink-50 to-red-50 border border-pink-200 rounded-xl p-6">
        <h2 className="text-xl font-semibold text-slate-800 mb-2">💕 Proof of Relationship Organizer</h2>
        <p className="text-slate-600 mb-4">
          Organize your communication evidence for IMM 5532. IRCC recommends at least 10 pages of proof showing your relationship is genuine.
        </p>
        
        <div className="bg-white rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-slate-700">Estimated Pages</span>
            <span className={`text-sm font-medium ${estimatedPages >= 10 ? 'text-green-600' : 'text-amber-600'}`}>
              {estimatedPages} / 10 pages {estimatedPages >= 10 ? '✓' : '(need more)'}
            </span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all ${estimatedPages >= 10 ? 'bg-green-500' : 'bg-amber-500'}`}
              style={{ width: `${pageProgress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Add New Entry */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h3 className="font-semibold text-slate-800 mb-4">Add Communication Evidence</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Type</label>
            <select
              value={newEntry.type}
              onChange={(e) => setNewEntry(prev => ({ ...prev, type: e.target.value }))}
              className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
            >
              {entryTypes.map(t => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Date</label>
            <input
              type="date"
              value={newEntry.date}
              onChange={(e) => setNewEntry(prev => ({ ...prev, date: e.target.value }))}
              className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
            />
          </div>
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-1.5">
            Content (copy/paste the message, email, or description)
          </label>
          <textarea
            value={newEntry.content}
            onChange={(e) => setNewEntry(prev => ({ ...prev, content: e.target.value }))}
            rows={4}
            className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent resize-none"
            placeholder="Paste the text message, email content, or describe the communication..."
          />
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-1.5">
            Context/Description (optional)
          </label>
          <input
            type="text"
            value={newEntry.description}
            onChange={(e) => setNewEntry(prev => ({ ...prev, description: e.target.value }))}
            className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
            placeholder="e.g., 'Planning our wedding', 'Daily good morning texts', 'Discussing moving to Canada'"
          />
        </div>
        
        <button
          onClick={addEntry}
          disabled={!newEntry.date || !newEntry.content}
          className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
        >
          + Add Entry
        </button>
      </div>

      {/* Entries List */}
      {entries.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex justify-between items-center">
            <div>
              <h3 className="font-semibold text-slate-800">Your Evidence ({entries.length} entries)</h3>
              <p className="text-sm text-slate-500">Sorted by date</p>
            </div>
            <button
              onClick={downloadPDF}
              disabled={loading || entries.length === 0}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
            >
              {loading ? 'Generating...' : '📥 Download PDF'}
            </button>
          </div>
          <div className="divide-y divide-slate-100">
            {entries.map((entry) => {
              const typeInfo = entryTypes.find(t => t.value === entry.type)
              return (
                <div key={entry.id} className="p-4 hover:bg-slate-50">
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xl">{typeInfo?.icon}</span>
                      <span className="font-medium text-slate-800">{typeInfo?.label}</span>
                      <span className="text-sm text-slate-500">{entry.date}</span>
                    </div>
                    <button
                      onClick={() => removeEntry(entry.id)}
                      className="text-red-500 hover:text-red-700 text-sm"
                    >
                      Remove
                    </button>
                  </div>
                  {entry.description && (
                    <p className="text-sm text-slate-600 italic mb-2">{entry.description}</p>
                  )}
                  <p className="text-sm text-slate-700 whitespace-pre-wrap bg-slate-50 p-3 rounded-lg">
                    {entry.content.length > 300 ? entry.content.substring(0, 300) + '...' : entry.content}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {entries.length === 0 && (
        <div className="bg-slate-50 rounded-xl border border-slate-200 p-12 text-center">
          <div className="text-4xl mb-4">💌</div>
          <h3 className="text-lg font-semibold text-slate-800 mb-2">No Evidence Added Yet</h3>
          <p className="text-slate-500 max-w-md mx-auto">
            Start adding your communication evidence above. Include text messages, emails, social media conversations, 
            letters, and call logs that show your relationship is genuine.
          </p>
        </div>
      )}

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
        <h3 className="font-semibold text-amber-800 mb-2">💡 What to Include</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-amber-700">
          <div>
            <p className="font-medium mb-1">Good Evidence:</p>
            <ul className="space-y-1">
              <li>✓ Daily/regular communication</li>
              <li>✓ Planning future together</li>
              <li>✓ Discussing important life events</li>
              <li>✓ Pet names and intimate language</li>
              <li>✓ Travel planning and visits</li>
            </ul>
          </div>
          <div>
            <p className="font-medium mb-1">Tips:</p>
            <ul className="space-y-1">
              <li>• Include dates for all evidence</li>
              <li>• Show communication over time (not just recent)</li>
              <li>• Variety is good - mix texts, calls, emails</li>
              <li>• Quality over quantity - meaningful exchanges</li>
              <li>• Translate non-English content</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

function PhotoAlbumOrganizer({ user }) {
  const [photos, setPhotos] = useState({})
  const [loading, setLoading] = useState(false)

  const categories = [
    { id: 'first_meeting', title: '💕 How We First Met', description: 'Photos from when you first met or early dating', slots: 3 },
    { id: 'dating', title: '🌹 Our Dating Journey', description: 'Photos from dates, outings, and spending time together', slots: 3 },
    { id: 'engagement', title: '💍 Our Engagement', description: 'Engagement photos, proposal, ring photos', slots: 2 },
    { id: 'wedding', title: '👰 Our Wedding Day', description: 'Wedding ceremony, reception, with family and friends', slots: 4 },
    { id: 'cultural', title: '🎊 Cultural & Religious Celebrations', description: 'Traditional ceremonies, religious events, cultural celebrations', slots: 2 },
    { id: 'family', title: '👨‍👩‍👧 With Our Families', description: 'Photos with both families, family gatherings, holidays', slots: 3 },
    { id: 'travel', title: '✈️ Our Travels Together', description: 'Vacations, trips, visits to each other', slots: 2 },
    { id: 'everyday', title: '🏠 Our Everyday Life', description: 'Daily life together, home, activities', slots: 1 },
  ]

  const totalSlots = categories.reduce((acc, cat) => acc + cat.slots, 0)
  const filledSlots = Object.values(photos).filter(p => p && p.image).length
  const progress = Math.round((filledSlots / totalSlots) * 100)

  const handleImagePaste = async (categoryId, slotIndex, e) => {
    const items = e.clipboardData?.items
    if (!items) return

    for (let item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        const reader = new FileReader()
        reader.onload = (event) => {
          const key = `${categoryId}_${slotIndex}`
          setPhotos(prev => ({
            ...prev,
            [key]: {
              ...prev[key],
              image: event.target.result,
              fileName: file.name || 'pasted_image.png'
            }
          }))
        }
        reader.readAsDataURL(file)
        break
      }
    }
  }

  const handleFileSelect = (categoryId, slotIndex, e) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      const key = `${categoryId}_${slotIndex}`
      setPhotos(prev => ({
        ...prev,
        [key]: {
          ...prev[key],
          image: event.target.result,
          fileName: file.name
        }
      }))
    }
    reader.readAsDataURL(file)
  }

  const updateDescription = (categoryId, slotIndex, field, value) => {
    const key = `${categoryId}_${slotIndex}`
    setPhotos(prev => ({
      ...prev,
      [key]: {
        ...prev[key],
        [field]: value
      }
    }))
  }

  const removePhoto = (categoryId, slotIndex) => {
    const key = `${categoryId}_${slotIndex}`
    setPhotos(prev => {
      const newPhotos = { ...prev }
      delete newPhotos[key]
      return newPhotos
    })
  }

  const downloadPDF = async () => {
    setLoading(true)
    try {
      // Prepare photos data for PDF
      const photoData = categories.map(cat => ({
        category: cat.title,
        photos: Array.from({ length: cat.slots }, (_, i) => {
          const key = `${cat.id}_${i}`
          const photo = photos[key]
          return photo ? {
            image: photo.image,
            date: photo.date || '',
            location: photo.location || '',
            description: photo.description || ''
          } : null
        }).filter(Boolean)
      })).filter(cat => cat.photos.length > 0)

      const response = await fetch(`${API_URL}/generate-photo-album-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ categories: photoData })
      })
      
      if (!response.ok) throw new Error('Failed to generate PDF')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'relationship_photo_album.pdf'
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      console.error(err)
      alert('Failed to generate PDF. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-xl p-6">
        <h2 className="text-xl font-semibold text-slate-800 mb-2">📷 Relationship Photo Album</h2>
        <p className="text-slate-600 mb-4">
          Organize 20 photographs that tell the story of your relationship. IRCC recommends photos from different occasions 
          with dates and descriptions on the back.
        </p>
        
        <div className="bg-white rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-slate-700">Photos Added</span>
            <span className={`text-sm font-medium ${filledSlots >= 20 ? 'text-green-600' : 'text-amber-600'}`}>
              {filledSlots} / {totalSlots} photos {filledSlots >= 20 ? '✓' : ''}
            </span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all ${filledSlots >= 20 ? 'bg-green-500' : 'bg-purple-500'}`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {filledSlots > 0 && (
          <button
            onClick={downloadPDF}
            disabled={loading}
            className="px-6 py-2.5 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
          >
            {loading ? 'Generating PDF...' : '📥 Download Photo Album PDF'}
          </button>
        )}
      </div>

      {categories.map((category) => (
        <div key={category.id} className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
            <h3 className="font-semibold text-slate-800">{category.title}</h3>
            <p className="text-sm text-slate-500">{category.description}</p>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 gap-6">
              {Array.from({ length: category.slots }, (_, slotIndex) => {
                const key = `${category.id}_${slotIndex}`
                const photo = photos[key]
                
                return (
                  <div key={slotIndex} className="flex gap-4 p-4 bg-slate-50 rounded-lg">
                    {/* Photo Area */}
                    <div className="w-48 flex-shrink-0">
                      {photo?.image ? (
                        <div className="relative">
                          <img 
                            src={photo.image} 
                            alt={`Photo ${slotIndex + 1}`}
                            className="w-48 h-36 object-cover rounded-lg border border-slate-200"
                          />
                          <button
                            onClick={() => removePhoto(category.id, slotIndex)}
                            className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-sm hover:bg-red-600"
                          >
                            ×
                          </button>
                        </div>
                      ) : (
                        <div
                          className="w-48 h-36 border-2 border-dashed border-slate-300 rounded-lg flex flex-col items-center justify-center cursor-pointer hover:border-purple-400 hover:bg-purple-50 transition-colors"
                          onPaste={(e) => handleImagePaste(category.id, slotIndex, e)}
                          tabIndex={0}
                        >
                          <span className="text-3xl mb-2">📷</span>
                          <span className="text-xs text-slate-500 text-center px-2">
                            Click & paste image<br/>or use button below
                          </span>
                          <input
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleFileSelect(category.id, slotIndex, e)}
                            className="hidden"
                            id={`file-${key}`}
                          />
                          <label
                            htmlFor={`file-${key}`}
                            className="mt-2 px-3 py-1 bg-slate-200 hover:bg-slate-300 rounded text-xs cursor-pointer"
                          >
                            Browse
                          </label>
                        </div>
                      )}
                    </div>
                    
                    {/* Description Area */}
                    <div className="flex-1 space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-xs font-medium text-slate-600 mb-1">Date</label>
                          <input
                            type="date"
                            value={photo?.date || ''}
                            onChange={(e) => updateDescription(category.id, slotIndex, 'date', e.target.value)}
                            className="w-full border border-slate-300 rounded px-3 py-1.5 text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-slate-600 mb-1">Location</label>
                          <input
                            type="text"
                            value={photo?.location || ''}
                            onChange={(e) => updateDescription(category.id, slotIndex, 'location', e.target.value)}
                            placeholder="City, Country"
                            className="w-full border border-slate-300 rounded px-3 py-1.5 text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          />
                        </div>
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-slate-600 mb-1">Description</label>
                        <textarea
                          value={photo?.description || ''}
                          onChange={(e) => updateDescription(category.id, slotIndex, 'description', e.target.value)}
                          placeholder="Describe this photo: who is in it, what was the occasion, why is it meaningful..."
                          rows={2}
                          className="w-full border border-slate-300 rounded px-3 py-1.5 text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                        />
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      ))}

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
        <h3 className="font-semibold text-amber-800 mb-2">💡 Photo Tips for IRCC</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-amber-700">
          <div>
            <p className="font-medium mb-1">What to Include:</p>
            <ul className="space-y-1">
              <li>✓ Photos from different time periods</li>
              <li>✓ Both of you clearly visible</li>
              <li>✓ Photos with family members</li>
              <li>✓ Special occasions (wedding, holidays)</li>
              <li>✓ Everyday moments together</li>
            </ul>
          </div>
          <div>
            <p className="font-medium mb-1">Best Practices:</p>
            <ul className="space-y-1">
              <li>• Include dates on all photos</li>
              <li>• Write brief descriptions</li>
              <li>• Show variety of occasions</li>
              <li>• Include photos with timestamps</li>
              <li>• Quality over quantity</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

function CopyForIRCCSponsorship({ formData }) {
  const [copiedField, setCopiedField] = useState(null)
  const [copiedAll, setCopiedAll] = useState(false)

  const copyToClipboard = async (value, fieldName) => {
    if (!value) return
    try {
      await navigator.clipboard.writeText(value)
      setCopiedField(fieldName)
      setTimeout(() => setCopiedField(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const fieldSections = [
    { section: 'IMM 1344 - Sponsor Information', fields: [
      { key: 'sponsor_family_name', label: 'Family Name (Surname)' },
      { key: 'sponsor_given_name', label: 'Given Name(s)' },
      { key: 'sponsor_dob', label: 'Date of Birth' },
      { key: 'sponsor_country_birth', label: 'Country of Birth' },
      { key: 'sponsor_citizenship', label: 'Citizenship Status' },
      { key: 'sponsor_address', label: 'Current Address' },
      { key: 'sponsor_email', label: 'Email Address' },
      { key: 'sponsor_phone', label: 'Phone Number' },
    ]},
    { section: 'IMM 0008 - Applicant Information', fields: [
      { key: 'applicant_family_name', label: 'Family Name (Surname)' },
      { key: 'applicant_given_name', label: 'Given Name(s)' },
      { key: 'applicant_dob', label: 'Date of Birth' },
      { key: 'applicant_country_birth', label: 'Country of Birth' },
      { key: 'applicant_citizenship', label: 'Country of Citizenship' },
      { key: 'applicant_residence', label: 'Country of Residence' },
      { key: 'applicant_address', label: 'Current Address' },
      { key: 'applicant_passport', label: 'Passport Number' },
    ]},
    { section: 'IMM 5532 - Relationship Information', fields: [
      { key: 'relationship_type', label: 'Relationship Type', transform: (v) => ({ spouse: 'Spouse', common_law: 'Common-law Partner', conjugal: 'Conjugal Partner' }[v] || v) },
      { key: 'date_married', label: 'Date of Marriage/Union' },
      { key: 'place_married', label: 'Place of Marriage' },
      { key: 'how_met', label: 'How You Met' },
      { key: 'relationship_history', label: 'Relationship History' },
    ]},
  ]

  const copyAllFields = async () => {
    const lines = []
    fieldSections.forEach(section => {
      lines.push(`=== ${section.section} ===`)
      section.fields.forEach(field => {
        let value = formData[field.key] || ''
        if (field.transform && value) value = field.transform(value)
        if (value) lines.push(`${field.label}: ${value}`)
      })
      lines.push('')
    })
    
    try {
      await navigator.clipboard.writeText(lines.join('\n'))
      setCopiedAll(true)
      setTimeout(() => setCopiedAll(false), 3000)
    } catch (err) {
      console.error('Failed to copy all:', err)
    }
  }

  const hasData = Object.values(formData).some(v => v)

  if (!hasData) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="text-4xl mb-4">📋</div>
        <h3 className="text-lg font-semibold text-slate-800 mb-2">No Data to Copy</h3>
        <p className="text-slate-500">Fill out the form using the Chat Assistant or Form Wizard first, then come back here to copy your information for the IRCC portal.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-semibold text-blue-800 mb-2">📋 Copy Your Information for IRCC</h3>
            <p className="text-sm text-blue-700">
              Click the copy button next to each field to copy it to your clipboard, then paste into the IRCC portal.
              Open the IRCC website in another tab and copy fields one by one.
            </p>
          </div>
          <button
            onClick={copyAllFields}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              copiedAll 
                ? 'bg-green-600 text-white' 
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {copiedAll ? '✓ Copied All!' : '📋 Copy All Fields'}
          </button>
        </div>
      </div>

      {fieldSections.map((section, sIdx) => (
        <div key={sIdx} className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
            <h3 className="font-semibold text-slate-800">{section.section}</h3>
          </div>
          <div className="p-6">
            <div className="space-y-3">
              {section.fields.map((field, fIdx) => {
                let value = formData[field.key] || ''
                if (field.transform && value) value = field.transform(value)
                const isCopied = copiedField === field.key
                
                return (
                  <div key={fIdx} className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
                    <div className="flex-1">
                      <span className="text-sm text-slate-500">{field.label}</span>
                      <p className={`font-medium ${value ? 'text-slate-800' : 'text-slate-300'}`}>
                        {value || '—'}
                      </p>
                    </div>
                    {value && (
                      <button
                        onClick={() => copyToClipboard(value, field.key)}
                        className={`ml-4 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          isCopied 
                            ? 'bg-green-100 text-green-700' 
                            : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                        }`}
                      >
                        {isCopied ? '✓ Copied!' : '📋 Copy'}
                      </button>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      ))}

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
        <h3 className="font-semibold text-amber-800 mb-2">🔗 IRCC Portal Links</h3>
        <div className="space-y-2 text-sm">
          <a href="https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/family-sponsorship/spouse-partner-children.html" target="_blank" rel="noopener noreferrer" className="block text-amber-700 hover:text-amber-900 underline">
            → Spousal Sponsorship Overview
          </a>
          <a href="https://www.canada.ca/en/immigration-refugees-citizenship/services/application/account.html" target="_blank" rel="noopener noreferrer" className="block text-amber-700 hover:text-amber-900 underline">
            → IRCC Online Account (Sign In/Create)
          </a>
          <a href="https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/family-sponsorship/spouse-partner-children/apply.html" target="_blank" rel="noopener noreferrer" className="block text-amber-700 hover:text-amber-900 underline">
            → How to Apply for Spousal Sponsorship
          </a>
        </div>
      </div>
    </div>
  )
}

function SponsorshipAssistant({ formData, setFormData, user }) {
  const [step, setStep] = useState(1)
  const [activeView, setActiveView] = useState('wizard')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'ll help you fill out your sponsorship forms. Just tell me the information naturally and I\'ll format it correctly for IRCC.\n\nFor example, say:\n"The sponsor is John Michael Smith, born March 15, 1985 in Toronto, Canada"\n\nI\'ll extract and format: Family Name: SMITH, Given Name(s): John Michael, DOB: 1985-03-15' }
  ])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)

  const updateField = (field, value) => setFormData(prev => ({ ...prev, [field]: value }))

  const handleChatSubmit = async (e) => {
    e.preventDefault()
    if (!chatInput.trim()) return
    
    const userMessage = chatInput.trim()
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setChatInput('')
    setChatLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat-form-fill`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, current_data: formData })
      })
      if (!response.ok) throw new Error('Failed to get response')
      const data = await response.json()
      
      // Update form fields if extracted
      if (data.extracted_fields && Object.keys(data.extracted_fields).length > 0) {
        setFormData(prev => ({ ...prev, ...data.extracted_fields }))
      }
      
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }])
    } finally {
      setChatLoading(false)
    }
  }

  const canProceed = () => {
    if (step === 1) {
      return formData.sponsor_family_name && formData.sponsor_given_name && formData.sponsor_dob && formData.sponsor_country_birth
    }
    if (step === 2) {
      return formData.applicant_family_name && formData.applicant_given_name && formData.applicant_dob && formData.applicant_country_birth && formData.applicant_citizenship
    }
    if (step === 3) {
      return formData.relationship_type && formData.date_married
    }
    return true
  }

  const handleSave = async () => {
    setLoading(true)
    setError(null)
    try {
      await supabase.from('sponsorship_forms').insert({ user_id: user.id, form_data: formData, status: 'draft' })
      setSuccess('Form saved successfully!')
      setTimeout(() => setSuccess(null), 3000)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadPDF = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/generate-pdf-summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      if (!response.ok) throw new Error('Failed to generate PDF')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'sponsorship_summary.pdf'
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const FormField = ({ label, field, type = 'text', required = false, placeholder = '', options = null }) => (
    <div className="mb-4">
      <label className="block text-sm font-medium text-slate-700 mb-1.5">
        {label} {required && <span className="text-red-500">*</span>}
      </label>
      {options ? (
        <select
          value={formData[field] || ''}
          onChange={(e) => updateField(field, e.target.value)}
          className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
        >
          <option value="">Select...</option>
          {options.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
        </select>
      ) : (
        <input
          type={type}
          value={formData[field] || ''}
          onChange={(e) => updateField(field, e.target.value)}
          className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
          placeholder={placeholder}
        />
      )}
    </div>
  )

  return (
    <div>
      {/* View Toggle */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={() => setActiveView('chat')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'chat' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          💬 Chat Assistant
        </button>
        <button
          onClick={() => setActiveView('wizard')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'wizard' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          📝 Form Wizard
        </button>
        <button
          onClick={() => setActiveView('checklist')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'checklist' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          ✅ Document Checklist
        </button>
        <button
          onClick={() => setActiveView('proof')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'proof' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          � Communication Evidence
        </button>
        <button
          onClick={() => setActiveView('photos')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'photos' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          📷 Photo Album
        </button>
        <button
          onClick={() => setActiveView('copy')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'copy' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          📋 Copy for IRCC
        </button>
        <button
          onClick={() => setActiveView('reports')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'reports' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          Form Reports
        </button>
      </div>

      {activeView === 'checklist' ? (
        <DocumentChecklist />
      ) : activeView === 'proof' ? (
        <ProofOfRelationship user={user} />
      ) : activeView === 'photos' ? (
        <PhotoAlbumOrganizer user={user} />
      ) : activeView === 'copy' ? (
        <CopyForIRCCSponsorship formData={formData} />
      ) : activeView === 'chat' ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden flex flex-col" style={{ height: '600px' }}>
              <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
                <h2 className="text-lg font-semibold text-slate-800">Sponsorship Assistant</h2>
                <p className="text-sm text-slate-500">Ask questions about spousal sponsorship</p>
              </div>
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] rounded-lg px-4 py-3 ${msg.role === 'user' ? 'bg-red-600 text-white' : 'bg-slate-100 text-slate-800'}`}>
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="bg-slate-100 rounded-lg px-4 py-3">
                      <p className="text-sm text-slate-500">Thinking...</p>
                    </div>
                  </div>
                )}
              </div>
              <form onSubmit={handleChatSubmit} className="p-4 border-t border-slate-200">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask about sponsorship requirements, documents, timelines..."
                    className="flex-1 border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent"
                  />
                  <button
                    type="submit"
                    disabled={chatLoading || !chatInput.trim()}
                    className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
                  >
                    Send
                  </button>
                </div>
              </form>
            </div>
          </div>
          <div className="space-y-6">
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-800 mb-3">Example Inputs</h3>
              <div className="space-y-2">
                {[
                  'Sponsor: Maria Garcia, born Jan 5 1990 in Mexico City',
                  'Applicant is Ahmed Hassan, DOB December 12, 1988, from Egypt',
                  'We got married on June 15, 2023 in Vancouver',
                  'Sponsor email: maria@email.com, phone 416-555-1234'
                ].map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setChatInput(q)}
                    className="w-full text-left text-sm text-slate-600 hover:text-red-600 hover:bg-slate-50 px-3 py-2 rounded-lg transition-colors"
                  >
                    "{q}"
                  </button>
                ))}
              </div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
              <h3 className="font-semibold text-amber-800 mb-2">IRCC Format</h3>
              <p className="text-sm text-amber-700">
                Names: Family name in CAPS, given names separate<br/>
                Dates: YYYY-MM-DD format<br/>
                Phone: +1 (XXX) XXX-XXXX
              </p>
            </div>
          </div>
        </div>
      ) : activeView === 'wizard' ? (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Progress Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-800 mb-4">Progress</h3>
              <div className="space-y-3">
                {[
                  { num: 1, label: 'Sponsor Info' },
                  { num: 2, label: 'Applicant Info' },
                  { num: 3, label: 'Relationship' }
                ].map(s => (
                  <div key={s.num} className={`flex items-center gap-3 p-3 rounded-lg ${step === s.num ? 'bg-red-50 border border-red-200' : step > s.num ? 'bg-green-50' : 'bg-slate-50'}`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${step === s.num ? 'bg-red-600 text-white' : step > s.num ? 'bg-green-500 text-white' : 'bg-slate-300 text-slate-600'}`}>
                      {step > s.num ? '✓' : s.num}
                    </div>
                    <span className={`text-sm ${step === s.num ? 'text-red-700 font-medium' : 'text-slate-600'}`}>{s.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Form Content */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
              <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
                <h2 className="text-lg font-semibold text-slate-800">
                  {step === 1 && 'Sponsor Information'}
                  {step === 2 && 'Principal Applicant Information'}
                  {step === 3 && 'Relationship Details'}
                </h2>
                <p className="text-sm text-slate-500">Step {step} of 3</p>
              </div>
              <div className="p-6">
                {step === 1 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                    <FormField label="Family Name (Surname)" field="sponsor_family_name" required placeholder="As shown on passport" />
                    <FormField label="Given Name(s)" field="sponsor_given_name" required placeholder="First and middle names" />
                    <FormField label="Date of Birth" field="sponsor_dob" type="date" required />
                    <FormField label="Country of Birth" field="sponsor_country_birth" required />
                    <FormField label="Country of Citizenship" field="sponsor_citizenship" />
                    <FormField label="Current Mailing Address" field="sponsor_address" placeholder="Street, City, Province, Postal Code" />
                    <FormField label="Email Address" field="sponsor_email" type="email" />
                    <FormField label="Phone Number" field="sponsor_phone" placeholder="+1 (XXX) XXX-XXXX" />
                  </div>
                )}
                {step === 2 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                    <FormField label="Family Name (Surname)" field="applicant_family_name" required placeholder="As shown on passport" />
                    <FormField label="Given Name(s)" field="applicant_given_name" required placeholder="First and middle names" />
                    <FormField label="Date of Birth" field="applicant_dob" type="date" required />
                    <FormField label="Country of Birth" field="applicant_country_birth" required />
                    <FormField label="Country of Citizenship" field="applicant_citizenship" required />
                    <FormField label="Current Country of Residence" field="applicant_residence" />
                    <FormField label="Current Mailing Address" field="applicant_address" placeholder="Full address in country of residence" />
                    <FormField label="Passport Number" field="applicant_passport" />
                  </div>
                )}
                {step === 3 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
                    <FormField label="Relationship Type" field="relationship_type" required options={[
                      { value: 'spouse', label: 'Spouse' },
                      { value: 'common_law', label: 'Common-law Partner' },
                      { value: 'conjugal', label: 'Conjugal Partner' }
                    ]} />
                    <FormField label="Date of Marriage/Union" field="date_married" type="date" required />
                    <FormField label="Place of Marriage" field="place_married" placeholder="City, Country" />
                    <FormField label="How did you meet?" field="how_met" />
                    <div className="md:col-span-2">
                      <label className="block text-sm font-medium text-slate-700 mb-1.5">Relationship History</label>
                      <textarea
                        value={formData.relationship_history || ''}
                        onChange={(e) => updateField('relationship_history', e.target.value)}
                        rows={4}
                        className="w-full border border-slate-300 rounded-lg p-4 focus:ring-2 focus:ring-red-500 focus:border-transparent resize-none"
                        placeholder="Describe how your relationship developed..."
                      />
                    </div>
                  </div>
                )}

                {error && <div className="mt-4 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>}
                {success && <div className="mt-4 p-4 bg-green-50 border border-green-200 text-green-700 rounded-lg text-sm">{success}</div>}

                <div className="flex justify-between mt-8 pt-6 border-t border-slate-200">
                  <button
                    onClick={() => setStep(s => Math.max(1, s - 1))}
                    disabled={step === 1}
                    className="px-6 py-2.5 border border-slate-300 rounded-lg text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <div className="flex gap-3">
                    <button onClick={handleSave} disabled={loading} className="px-6 py-2.5 border border-slate-300 rounded-lg text-slate-600 hover:bg-slate-50">
                      {loading ? 'Saving...' : 'Save Draft'}
                    </button>
                    {step < 3 ? (
                      <button
                        onClick={() => setStep(s => s + 1)}
                        disabled={!canProceed()}
                        className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
                      >
                        Continue
                      </button>
                    ) : (
                      <button
                        onClick={handleDownloadPDF}
                        disabled={loading || !canProceed()}
                        className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:bg-slate-300 disabled:cursor-not-allowed"
                      >
                        {loading ? 'Generating...' : 'Download PDF Summary'}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <FormReports formData={formData} />
      )}
    </div>
  )
}

function FormReports({ formData }) {
  const [activeForm, setActiveForm] = useState('IMM1344')

  const renderIMM1344 = () => (
    <div className="space-y-6">
      <div className="bg-slate-50 rounded-lg p-4">
        <h4 className="font-medium text-slate-800 mb-3">IMM 1344 - Application to Sponsor</h4>
        <p className="text-sm text-slate-600 mb-4">Sponsor eligibility and undertaking form</p>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div><span className="text-slate-500">Family Name:</span> <span className="font-medium">{formData.sponsor_family_name || '—'}</span></div>
        <div><span className="text-slate-500">Given Name(s):</span> <span className="font-medium">{formData.sponsor_given_name || '—'}</span></div>
        <div><span className="text-slate-500">Date of Birth:</span> <span className="font-medium">{formData.sponsor_dob || '—'}</span></div>
        <div><span className="text-slate-500">Country of Birth:</span> <span className="font-medium">{formData.sponsor_country_birth || '—'}</span></div>
        <div><span className="text-slate-500">Citizenship:</span> <span className="font-medium">{formData.sponsor_citizenship || '—'}</span></div>
        <div><span className="text-slate-500">Address:</span> <span className="font-medium">{formData.sponsor_address || '—'}</span></div>
        <div><span className="text-slate-500">Email:</span> <span className="font-medium">{formData.sponsor_email || '—'}</span></div>
        <div><span className="text-slate-500">Phone:</span> <span className="font-medium">{formData.sponsor_phone || '—'}</span></div>
      </div>
    </div>
  )

  const renderIMM0008 = () => (
    <div className="space-y-6">
      <div className="bg-slate-50 rounded-lg p-4">
        <h4 className="font-medium text-slate-800 mb-3">IMM 0008 - Generic Application Form</h4>
        <p className="text-sm text-slate-600 mb-4">Principal applicant information</p>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div><span className="text-slate-500">Family Name:</span> <span className="font-medium">{formData.applicant_family_name || '—'}</span></div>
        <div><span className="text-slate-500">Given Name(s):</span> <span className="font-medium">{formData.applicant_given_name || '—'}</span></div>
        <div><span className="text-slate-500">Date of Birth:</span> <span className="font-medium">{formData.applicant_dob || '—'}</span></div>
        <div><span className="text-slate-500">Country of Birth:</span> <span className="font-medium">{formData.applicant_country_birth || '—'}</span></div>
        <div><span className="text-slate-500">Citizenship:</span> <span className="font-medium">{formData.applicant_citizenship || '—'}</span></div>
        <div><span className="text-slate-500">Country of Residence:</span> <span className="font-medium">{formData.applicant_residence || '—'}</span></div>
        <div><span className="text-slate-500">Address:</span> <span className="font-medium">{formData.applicant_address || '—'}</span></div>
        <div><span className="text-slate-500">Passport Number:</span> <span className="font-medium">{formData.applicant_passport || '—'}</span></div>
      </div>
    </div>
  )

  const renderIMM5532 = () => (
    <div className="space-y-6">
      <div className="bg-slate-50 rounded-lg p-4">
        <h4 className="font-medium text-slate-800 mb-3">IMM 5532 - Relationship Information</h4>
        <p className="text-sm text-slate-600 mb-4">Details about the relationship between sponsor and applicant</p>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div><span className="text-slate-500">Relationship Type:</span> <span className="font-medium capitalize">{formData.relationship_type?.replace('_', ' ') || '—'}</span></div>
        <div><span className="text-slate-500">Date of Marriage/Union:</span> <span className="font-medium">{formData.date_married || '—'}</span></div>
        <div><span className="text-slate-500">Place of Marriage:</span> <span className="font-medium">{formData.place_married || '—'}</span></div>
        <div><span className="text-slate-500">How You Met:</span> <span className="font-medium">{formData.how_met || '—'}</span></div>
      </div>
      {formData.relationship_history && (
        <div className="mt-4">
          <span className="text-slate-500 text-sm">Relationship History:</span>
          <p className="mt-1 text-sm bg-slate-50 p-3 rounded-lg">{formData.relationship_history}</p>
        </div>
      )}
    </div>
  )

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
        <h2 className="text-lg font-semibold text-slate-800">Form Reports</h2>
        <p className="text-sm text-slate-500">View your data organized by IRCC form</p>
      </div>
      <div className="border-b border-slate-200">
        <div className="flex">
          {['IMM1344', 'IMM0008', 'IMM5532'].map(form => (
            <button
              key={form}
              onClick={() => setActiveForm(form)}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${activeForm === form ? 'border-red-600 text-red-600 bg-white' : 'border-transparent text-slate-600 hover:text-slate-800'}`}
            >
              {form}
            </button>
          ))}
        </div>
      </div>
      <div className="p-6">
        {activeForm === 'IMM1344' && renderIMM1344()}
        {activeForm === 'IMM0008' && renderIMM0008()}
        {activeForm === 'IMM5532' && renderIMM5532()}
      </div>
    </div>
  )
}

function UserHistory({ user }) {
  const [predictions, setPredictions] = useState([])
  const [forms, setForms] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('predictions')

  useEffect(() => {
    const fetchData = async () => {
      const [predRes, formRes] = await Promise.all([
        supabase.from('predictions').select('*').eq('user_id', user.id).order('created_at', { ascending: false }),
        supabase.from('sponsorship_forms').select('*').eq('user_id', user.id).order('created_at', { ascending: false })
      ])
      setPredictions(predRes.data || [])
      setForms(formRes.data || [])
      setLoading(false)
    }
    fetchData()
  }, [user.id])

  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
        <div className="animate-pulse text-slate-400">Loading your history...</div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="bg-slate-50 px-6 py-4 border-b border-slate-200">
        <h2 className="text-lg font-semibold text-slate-800">My Cases</h2>
        <p className="text-sm text-slate-500">View your saved predictions and sponsorship forms</p>
      </div>
      <div className="border-b border-slate-200">
        <div className="flex">
          <button
            onClick={() => setActiveTab('predictions')}
            className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'predictions' ? 'border-red-600 text-red-600' : 'border-transparent text-slate-600 hover:text-slate-800'}`}
          >
            Predictions ({predictions.length})
          </button>
          <button
            onClick={() => setActiveTab('forms')}
            className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'forms' ? 'border-red-600 text-red-600' : 'border-transparent text-slate-600 hover:text-slate-800'}`}
          >
            Sponsorship Forms ({forms.length})
          </button>
        </div>
      </div>
      <div className="p-6">
        {activeTab === 'predictions' && (
          predictions.length === 0 ? (
            <p className="text-slate-500 text-center py-8">No predictions yet. Analyze a case to get started.</p>
          ) : (
            <div className="space-y-4">
              {predictions.map(p => (
                <div key={p.id} className="border border-slate-200 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${p.prediction === 'Allowed' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {p.prediction}
                      </span>
                      <span className="ml-3 text-sm text-slate-500">{(p.confidence * 100).toFixed(1)}% confidence</span>
                    </div>
                    <span className="text-xs text-slate-400">{new Date(p.created_at).toLocaleDateString()}</span>
                  </div>
                  <p className="text-sm text-slate-600 line-clamp-2">{p.case_text}</p>
                  {p.country_of_origin && <p className="text-xs text-slate-500 mt-2">Country: {p.country_of_origin}</p>}
                </div>
              ))}
            </div>
          )
        )}
        {activeTab === 'forms' && (
          forms.length === 0 ? (
            <p className="text-slate-500 text-center py-8">No saved forms yet. Start a sponsorship application to get started.</p>
          ) : (
            <div className="space-y-4">
              {forms.map(f => (
                <div key={f.id} className="border border-slate-200 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <span className="font-medium text-slate-800">
                        {f.form_data?.sponsor_family_name}, {f.form_data?.sponsor_given_name}
                      </span>
                      <span className="ml-3 text-sm text-slate-500">sponsoring</span>
                      <span className="ml-1 font-medium text-slate-800">
                        {f.form_data?.applicant_family_name}, {f.form_data?.applicant_given_name}
                      </span>
                    </div>
                    <span className="text-xs text-slate-400">{new Date(f.created_at).toLocaleDateString()}</span>
                  </div>
                  <div className="flex gap-4 text-sm text-slate-500">
                    <span>Type: {f.form_data?.relationship_type?.replace('_', ' ') || '—'}</span>
                    <span>Status: {f.status}</span>
                  </div>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  )
}

export default App

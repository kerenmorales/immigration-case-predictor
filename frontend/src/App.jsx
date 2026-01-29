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
          <nav className="flex gap-1">
            {[
              { id: 'home', label: 'Overview' },
              { id: 'eligibility', label: 'Eligibility Check' },
              { id: 'predictor', label: 'Case Predictor' },
              { id: 'sponsorship', label: 'Sponsorship Forms' },
              { id: 'history', label: 'My Cases' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
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
        {activeTab === 'eligibility' && <EligibilityCheck />}
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

function EligibilityCheck() {
  const [appType, setAppType] = useState(null)
  const [questions, setQuestions] = useState([])
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState({})
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const appTypes = [
    { id: 'visitor_visa', name: 'Visitor Visa', icon: '✈️', desc: 'Tourism, visiting family, or business' },
    { id: 'work_permit', name: 'Work Permit', icon: '💼', desc: 'Employment in Canada' },
    { id: 'super_visa', name: 'Super Visa', icon: '👨‍👩‍👧', desc: 'Parents & grandparents (up to 5 years)' }
  ]

  const startAssessment = async (type) => {
    setAppType(type)
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/eligibility/questions/${type}`)
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

  const handleAnswer = (questionId, value) => {
    setAnswers(prev => ({ ...prev, [questionId]: value }))
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
                <h3 className="font-semibold text-slate-800 mb-3">� Your Action Plan</h3>
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
          Form Wizard
        </button>
        <button
          onClick={() => setActiveView('reports')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${activeView === 'reports' ? 'bg-red-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
        >
          Form Reports
        </button>
      </div>

      {activeView === 'chat' ? (
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

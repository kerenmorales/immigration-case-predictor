import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Tab navigation
function App() {
  const [activeTab, setActiveTab] = useState('predictor')

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-6">
        <div className="max-w-5xl mx-auto px-4">
          <h1 className="text-2xl font-bold">🇨🇦 Immigration Law Assistant</h1>
          <p className="text-indigo-100 mt-1">AI-powered tools for immigration lawyers</p>
        </div>
      </header>

      {/* Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-5xl mx-auto px-4">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('predictor')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'predictor'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Case Outcome Predictor
            </button>
            <button
              onClick={() => setActiveTab('sponsorship')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'sponsorship'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Sponsorship Form Assistant
            </button>
          </nav>
        </div>
      </div>

      {/* Content */}
      <main className="max-w-5xl mx-auto py-8 px-4">
        {activeTab === 'predictor' ? <CasePredictor /> : <SponsorshipAssistant />}
      </main>
    </div>
  )
}

// Case Outcome Predictor Component
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
        body: JSON.stringify({
          text: caseText,
          country_of_origin: country || null,
          claim_type: claimType || null
        })
      })
      
      if (!response.ok) throw new Error('Prediction failed')
      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getOutcomeColor = (outcome) => {
    if (outcome === 'Allowed') return 'text-green-600 bg-green-50'
    if (outcome === 'Dismissed') return 'text-red-600 bg-red-50'
    return 'text-yellow-600 bg-yellow-50'
  }

  return (
    <div>
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Predict Case Outcome</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Case Description / Facts
            </label>
            <textarea
              value={caseText}
              onChange={(e) => setCaseText(e.target.value)}
              rows={6}
              className="w-full border border-gray-300 rounded-md p-3 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              placeholder="Describe the case facts, grounds for the claim, evidence..."
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Country of Origin
              </label>
              <input
                type="text"
                value={country}
                onChange={(e) => setCountry(e.target.value)}
                className="w-full border border-gray-300 rounded-md p-2"
                placeholder="e.g., Iran, Nigeria"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Claim Type
              </label>
              <select
                value={claimType}
                onChange={(e) => setClaimType(e.target.value)}
                className="w-full border border-gray-300 rounded-md p-2"
              >
                <option value="">Select type...</option>
                <option value="political">Political persecution</option>
                <option value="religious">Religious persecution</option>
                <option value="gender">Gender-based persecution</option>
                <option value="ethnic">Ethnic persecution</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !caseText.trim()}
            className="w-full bg-indigo-600 text-white py-3 rounded-md font-medium hover:bg-indigo-700 disabled:bg-gray-400"
          >
            {loading ? 'Analyzing...' : 'Predict Outcome'}
          </button>
        </form>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md mb-8">
          {error}
        </div>
      )}

      {prediction && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
          
          <div className={`inline-block px-4 py-2 rounded-full font-semibold text-lg mb-6 ${getOutcomeColor(prediction.prediction)}`}>
            {prediction.prediction}
          </div>
          
          <div className="mb-6">
            <p className="text-sm font-medium text-gray-700 mb-2">Confidence</p>
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div 
                className="bg-indigo-600 h-4 rounded-full"
                style={{ width: `${prediction.confidence * 100}%` }}
              />
            </div>
            <p className="text-sm text-gray-600 mt-1">{(prediction.confidence * 100).toFixed(1)}%</p>
          </div>

          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-md">
            <p className="text-sm text-yellow-800">
              <strong>Disclaimer:</strong> This is for informational purposes only and should not replace professional legal advice.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

// Sponsorship Form Assistant Component
function SponsorshipAssistant() {
  const [started, setStarted] = useState(false)
  const [messages, setMessages] = useState([])
  const [currentQuestion, setCurrentQuestion] = useState(null)
  const [input, setInput] = useState('')
  const [responses, setResponses] = useState({})
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [completed, setCompleted] = useState(false)
  const [showSummary, setShowSummary] = useState(false)

  const QUESTIONS = [
    { section: 'sponsor', id: 'sponsor_full_name', text: "What is the sponsor's full legal name?" },
    { section: 'sponsor', id: 'sponsor_dob', text: "What is the sponsor's date of birth?" },
    { section: 'sponsor', id: 'sponsor_citizenship', text: "What is the sponsor's citizenship status? (Canadian Citizen/Permanent Resident)" },
    { section: 'sponsor', id: 'sponsor_address', text: "What is the sponsor's current residential address?" },
    { section: 'sponsor', id: 'sponsor_phone', text: "What is the sponsor's phone number?" },
    { section: 'sponsor', id: 'sponsor_email', text: "What is the sponsor's email address?" },
    { section: 'applicant', id: 'applicant_full_name', text: "What is the applicant's (spouse) full legal name?" },
    { section: 'applicant', id: 'applicant_dob', text: "What is the applicant's date of birth?" },
    { section: 'applicant', id: 'applicant_citizenship', text: "What is the applicant's country of citizenship?" },
    { section: 'applicant', id: 'applicant_passport', text: "What is the applicant's passport number?" },
    { section: 'applicant', id: 'applicant_address', text: "What is the applicant's current address?" },
    { section: 'applicant', id: 'applicant_phone', text: "What is the applicant's phone number?" },
    { section: 'applicant', id: 'applicant_email', text: "What is the applicant's email address?" },
    { section: 'relationship', id: 'marriage_date', text: "When did you get married?" },
    { section: 'relationship', id: 'marriage_location', text: "Where did you get married? (City, Country)" },
    { section: 'relationship', id: 'first_met_date', text: "When did you first meet?" },
    { section: 'relationship', id: 'first_met_location', text: "Where did you first meet?" },
    { section: 'relationship', id: 'relationship_start', text: "When did your relationship begin?" },
    { section: 'relationship', id: 'living_together', text: "Are you currently living together? (Yes/No)" },
  ]

  const startChat = () => {
    setStarted(true)
    setMessages([{ type: 'bot', text: "Hi! I'll help you collect information for your IRCC spousal sponsorship application. Let's start with the sponsor information." }])
    setCurrentQuestion(QUESTIONS[0])
    setProgress({ current: 0, total: QUESTIONS.length })
    setTimeout(() => {
      setMessages(prev => [...prev, { type: 'bot', text: QUESTIONS[0].text }])
    }, 500)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || !currentQuestion) return

    const answer = input.trim()
    setMessages(prev => [...prev, { type: 'user', text: answer }])
    setResponses(prev => ({ ...prev, [currentQuestion.id]: answer }))
    setInput('')

    const currentIndex = QUESTIONS.findIndex(q => q.id === currentQuestion.id)
    const nextIndex = currentIndex + 1

    if (nextIndex >= QUESTIONS.length) {
      setCompleted(true)
      setProgress({ current: QUESTIONS.length, total: QUESTIONS.length })
      setTimeout(() => {
        setMessages(prev => [...prev, { type: 'bot', text: "Thank you! I've collected all the information. You can now view the summary or download your data." }])
      }, 500)
    } else {
      const nextQuestion = QUESTIONS[nextIndex]
      setCurrentQuestion(nextQuestion)
      setProgress({ current: nextIndex, total: QUESTIONS.length })

      // Add section header if changing sections
      const currentSection = QUESTIONS[currentIndex].section
      const nextSection = nextQuestion.section
      
      setTimeout(() => {
        if (currentSection !== nextSection) {
          const sectionNames = { applicant: "Now let's collect the applicant's information.", relationship: "Finally, let's talk about your relationship." }
          if (sectionNames[nextSection]) {
            setMessages(prev => [...prev, { type: 'section', text: sectionNames[nextSection] }])
          }
        }
        setMessages(prev => [...prev, { type: 'bot', text: nextQuestion.text }])
      }, 500)
    }
  }

  const downloadData = () => {
    const data = JSON.stringify({ timestamp: new Date().toISOString(), responses }, null, 2)
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'sponsorship_data.json'
    a.click()
  }

  const startOver = () => {
    setStarted(false)
    setMessages([])
    setCurrentQuestion(null)
    setResponses({})
    setProgress({ current: 0, total: 0 })
    setCompleted(false)
    setShowSummary(false)
  }

  if (!started) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8 text-center">
        <h2 className="text-2xl font-bold mb-4">🇨🇦 Spousal Sponsorship Assistant</h2>
        <p className="text-gray-600 mb-6">I'll guide you through collecting all the information needed for your IRCC spousal sponsorship application.</p>
        <ul className="text-left inline-block mb-6 text-gray-600">
          <li className="mb-2">• Sponsor information (IMM 1344)</li>
          <li className="mb-2">• Applicant information (IMM 0008)</li>
          <li className="mb-2">• Relationship details (IMM 5532)</li>
        </ul>
        <div>
          <button onClick={startChat} className="bg-indigo-600 text-white px-8 py-3 rounded-full font-medium hover:bg-indigo-700">
            Start Application
          </button>
        </div>
      </div>
    )
  }

  if (showSummary) {
    const sections = {
      'Sponsor Information': ['sponsor_full_name', 'sponsor_dob', 'sponsor_citizenship', 'sponsor_address', 'sponsor_phone', 'sponsor_email'],
      'Applicant Information': ['applicant_full_name', 'applicant_dob', 'applicant_citizenship', 'applicant_passport', 'applicant_address', 'applicant_phone', 'applicant_email'],
      'Relationship Information': ['marriage_date', 'marriage_location', 'first_met_date', 'first_met_location', 'relationship_start', 'living_together']
    }

    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-6">Application Summary</h2>
        {Object.entries(sections).map(([sectionName, fields]) => (
          <div key={sectionName} className="mb-6">
            <h3 className="text-lg font-semibold text-indigo-600 mb-3 pb-2 border-b">{sectionName}</h3>
            {fields.map(field => (
              <div key={field} className="flex py-2 border-b border-gray-100">
                <span className="font-medium text-gray-600 w-48">{field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</span>
                <span>{responses[field] || '-'}</span>
              </div>
            ))}
          </div>
        ))}
        <div className="flex gap-4 mt-6">
          <button onClick={() => setShowSummary(false)} className="bg-gray-200 px-6 py-2 rounded-full">Back to Chat</button>
          <button onClick={downloadData} className="bg-indigo-600 text-white px-6 py-2 rounded-full">Download Data</button>
          <button onClick={startOver} className="bg-gray-500 text-white px-6 py-2 rounded-full">Start Over</button>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Progress */}
      <div className="p-4 bg-gray-50 border-b">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div className="bg-indigo-600 h-2 rounded-full transition-all" style={{ width: `${(progress.current / progress.total) * 100}%` }} />
        </div>
        <p className="text-sm text-gray-500 mt-2 text-center">Question {progress.current} of {progress.total}</p>
      </div>

      {/* Chat */}
      <div className="h-96 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.type === 'user' ? 'justify-end' : msg.type === 'section' ? 'justify-center' : 'justify-start'}`}>
            {msg.type === 'section' ? (
              <div className="bg-indigo-50 text-indigo-700 px-4 py-2 rounded-lg text-sm font-medium">{msg.text}</div>
            ) : (
              <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-2xl ${msg.type === 'user' ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-800'}`}>
                {msg.text}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Input */}
      {!completed ? (
        <form onSubmit={handleSubmit} className="p-4 border-t flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your answer..."
            className="flex-1 border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button type="submit" className="bg-indigo-600 text-white px-6 py-2 rounded-full hover:bg-indigo-700">Send</button>
        </form>
      ) : (
        <div className="p-4 border-t flex gap-4 justify-center">
          <button onClick={() => setShowSummary(true)} className="bg-indigo-600 text-white px-6 py-2 rounded-full">View Summary</button>
          <button onClick={downloadData} className="bg-gray-200 px-6 py-2 rounded-full">Download Data</button>
          <button onClick={startOver} className="bg-gray-500 text-white px-6 py-2 rounded-full">Start Over</button>
        </div>
      )}
    </div>
  )
}

export default App

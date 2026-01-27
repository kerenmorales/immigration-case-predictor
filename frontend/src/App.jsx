import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
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
      
      if (!response.ok) {
        throw new Error('Prediction failed. Make sure the backend is running.')
      }
      
      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getOutcomeColor = (outcome) => {
    switch (outcome) {
      case 'Granted': return 'text-green-600 bg-green-50'
      case 'Rejected': return 'text-red-600 bg-red-50'
      default: return 'text-yellow-600 bg-yellow-50'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto py-12 px-4">
        <header className="text-center mb-10">
          <h1 className="text-3xl font-bold text-gray-900">
            Immigration Case Outcome Predictor
          </h1>
          <p className="mt-2 text-gray-600">
            AI-powered analysis based on 59,000+ Canadian IRB decisions
          </p>
        </header>

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Case Description / Facts
              </label>
              <textarea
                value={caseText}
                onChange={(e) => setCaseText(e.target.value)}
                rows={6}
                className="w-full border border-gray-300 rounded-md p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Describe the case facts, grounds for the claim, evidence, and any relevant circumstances..."
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Country of Origin (optional)
                </label>
                <input
                  type="text"
                  value={country}
                  onChange={(e) => setCountry(e.target.value)}
                  className="w-full border border-gray-300 rounded-md p-2"
                  placeholder="e.g., Iran, Nigeria, China"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Claim Type (optional)
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
                  <option value="social">Particular social group</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !caseText.trim()}
              className="w-full bg-blue-600 text-white py-3 rounded-md font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
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
            <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
            
            <div className={`inline-block px-4 py-2 rounded-full font-semibold text-lg mb-6 ${getOutcomeColor(prediction.prediction)}`}>
              {prediction.prediction}
            </div>
            
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Confidence</h3>
              <div className="w-full bg-gray-200 rounded-full h-4">
                <div 
                  className="bg-blue-600 h-4 rounded-full"
                  style={{ width: `${prediction.confidence * 100}%` }}
                />
              </div>
              <p className="text-sm text-gray-600 mt-1">
                {(prediction.confidence * 100).toFixed(1)}% confidence
              </p>
            </div>

            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Probability Breakdown</h3>
              <div className="space-y-2">
                {Object.entries(prediction.probabilities).map(([outcome, prob]) => (
                  <div key={outcome} className="flex items-center">
                    <span className="w-24 text-sm">{outcome}</span>
                    <div className="flex-1 bg-gray-200 rounded-full h-2 mx-2">
                      <div 
                        className={`h-2 rounded-full ${
                          outcome === 'Granted' ? 'bg-green-500' :
                          outcome === 'Rejected' ? 'bg-red-500' : 'bg-yellow-500'
                        }`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-600 w-16">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {prediction.factors.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-2">Key Factors Detected</h3>
                <ul className="list-disc list-inside text-sm text-gray-600">
                  {prediction.factors.map((factor, i) => (
                    <li key={i}>{factor}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-md">
              <p className="text-sm text-yellow-800">
                <strong>Disclaimer:</strong> This prediction is based on historical patterns and should not replace professional legal advice. Each case has unique circumstances that may affect the outcome.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

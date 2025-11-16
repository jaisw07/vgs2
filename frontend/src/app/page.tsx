"use client";

import React, { useState } from 'react';
import { Activity, Heart, Stethoscope, Brain, AlertCircle, CheckCircle, Menu, X, Home, Info, Phone, User, MessageSquare, ArrowRight, Loader } from 'lucide-react';

// Types based on API specification
interface StartResponse {
  session_id: string;
  question: string;
  ig: number;
}

interface DescribeResponse {
  parsed_symptoms: Record<string, number>;
  top_diseases: [string, number][];
  question: string;
  ig: number;
}

interface AnswerResponse {
  question: string | null;
  ig: number | null;
  top_diseases: [string, number][];
  is_finished: boolean;
  finish_reason: string | null;
}

interface DiagnosticHistory {
  question: string;
  answer: string;
  symptom?: string;
}

const InteractiveDiagnosticSystem = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  // Session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<string | null>(null);
  const [currentIg, setCurrentIg] = useState<number | null>(null);
  const [topDiseases, setTopDiseases] = useState<[string, number][]>([]);
  const [isFinished, setIsFinished] = useState(false);
  const [finishReason, setFinishReason] = useState<string | null>(null);
  const [history, setHistory] = useState<DiagnosticHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Free text input
  const [freeTextInput, setFreeTextInput] = useState('');
  const [showFreeText, setShowFreeText] = useState(false);
  const [parsedSymptoms, setParsedSymptoms] = useState<Record<string, number>>({});

  const API_BASE_URL = 'http://127.0.0.1:8000';

  // Extract symptom from question (convert "Do you have abdominal pain?" to "abdominal_pain")
  const extractSymptomFromQuestion = (question: string): string => {
    const match = question.match(/Do you have (.+)\?/i);
    if (match && match[1]) {
      return match[1].toLowerCase().replace(/\s+/g, '_');
    }
    return '';
  };

  // Start a new diagnostic session
  const startSession = async () => {
    setLoading(true);
    setError(null);
    setHistory([]);
    setParsedSymptoms({});
    setIsFinished(false);
    setFinishReason(null);

    console.log('Starting session...'); // Debugging log

    try {
      const response = await fetch(`${API_BASE_URL}/start`, {
        method: 'POST',
        cache: 'no-store', // Prevent caching
      });

      console.log('Response status:', response.status); // Debugging log

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response text:', errorText); // Log error details
        throw new Error('Failed to start session');
      }

      const data: StartResponse = await response.json();
      console.log('Session started successfully:', data); // Debugging log

      setSessionId(data.session_id);
      setCurrentQuestion(data.question);
      setCurrentIg(data.ig);
      setShowFreeText(true);
    } catch (err) {
      setError('Failed to connect to the diagnostic system. Please ensure the API server is running at ' + API_BASE_URL);
      console.error('Start session error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Submit free text description
  const submitDescription = async () => {
    if (!sessionId || !freeTextInput.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/describe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          text: freeTextInput,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process description');
      }
      
      const data: DescribeResponse = await response.json();
      setParsedSymptoms(data.parsed_symptoms);
      setTopDiseases(data.top_diseases);
      setCurrentQuestion(data.question);
      setCurrentIg(data.ig);
      setShowFreeText(false);
      
      setHistory([{
        question: 'Initial Description',
        answer: freeTextInput,
      }]);
    } catch (err) {
      setError('Failed to process your description. Please try again.');
      console.error('Describe error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Submit answer to current question
  const submitAnswer = async (answer: number, answerText: string) => {
    if (!sessionId || !currentQuestion) return;
    
    const symptom = extractSymptomFromQuestion(currentQuestion);
    if (!symptom) {
      setError('Could not extract symptom from question');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/answer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          symptom: symptom,
          answer: answer,
        }),
      });
      
      if (response.status === 400) {
        const errorData = await response.json();
        setError(errorData.detail || 'Logical constraint violated. Please check your answers.');
        setLoading(false);
        return;
      }
      
      if (!response.ok) {
        throw new Error('Failed to submit answer');
      }
      
      const data: AnswerResponse = await response.json();
      
      setHistory([...history, {
        question: currentQuestion,
        answer: answerText,
        symptom: symptom,
      }]);
      
      setTopDiseases(data.top_diseases);
      setCurrentQuestion(data.question);
      setCurrentIg(data.ig);
      setIsFinished(data.is_finished);
      setFinishReason(data.finish_reason);
    } catch (err) {
      setError('Failed to submit answer. Please try again.');
      console.error('Answer error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Format disease name for display
  const formatDiseaseName = (disease: string): string => {
    return disease.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  // Format symptom name for display
  const formatSymptomName = (symptom: string): string => {
    return symptom.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  const resetSession = () => {
    setSessionId(null);
    setCurrentQuestion(null);
    setTopDiseases([]);
    setHistory([]);
    setParsedSymptoms({});
    setShowFreeText(false);
    setFreeTextInput('');
    setIsFinished(false);
    setFinishReason(null);
    setError(null);
  };

  const NavBar = () => (
    <nav className="bg-white shadow-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <div className="bg-gradient-to-br from-teal-500 to-blue-600 p-2 rounded-lg">
              <Stethoscope className="text-white" size={24} />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-teal-600 to-blue-600 bg-clip-text text-transparent">
              Interactive Diagnostics
            </span>
          </div>

          <div className="hidden md:flex space-x-8">
            <button
              onClick={() => setCurrentPage('home')}
              className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-all ${
                currentPage === 'home'
                  ? 'text-teal-600 bg-teal-50'
                  : 'text-gray-600 hover:text-teal-600 hover:bg-gray-50'
              }`}
            >
              <Home size={18} />
              <span>Home</span>
            </button>
            <button
              onClick={() => setCurrentPage('about')}
              className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-all ${
                currentPage === 'about'
                  ? 'text-teal-600 bg-teal-50'
                  : 'text-gray-600 hover:text-teal-600 hover:bg-gray-50'
              }`}
            >
              <Info size={18} />
              <span>About</span>
            </button>
            <button
              onClick={() => setCurrentPage('contact')}
              className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-all ${
                currentPage === 'contact'
                  ? 'text-teal-600 bg-teal-50'
                  : 'text-gray-600 hover:text-teal-600 hover:bg-gray-50'
              }`}
            >
              <Phone size={18} />
              <span>Contact</span>
            </button>
          </div>

          <button
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {mobileMenuOpen && (
          <div className="md:hidden pb-4 space-y-2">
            <button
              onClick={() => {
                setCurrentPage('home');
                setMobileMenuOpen(false);
              }}
              className="flex items-center space-x-2 w-full px-3 py-2 rounded-lg text-gray-600 hover:bg-teal-50 hover:text-teal-600"
            >
              <Home size={18} />
              <span>Home</span>
            </button>
            <button
              onClick={() => {
                setCurrentPage('about');
                setMobileMenuOpen(false);
              }}
              className="flex items-center space-x-2 w-full px-3 py-2 rounded-lg text-gray-600 hover:bg-teal-50 hover:text-teal-600"
            >
              <Info size={18} />
              <span>About</span>
            </button>
            <button
              onClick={() => {
                setCurrentPage('contact');
                setMobileMenuOpen(false);
              }}
              className="flex items-center space-x-2 w-full px-3 py-2 rounded-lg text-gray-600 hover:bg-teal-50 hover:text-teal-600"
            >
              <Phone size={18} />
              <span>Contact</span>
            </button>
          </div>
        )}
      </div>
    </nav>
  );

  const HomePage = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-teal-50 to-cyan-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {!sessionId ? (
          <>
            <div className="text-center mb-12">
              <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
                Interactive Disease Diagnosis
              </h1>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Advanced conversational diagnostic system to identify potential health conditions through intelligent questioning
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                <div className="bg-gradient-to-br from-teal-500 to-teal-600 w-12 h-12 rounded-xl flex items-center justify-center mb-4">
                  <Brain className="text-white" size={24} />
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">Smart Questions</h3>
                <p className="text-gray-600">Adaptive questioning system that narrows down possibilities efficiently</p>
              </div>

              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                <div className="bg-gradient-to-br from-blue-500 to-blue-600 w-12 h-12 rounded-xl flex items-center justify-center mb-4">
                  <Activity className="text-white" size={24} />
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">Real-time Analysis</h3>
                <p className="text-gray-600">Probability-based diagnosis that updates with each answer</p>
              </div>

              <div className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                <div className="bg-gradient-to-br from-cyan-500 to-cyan-600 w-12 h-12 rounded-xl flex items-center justify-center mb-4">
                  <MessageSquare className="text-white" size={24} />
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-2">Conversational</h3>
                <p className="text-gray-600">Natural interaction through simple yes/no questions</p>
              </div>
            </div>

            <div className="max-w-2xl mx-auto text-center">
              <button
                onClick={startSession}
                disabled={loading}
                className="bg-gradient-to-r from-teal-600 to-blue-600 text-white px-12 py-4 rounded-lg font-semibold text-lg hover:from-teal-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mx-auto space-x-2"
              >
                {loading ? (
                  <>
                    <Loader className="animate-spin" size={20} />
                    <span>Starting Session...</span>
                  </>
                ) : (
                  <>
                    <span>Start Diagnosis</span>
                    <ArrowRight size={20} />
                  </>
                )}
              </button>
              {error && (
                <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
                  <AlertCircle className="inline mr-2" size={18} />
                  {error}
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="max-w-5xl mx-auto">
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Main Diagnostic Area */}
              <div className="lg:col-span-2 space-y-6">
                {/* Free Text Input */}
                {showFreeText && (
                  <div className="bg-white rounded-2xl shadow-xl p-8">
                    <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
                      <MessageSquare className="mr-2 text-teal-600" size={24} />
                      Describe Your Symptoms
                    </h2>
                    <p className="text-gray-600 mb-4">
                      Start by describing your symptoms in your own words. This helps us understand your condition better.
                    </p>
                    <textarea
                      value={freeTextInput}
                      onChange={(e) => setFreeTextInput(e.target.value)}
                      placeholder="Example: I have a fever and a headache that started two days ago..."
                      rows={6}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none resize-none mb-4"
                    />
                    <div className="flex space-x-3">
                      <button
                        onClick={submitDescription}
                        disabled={loading || !freeTextInput.trim()}
                        className="flex-1 bg-gradient-to-r from-teal-600 to-blue-600 text-white py-3 rounded-lg font-semibold hover:from-teal-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                      >
                        {loading ? (
                          <>
                            <Loader className="animate-spin" size={18} />
                            <span>Processing...</span>
                          </>
                        ) : (
                          <>
                            <span>Continue</span>
                            <ArrowRight size={18} />
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => setShowFreeText(false)}
                        className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-all"
                      >
                        Skip
                      </button>
                    </div>
                  </div>
                )}

                {/* Parsed Symptoms */}
                {Object.keys(parsedSymptoms).length > 0 && (
                  <div className="bg-white rounded-2xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Recognized Symptoms</h3>
                    <div className="flex flex-wrap gap-2">
                      {Object.keys(parsedSymptoms).map((symptom) => (
                        <span
                          key={symptom}
                          className="px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-sm font-medium"
                        >
                          {formatSymptomName(symptom)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Current Question */}
                {!showFreeText && !isFinished && currentQuestion && (
                  <div className="bg-white rounded-2xl shadow-xl p-8">
                    <div className="mb-6">
                      <div className="flex items-center justify-between mb-2">
                        <h2 className="text-2xl font-bold text-gray-800">Question</h2>
                        {currentIg !== null && (
                          <span className="text-sm text-gray-500">
                            Information Gain: {currentIg.toFixed(3)}
                          </span>
                        )}
                      </div>
                      <p className="text-xl text-gray-700 mt-4">{currentQuestion}</p>
                    </div>

                    {error && (
                      <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
                        <AlertCircle className="inline mr-2" size={18} />
                        {error}
                      </div>
                    )}

                    <div className="grid grid-cols-3 gap-4">
                      <button
                        onClick={() => submitAnswer(1, 'Yes')}
                        disabled={loading}
                        className="bg-green-600 hover:bg-green-700 text-white py-4 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {loading ? <Loader className="animate-spin mx-auto" size={20} /> : 'Yes'}
                      </button>
                      <button
                        onClick={() => submitAnswer(0, 'No')}
                        disabled={loading}
                        className="bg-red-600 hover:bg-red-700 text-white py-4 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {loading ? <Loader className="animate-spin mx-auto" size={20} /> : 'No'}
                      </button>
                      <button
                        onClick={() => submitAnswer(-1, 'Unknown')}
                        disabled={loading}
                        className="bg-gray-600 hover:bg-gray-700 text-white py-4 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {loading ? <Loader className="animate-spin mx-auto" size={20} /> : "Don't Know"}
                      </button>
                    </div>
                  </div>
                )}

                {/* Finished State */}
                {isFinished && (
                  <div className="bg-white rounded-2xl shadow-xl p-8">
                    <div className="text-center mb-6">
                      <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                        <CheckCircle className="text-green-600" size={32} />
                      </div>
                      <h2 className="text-2xl font-bold text-gray-800 mb-2">Diagnosis Complete</h2>
                      {finishReason && (
                        <p className="text-gray-600">{finishReason}</p>
                      )}
                    </div>

                    <button
                      onClick={resetSession}
                      className="w-full bg-gradient-to-r from-teal-600 to-blue-600 text-white py-3 rounded-lg font-semibold hover:from-teal-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl"
                    >
                      Start New Diagnosis
                    </button>
                  </div>
                )}

                {/* History */}
                {history.length > 0 && (
                  <div className="bg-white rounded-2xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Question History</h3>
                    <div className="space-y-3 max-h-64 overflow-y-auto">
                      {history.map((item, index) => (
                        <div key={index} className="border-l-4 border-teal-500 pl-4 py-2">
                          <p className="text-sm text-gray-600">{item.question}</p>
                          <p className="text-sm font-semibold text-gray-800 mt-1">
                            Answer: {item.answer}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Disease Probabilities Sidebar */}
              <div className="lg:col-span-1">
                <div className="bg-white rounded-2xl shadow-xl p-6 sticky top-24">
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                    <Activity className="mr-2 text-teal-600" size={20} />
                    Possible Conditions
                  </h3>
                  {topDiseases.length > 0 ? (
                    <div className="space-y-3">
                      {topDiseases.map(([disease, probability], index) => (
                        <div key={disease} className="border-b border-gray-100 pb-3 last:border-0">
                          <div className="flex justify-between items-start mb-2">
                            <span className="font-semibold text-gray-800 text-sm">
                              {index + 1}. {formatDiseaseName(disease)}
                            </span>
                            <span className="text-teal-600 font-bold text-sm">
                              {(probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-teal-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                              style={{ width: `${probability * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-sm">
                      Answer questions to see possible conditions
                    </p>
                  )}
                  
                  <div className="mt-6 pt-6 border-t border-gray-200">
                    <p className="text-xs text-gray-500 leading-relaxed">
                      <strong>Note:</strong> These are preliminary assessments based on your responses. 
                      Always consult a healthcare professional for proper diagnosis.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const AboutPage = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-teal-50 to-cyan-50 py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-8">About Interactive Diagnostics</h1>
        
        <div className="bg-white rounded-2xl shadow-xl p-8 space-y-6">
          <p className="text-gray-700 leading-relaxed">
            The Interactive Diagnostic System is a sophisticated medical AI platform that uses conversational 
            questioning to identify potential health conditions. Unlike traditional symptom checkers, our system 
            adapts its questions based on your previous answers to efficiently narrow down possibilities.
          </p>

          <h2 className="text-2xl font-semibold text-gray-800 mt-8">How It Works</h2>
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="bg-teal-100 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mt-1">
                <span className="text-teal-700 font-bold">1</span>
              </div>
              <div>
                <h3 className="font-semibold text-gray-800">Start a Session</h3>
                <p className="text-gray-600 text-sm">Begin by optionally describing your symptoms in your own words</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-teal-100 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mt-1">
                <span className="text-teal-700 font-bold">2</span>
              </div>
              <div>
                <h3 className="font-semibold text-gray-800">Answer Questions</h3>
                <p className="text-gray-600 text-sm">Respond with Yes, No, or Don't Know to targeted questions</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-teal-100 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mt-1">
                <span className="text-teal-700 font-bold">3</span>
              </div>
              <div>
                <h3 className="font-semibold text-gray-800">Real-time Analysis</h3>
                <p className="text-gray-600 text-sm">Watch as disease probabilities update with each answer</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-teal-100 rounded-full w-8 h-8 flex items-center justify-center flex-shrink-0 mt-1">
                <span className="text-teal-700 font-bold">4</span>
              </div>
              <div>
                <h3 className="font-semibold text-gray-800">Get Results</h3>
                <p className="text-gray-600 text-sm">Receive a ranked list of possible conditions</p>
              </div>
            </div>
          </div>

          <h2 className="text-2xl font-semibold text-gray-800 mt-8">Key Features</h2>
          <ul className="space-y-2 text-gray-700">
            <li className="flex items-start">
              <CheckCircle className="text-teal-600 mr-2 mt-1 flex-shrink-0" size={20} />
              <span>Information gain optimization for efficient questioning</span>
            </li>
            <li className="flex items-start">
              <CheckCircle className="text-teal-600 mr-2 mt-1 flex-shrink-0" size={20} />
              <span>Probability-based disease ranking</span>
            </li>
            <li className="flex items-start">
              <CheckCircle className="text-teal-600 mr-2 mt-1 flex-shrink-0" size={20} />
              <span>Natural language symptom parsing</span>
            </li>
            <li className="flex items-start">
              <CheckCircle className="text-teal-600 mr-2 mt-1 flex-shrink-0" size={20} />
              <span>Logical constraint validation</span>
            </li>
          </ul>

          <h2 className="text-2xl font-semibold text-gray-800 mt-8">API Integration</h2>
          <p className="text-gray-700 leading-relaxed">
            This system connects to a FastAPI backend running at <code className="bg-gray-100 px-2 py-1 rounded">http://127.0.0.1:8000</code>. 
            The API provides three main endpoints: <code className="bg-gray-100 px-2 py-1 rounded">/start</code> to begin a session, 
            <code className="bg-gray-100 px-2 py-1 rounded mx-1">/describe</code> for free-text symptom input, and 
            <code className="bg-gray-100 px-2 py-1 rounded mx-1">/answer</code> to submit responses to questions.
          </p>

          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mt-8">
            <h3 className="font-semibold text-gray-800 mb-2">Important Disclaimer</h3>
            <p className="text-gray-700 leading-relaxed text-sm">
              This diagnostic system is for informational and educational purposes only. It is not a substitute 
              for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
              or other qualified health provider with any questions you may have regarding a medical condition.
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const ContactPage = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-teal-50 to-cyan-50 py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-8">Contact Us</h1>
        
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="grid md:grid-cols-2 gap-8 mb-8">
            <div>
              <h3 className="text-xl font-semibold text-gray-800 mb-4">Get in Touch</h3>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <Phone className="text-teal-600 mt-1" size={20} />
                  <div>
                    <p className="font-medium text-gray-800">Phone</p>
                    <p className="text-gray-600">+1 (555) 123-4567</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <User className="text-teal-600 mt-1" size={20} />
                  <div>
                    <p className="font-medium text-gray-800">Email</p>
                    <p className="text-gray-600">support@interactivediagnostics.ai</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <Info className="text-teal-600 mt-1" size={20} />
                  <div>
                    <p className="font-medium text-gray-800">Address</p>
                    <p className="text-gray-600">123 Medical Plaza<br />Healthcare District<br />San Francisco, CA 94102</p>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-gray-800 mb-4">Business Hours</h3>
              <div className="space-y-2 text-gray-700">
                <p>Monday - Friday: 9:00 AM - 6:00 PM</p>
                <p>Saturday: 10:00 AM - 4:00 PM</p>
                <p>Sunday: Closed</p>
              </div>
              
              <div className="mt-6">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">Technical Support</h3>
                <p className="text-gray-600 text-sm">
                  For API integration support or technical issues, please email our technical team at 
                  <a href="mailto:tech@interactivediagnostics.ai" className="text-teal-600 hover:underline ml-1">
                    tech@interactivediagnostics.ai
                  </a>
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Send us a Message</h3>
            <div className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <input
                  type="text"
                  placeholder="Your Name"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none"
                />
                <input
                  type="email"
                  placeholder="Your Email"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none"
                />
              </div>
              <input
                type="text"
                placeholder="Subject"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none"
              />
              <textarea
                placeholder="Your Message"
                rows={6}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none resize-none"
              />
              <button
                onClick={() => alert('Message sent! We will get back to you soon.')}
                className="w-full bg-gradient-to-r from-teal-600 to-blue-600 text-white py-3 rounded-lg font-semibold hover:from-teal-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl"
              >
                Send Message
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar />
      {currentPage === 'home' && <HomePage />}
      {currentPage === 'about' && <AboutPage />}
      {currentPage === 'contact' && <ContactPage />}
    </div>
  );
};

export default InteractiveDiagnosticSystem;
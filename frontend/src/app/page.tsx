"use client";

import React, { useState } from 'react';
import NavBar from '../components/NavBar';
import HomePage from '../pages/HomePage';
import AboutPage from '../pages/AboutPage';
// import ContactPage from '../pages/ContactPage';
import Results from '../pages/ResultsPage';
import { StartResponse, DescribeResponse, AnswerResponse, DiagnosticHistory } from '../types';

const InteractiveDiagnosticSystem = () => {
  const [currentPage, setCurrentPage] = useState('home');
  
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
        const errorData = await response.json().catch(() => null);
        const errorMessage = errorData?.detail || `Failed to process description (Status: ${response.status})`;
        throw new Error(errorMessage);
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
      const errorMessage = err instanceof Error ? err.message : 'Failed to process your description. Please try again.';
      setError(errorMessage);
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

  return (
    <div className="min-h-screen bg-gray-50">
      <NavBar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      {currentPage === 'home' && (
        <HomePage
          sessionId={sessionId}
          startSession={startSession}
          loading={loading}
          error={error}
          showFreeText={showFreeText}
          freeTextInput={freeTextInput}
          setFreeTextInput={setFreeTextInput}
          submitDescription={submitDescription}
          setShowFreeText={setShowFreeText}
          parsedSymptoms={parsedSymptoms}
          formatSymptomName={formatSymptomName}
          isFinished={isFinished}
          currentQuestion={currentQuestion}
          currentIg={currentIg}
          submitAnswer={submitAnswer}
          finishReason={finishReason}
          resetSession={resetSession}
          history={history}
          topDiseases={topDiseases}
          formatDiseaseName={formatDiseaseName}
        />
      )}
      {currentPage === 'about' && <AboutPage />}
      {currentPage === 'result' && <Results />}
    </div>
  );
};

export default InteractiveDiagnosticSystem;

"use client";
import React from 'react';
import { Brain, Activity, MessageSquare, Loader, ArrowRight, AlertCircle } from 'lucide-react';
import DiagnosticContainer from '../components/diagnostic/DiagnosticContainer';
import DiseaseProbabilities from '../components/diagnostic/DiseaseProbabilities';
import { DiagnosticHistory } from '@/types';

interface HomePageProps {
    sessionId: string | null;
    startSession: () => void;
    loading: boolean;
    error: string | null;
    showFreeText: boolean;
    freeTextInput: string;
    setFreeTextInput: (value: string) => void;
    submitDescription: () => void;
    setShowFreeText: (value: boolean) => void;
    parsedSymptoms: Record<string, number>;
    formatSymptomName: (symptom: string) => string;
    isFinished: boolean;
    currentQuestion: string | null;
    currentIg: number | null;
    submitAnswer: (answer: number, answerText: string) => void;
    finishReason: string | null;
    resetSession: () => void;
    history: DiagnosticHistory[];
    topDiseases: [string, number][];
    formatDiseaseName: (disease: string) => string;
}

const HomePage = (props: HomePageProps) => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-teal-50 to-cyan-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            {!props.sessionId ? (
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
                            onClick={props.startSession}
                            disabled={props.loading}
                            className="bg-gradient-to-r from-teal-600 to-blue-600 text-white px-12 py-4 rounded-lg font-semibold text-lg hover:from-teal-700 hover:to-blue-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mx-auto space-x-2"
                        >
                            {props.loading ? (
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
                        {props.error && (
                            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
                                <AlertCircle className="inline mr-2" size={18} />
                                {props.error}
                            </div>
                        )}
                    </div>
                </>
            ) : (
                <div className="max-w-5xl mx-auto">
                    <div className="grid lg:grid-cols-3 gap-6">
                        <DiagnosticContainer
                            showFreeText={props.showFreeText}
                            freeTextInput={props.freeTextInput}
                            setFreeTextInput={props.setFreeTextInput}
                            submitDescription={props.submitDescription}
                            setShowFreeText={props.setShowFreeText}
                            loading={props.loading}
                            parsedSymptoms={props.parsedSymptoms}
                            formatSymptomName={props.formatSymptomName}
                            isFinished={props.isFinished}
                            currentQuestion={props.currentQuestion}
                            currentIg={props.currentIg}
                            error={props.error}
                            submitAnswer={props.submitAnswer}
                            finishReason={props.finishReason}
                            resetSession={props.resetSession}
                            history={props.history}
                        />
                        <div className="lg:col-span-1">
                            <DiseaseProbabilities
                                topDiseases={props.topDiseases}
                                formatDiseaseName={props.formatDiseaseName}
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    </div>
);

export default HomePage;
"use client";
import React from 'react';
import { CheckCircle } from 'lucide-react';

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

export default AboutPage;
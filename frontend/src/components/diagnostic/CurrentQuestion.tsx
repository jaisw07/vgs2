"use client";
import React from 'react';
import { AlertCircle, Loader } from 'lucide-react';

interface CurrentQuestionProps {
    currentQuestion: string;
    currentIg: number | null;
    error: string | null;
    submitAnswer: (answer: number, answerText: string) => void;
    loading: boolean;
}

const CurrentQuestion = ({ currentQuestion, currentIg, error, submitAnswer, loading }: CurrentQuestionProps) => (
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
);

export default CurrentQuestion;
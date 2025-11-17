"use client";
import React from 'react';
import { CheckCircle } from 'lucide-react';

interface FinishedStateProps {
    finishReason: string | null;
    resetSession: () => void;
}

const FinishedState = ({ finishReason, resetSession }: FinishedStateProps) => (
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
);

export default FinishedState;
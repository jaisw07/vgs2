"use client";
import React from 'react';

interface DiagnosticHistory {
    question: string;
    answer: string;
    symptom?: string;
}

interface HistoryProps {
    history: DiagnosticHistory[];
}

const History = ({ history }: HistoryProps) => (
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
);

export default History;
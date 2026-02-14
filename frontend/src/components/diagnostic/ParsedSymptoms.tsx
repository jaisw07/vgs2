"use client";
import React from 'react';

interface ParsedSymptomsProps {
    parsedSymptoms: Record<string, number>;
    formatSymptomName: (symptom: string) => string;
}

const ParsedSymptoms = ({ parsedSymptoms, formatSymptomName }: ParsedSymptomsProps) => (
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
);

export default ParsedSymptoms;
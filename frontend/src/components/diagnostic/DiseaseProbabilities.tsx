"use client";
import React from 'react';
import { Activity } from 'lucide-react';

interface DiseaseProbabilitiesProps {
    topDiseases: [string, number][];
    formatDiseaseName: (disease: string) => string;
}

const DiseaseProbabilities = ({ topDiseases, formatDiseaseName }: DiseaseProbabilitiesProps) => (
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
);

export default DiseaseProbabilities;
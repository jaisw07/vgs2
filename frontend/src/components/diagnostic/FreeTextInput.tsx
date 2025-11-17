"use client";
import React from 'react';
import { MessageSquare, ArrowRight, Loader } from 'lucide-react';

interface FreeTextInputProps {
    freeTextInput: string;
    setFreeTextInput: (value: string) => void;
    submitDescription: () => void;
    setShowFreeText: (value: boolean) => void;
    loading: boolean;
}

const FreeTextInput = ({ freeTextInput, setFreeTextInput, submitDescription, setShowFreeText, loading }: FreeTextInputProps) => (
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
);

export default FreeTextInput;
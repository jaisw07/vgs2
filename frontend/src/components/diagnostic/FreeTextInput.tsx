"use client";
import React, { useState, useRef } from 'react';
import { MessageSquare, ArrowRight, Loader, Mic, MicOff } from 'lucide-react';

interface FreeTextInputProps {
    freeTextInput: string;
    setFreeTextInput: (value: string) => void;
    submitDescription: () => void;
    setShowFreeText: (value: boolean) => void;
    loading: boolean;
}

const FreeTextInput = ({ freeTextInput, setFreeTextInput, submitDescription, setShowFreeText, loading }: FreeTextInputProps) => {
    const [isRecording, setIsRecording] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [suggestions, setSuggestions] = useState<Array<{value: string, label: string, confidence: string, match_type: string}>>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
    const [suggestionPosition, setSuggestionPosition] = useState({ top: 0, left: 0 });
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const suggestTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const startRecording = async () => {
        try {
            setShowSuggestions(false); // Hide suggestions while recording
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                await transcribeAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const transcribeAudio = async (audioBlob: Blob) => {
        setIsTranscribing(true);
        try {
            // Convert audio blob to base64
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            
            reader.onloadend = async () => {
                const base64Audio = reader.result as string;
                const base64Content = base64Audio.split(',')[1];

                const apiKey = process.env.NEXT_PUBLIC_GOOGLE_API_KEY;
                
                if (!apiKey) {
                    throw new Error('Google API key not found');
                }

                const response = await fetch(
                    `https://speech.googleapis.com/v1/speech:recognize?key=${apiKey}`,
                    {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            config: {
                                encoding: 'WEBM_OPUS',
                                sampleRateHertz: 48000,
                                languageCode: 'en-US',
                                enableAutomaticPunctuation: true,
                            },
                            audio: {
                                content: base64Content,
                            },
                        }),
                    }
                );

                if (!response.ok) {
                    throw new Error('Transcription failed');
                }

                const data = await response.json();
                
                if (data.results && data.results.length > 0) {
                    const transcript = data.results
                        .map((result: any) => result.alternatives[0].transcript)
                        .join(' ');
                    
                    setFreeTextInput(freeTextInput + (freeTextInput ? ' ' : '') + transcript);
                } else {
                    alert('No speech detected. Please try again.');
                }
            };
        } catch (error) {
            console.error('Transcription error:', error);
            alert('Failed to transcribe audio. Please try again.');
        } finally {
            setIsTranscribing(false);
        }
    };

    const fetchSuggestions = async (text: string) => {
        if (text.length < 3) {
            setSuggestions([]);
            setShowSuggestions(false);
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/suggest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    cursor_position: text.length,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                if (data.suggestions && data.suggestions.length > 0) {
                    setSuggestions(data.suggestions);
                    setShowSuggestions(true);
                    setSelectedSuggestionIndex(0);
                } else {
                    setSuggestions([]);
                    setShowSuggestions(false);
                }
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
        }
    };

    const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const newText = e.target.value;
        setFreeTextInput(newText);

        // Calculate cursor position for floating suggestion
        if (textareaRef.current) {
            const textarea = textareaRef.current;
            const cursorPosition = textarea.selectionStart;
            
            // Get the position of the cursor relative to textarea
            const textBeforeCursor = newText.substring(0, cursorPosition);
            const lines = textBeforeCursor.split('\n');
            const currentLineNumber = lines.length - 1;
            const currentLineText = lines[currentLineNumber];
            
            // Approximate character width and line height
            const charWidth = 8;
            const lineHeight = 24;
            
            const top = currentLineNumber * lineHeight + 30;
            const left = currentLineText.length * charWidth + 10;
            
            setSuggestionPosition({ top, left });
        }

        // Clear previous timeout
        if (suggestTimeoutRef.current) {
            clearTimeout(suggestTimeoutRef.current);
        }

        // Debounce suggestions
        suggestTimeoutRef.current = setTimeout(() => {
            fetchSuggestions(newText);
        }, 300);
    };

    const applySuggestion = (suggestion: {value: string, label: string}) => {
        const words = freeTextInput.split(' ');
        words[words.length - 1] = suggestion.label;
        setFreeTextInput(words.join(' ') + ' ');
        setShowSuggestions(false);
        setSuggestions([]);
        textareaRef.current?.focus();
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (!showSuggestions || suggestions.length === 0) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSelectedSuggestionIndex((prev) => 
                prev < suggestions.length - 1 ? prev + 1 : prev
            );
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSelectedSuggestionIndex((prev) => (prev > 0 ? prev - 1 : 0));
        } else if (e.key === 'Enter' && showSuggestions) {
            e.preventDefault();
            applySuggestion(suggestions[selectedSuggestionIndex]);
        } else if (e.key === 'Escape') {
            setShowSuggestions(false);
        }
    };

    return (
    <div className="bg-white rounded-2xl shadow-xl p-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
            <MessageSquare className="mr-2 text-teal-600" size={24} />
            Describe Your Symptoms
        </h2>
        <p className="text-gray-600 mb-4">
            Start by describing your symptoms in your own words or use voice input. As you type, we'll suggest matching symptoms.
        </p>
        <div className="relative">
            <textarea
                ref={textareaRef}
                value={freeTextInput}
                onChange={handleTextChange}
                onKeyDown={handleKeyDown}
                onFocus={() => setShowSuggestions(suggestions.length > 0)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                placeholder="Example: I have a fever and a headache that started two days ago..."
                rows={6}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent outline-none resize-none text-gray-800"
                disabled={isRecording || isTranscribing}
            />
            
            {/* Floating Suggestion Box */}
            {showSuggestions && suggestions.length > 0 && (
                <div 
                    className="absolute z-50 bg-white rounded-lg shadow-2xl border-2 border-teal-500 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200"
                    style={{
                        top: `${suggestionPosition.top}px`,
                        left: `${suggestionPosition.left}px`,
                        minWidth: '280px',
                        maxWidth: '350px'
                    }}
                >
                    {/* Header */}
                    <div className="bg-gradient-to-r from-teal-500 to-blue-500 px-4 py-2 flex items-center justify-between">
                        <span className="text-white text-xs font-semibold flex items-center">
                            <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                            </svg>
                            Did you mean?
                        </span>
                        <button 
                            onClick={() => setShowSuggestions(false)}
                            className="text-white hover:text-gray-200 transition-colors"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                    
                    {/* Suggestions List */}
                    <div className="max-h-48 overflow-y-auto">
                        {suggestions.map((suggestion, index) => (
                            <div
                                key={index}
                                onClick={() => applySuggestion(suggestion)}
                                className={`px-4 py-3 cursor-pointer transition-all border-b border-gray-100 last:border-b-0 ${
                                    index === selectedSuggestionIndex
                                        ? 'bg-teal-50 border-l-4 border-l-teal-500'
                                        : 'hover:bg-gray-50 border-l-4 border-l-transparent'
                                }`}
                            >
                                <div className="flex items-center justify-between">
                                    <span className="font-medium text-gray-800 text-sm">{suggestion.label}</span>
                                    {suggestion.confidence === 'high' && (
                                        <span className="text-xs px-2 py-1 rounded-full bg-green-100 text-green-700 font-medium">
                                            ✓ Match
                                        </span>
                                    )}
                                    {suggestion.confidence === 'ai_suggested' && (
                                        <span className="text-xs px-2 py-1 rounded-full bg-purple-100 text-purple-700 font-medium flex items-center">
                                            <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                                <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
                                            </svg>
                                            AI
                                        </span>
                                    )}
                                    {suggestion.confidence === 'medium' && (
                                        <span className="text-xs px-2 py-1 rounded-full bg-blue-100 text-blue-700 font-medium">
                                            Similar
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                    
                    {/* Footer Hint */}
                    <div className="bg-gray-50 px-4 py-2 text-xs text-gray-500 border-t border-gray-200">
                        <span className="flex items-center">
                            <kbd className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs mr-1">↑↓</kbd>
                            Navigate
                            <kbd className="px-2 py-0.5 bg-white border border-gray-300 rounded text-xs ml-3 mr-1">Enter</kbd>
                            Select
                        </span>
                    </div>
                </div>
            )}
            
            {isTranscribing && (
                <div className="absolute inset-0 bg-white bg-opacity-90 flex items-center justify-center rounded-lg">
                    <div className="flex items-center space-x-2 text-teal-600">
                        <Loader className="animate-spin" size={24} />
                        <span className="font-semibold">Transcribing...</span>
                    </div>
                </div>
            )}
        </div>
        <div className="flex space-x-3 mb-4 mt-4">
            <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={loading || isTranscribing}
                className={`flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed ${
                    isRecording
                        ? 'bg-red-600 text-white hover:bg-red-700'
                        : 'bg-gradient-to-r from-blue-600 to-teal-600 text-white hover:from-purple-700 hover:to-pink-700'
                }`}
            >
                {isRecording ? (
                    <>
                        <MicOff size={18} />
                        <span>Stop Recording</span>
                    </>
                ) : (
                    <>
                        <Mic size={18} />
                        <span>Voice Input</span>
                    </>
                )}
            </button>
        </div>
        <div className="flex space-x-3">
            <button
                onClick={submitDescription}
                disabled={loading || !freeTextInput.trim() || isRecording || isTranscribing}
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
                disabled={isRecording || isTranscribing}
                className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
                Skip
            </button>
        </div>
    </div>
);
};

export default FreeTextInput;
"use client";
import React from 'react';
import FreeTextInput from './FreeTextInput';
import ParsedSymptoms from './ParsedSymptoms';
import CurrentQuestion from './CurrentQuestion';
import FinishedState from './FinishedState';
import History from './History';
import { DiagnosticHistory } from '@/types';

interface DiagnosticContainerProps {
    showFreeText: boolean;
    freeTextInput: string;
    setFreeTextInput: (value: string) => void;
    submitDescription: () => void;
    setShowFreeText: (value: boolean) => void;
    loading: boolean;
    parsedSymptoms: Record<string, number>;
    formatSymptomName: (symptom: string) => string;
    isFinished: boolean;
    currentQuestion: string | null;
    currentIg: number | null;
    error: string | null;
    submitAnswer: (answer: number, answerText: string) => void;
    finishReason: string | null;
    resetSession: () => void;
    history: DiagnosticHistory[];
}

const DiagnosticContainer = (props: DiagnosticContainerProps) => (
    <div className="lg:col-span-2 space-y-6">
        {props.showFreeText && (
            <FreeTextInput
                freeTextInput={props.freeTextInput}
                setFreeTextInput={props.setFreeTextInput}
                submitDescription={props.submitDescription}
                setShowFreeText={props.setShowFreeText}
                loading={props.loading}
            />
        )}

        {Object.keys(props.parsedSymptoms).length > 0 && (
            <ParsedSymptoms
                parsedSymptoms={props.parsedSymptoms}
                formatSymptomName={props.formatSymptomName}
            />
        )}

        {!props.showFreeText && !props.isFinished && props.currentQuestion && (
            <CurrentQuestion
                currentQuestion={props.currentQuestion}
                currentIg={props.currentIg}
                error={props.error}
                submitAnswer={props.submitAnswer}
                loading={props.loading}
            />
        )}

        {props.isFinished && (
            <FinishedState
                finishReason={props.finishReason}
                resetSession={props.resetSession}
            />
        )}

        {props.history.length > 0 && (
            <History history={props.history} />
        )}
    </div>
);

export default DiagnosticContainer;
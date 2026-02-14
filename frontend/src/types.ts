
export interface StartResponse {
  session_id: string;
  question: string;
  ig: number;
}

export interface DescribeResponse {
  parsed_symptoms: Record<string, number>;
  top_diseases: [string, number][];
  question: string;
  ig: number;
}

export interface AnswerResponse {
  question: string | null;
  ig: number | null;
  top_diseases: [string, number][];
  is_finished: boolean;
  finish_reason: string | null;
}

export interface DiagnosticHistory {
  question: string;
  answer: string;
  symptom?: string;
}
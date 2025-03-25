export interface InterviewSettings {
  categories: string[];
  difficultyLevels: string[];
  numQuestions: number;
}

export interface Question {
  questionId: number;
  question: string;
}

export interface AnswerAnalysis {
  similarity: number;
  cosineSimilarity: number;
  fuzzySimilarity: number;
  sentiment: string;
}

export interface InterviewResults {
  score: number;
  totalQuestions: number;
  overallSentiment: string;
  feedback: string;
  analysis: {
    averageSentiment: number;
  };
}
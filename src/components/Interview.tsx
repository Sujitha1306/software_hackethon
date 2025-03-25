import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Mic, Volume2, Loader } from 'lucide-react';
import type { InterviewSettings, Question, AnswerAnalysis, InterviewResults } from '../types';

const Interview: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const settings = location.state as InterviewSettings;

  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [analysis, setAnalysis] = useState<AnswerAnalysis | null>(null);
  const [results, setResults] = useState<InterviewResults | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    startInterview();
  }, []);

  const startInterview = async () => {
    try {
      setIsLoading(true);
      const response = await axios.post('http://127.0.0.1:5000/api/interview/start', settings);
      setCurrentQuestion(response.data);
    } catch (error) {
      console.error('Error starting interview:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSpeechToText = () => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new (window as any).webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setAnswer(prev => prev + ' ' + transcript);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.start();
    } else {
      alert('Speech recognition is not supported in this browser.');
    }
  };

  const speakText = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  const handleSubmit = async () => {
    try {
      setIsLoading(true);
      const response = await axios.post('http://127.0.0.1:5000/api/interview/submit', {
        answer
      });

      if (response.data.interviewCompleted) {
        setResults(response.data);
      } else {
        setAnalysis(response.data.analysis);
        setCurrentQuestion(response.data.nextQuestion);
        setAnswer('');
      }
    } catch (error) {
      console.error('Error submitting answer:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Loader className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  if (results) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Interview Results</h1>
          
          <div className="space-y-6">
            <div className="bg-blue-50 p-6 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Score</h2>
              <p className="text-4xl font-bold text-blue-600">
                {results.score} / {results.totalQuestions}
              </p>
            </div>

            <div className="bg-purple-50 p-6 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Overall Sentiment</h2>
              <p className="text-2xl font-semibold text-purple-600">
                {results.overallSentiment}
              </p>
            </div>

            <div className="bg-green-50 p-6 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Feedback</h2>
              <p className="text-lg text-green-700">{results.feedback}</p>
            </div>
          </div>

          <button
            onClick={() => navigate('/')}
            className="mt-8 bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-8 rounded-lg transition-colors"
          >
            Start New Interview
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        {currentQuestion && (
          <div className="bg-white rounded-lg shadow-md p-8 mb-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-800">
                Question {currentQuestion.questionId}
              </h2>
              <button
                onClick={() => speakText(currentQuestion.question)}
                className="text-blue-500 hover:text-blue-600"
                title="Listen to question"
              >
                <Volume2 className="w-6 h-6" />
              </button>
            </div>
            
            <p className="text-lg text-gray-700 mb-6">{currentQuestion.question}</p>

            <div className="space-y-4">
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Type your answer here..."
                className="w-full h-32 p-4 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />

              <div className="flex space-x-4">
                <button
                  onClick={handleSpeechToText}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg ${
                    isListening
                      ? 'bg-red-500 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <Mic className="w-5 h-5" />
                  <span>{isListening ? 'Recording...' : 'Record Answer'}</span>
                </button>

                <button
                  onClick={handleSubmit}
                  disabled={!answer.trim()}
                  className={`px-6 py-2 rounded-lg font-semibold ${
                    answer.trim()
                      ? 'bg-blue-500 text-white hover:bg-blue-600'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  Submit Answer
                </button>
              </div>
            </div>
          </div>
        )}

        {analysis && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-semibold mb-4">Answer Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="font-medium">Similarity Score</p>
                <p className="text-2xl font-bold text-blue-600">
                  {(analysis.similarity * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <p className="font-medium">Sentiment</p>
                <p className="text-2xl font-bold text-purple-600">
                  {analysis.sentiment}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Interview;
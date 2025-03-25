import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain } from 'lucide-react';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full bg-white rounded-xl shadow-2xl p-8 text-center">
        <Brain className="w-20 h-20 mx-auto text-blue-500 mb-6" />
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          AI-Powered Interview Practice
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Enhance your interview skills with our intelligent interview simulator.
          Get real-time feedback and improve your performance.
        </p>
        <button
          onClick={() => navigate('/categories')}
          className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-8 rounded-lg transition-colors duration-200 transform hover:scale-105"
        >
          Start Interview
        </button>
      </div>
    </div>
  );
};

export default LandingPage;
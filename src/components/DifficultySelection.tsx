import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { AlertCircle } from 'lucide-react';

const DifficultySelection: React.FC = () => {
  const [difficulties, setDifficulties] = useState<string[]>([]);
  const [selectedDifficulties, setSelectedDifficulties] = useState<string[]>([]);
  const [numQuestions, setNumQuestions] = useState<number>(5);
  const navigate = useNavigate();
  const location = useLocation();
  const selectedCategories = location.state?.categories || [];

  useEffect(() => {
    const fetchDifficulties = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/api/difficulty-levels');
        setDifficulties(response.data);
      } catch (error) {
        console.error('Error fetching difficulties:', error);
      }
    };
    fetchDifficulties();
  }, []);

  const toggleDifficulty = (difficulty: string) => {
    setSelectedDifficulties(prev =>
      prev.includes(difficulty)
        ? prev.filter(d => d !== difficulty)
        : [...prev, difficulty]
    );
  };

  const handleStart = () => {
    if (selectedDifficulties.length > 0) {
      navigate('/interview', {
        state: {
          categories: selectedCategories,
          difficultyLevels: selectedDifficulties,
          numQuestions
        }
      });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">
          Interview Settings
        </h1>

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Select Difficulty Levels</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {difficulties.map(difficulty => (
              <button
                key={difficulty}
                onClick={() => toggleDifficulty(difficulty)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  selectedDifficulties.includes(difficulty)
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300'
                }`}
              >
                <span className="text-lg font-medium">{difficulty}</span>
              </button>
            ))}
          </div>

          <h2 className="text-xl font-semibold mb-4">Number of Questions</h2>
          <input
            type="range"
            min="1"
            max="20"
            value={numQuestions}
            onChange={(e) => setNumQuestions(Number(e.target.value))}
            className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
          />
          <div className="text-center mt-2">
            <span className="text-lg font-medium">{numQuestions} questions</span>
          </div>
        </div>

        {selectedDifficulties.length === 0 && (
          <div className="flex items-center space-x-2 text-yellow-600 mb-4">
            <AlertCircle className="w-5 h-5" />
            <span>Please select at least one difficulty level</span>
          </div>
        )}

        <button
          onClick={handleStart}
          disabled={selectedDifficulties.length === 0}
          className={`w-full md:w-auto px-8 py-3 rounded-lg font-semibold text-white transition-colors ${
            selectedDifficulties.length > 0
              ? 'bg-blue-500 hover:bg-blue-600'
              : 'bg-gray-300 cursor-not-allowed'
          }`}
        >
          Start Interview
        </button>
      </div>
    </div>
  );
};

export default DifficultySelection;
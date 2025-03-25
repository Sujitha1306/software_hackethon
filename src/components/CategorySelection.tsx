import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { CheckCircle } from 'lucide-react';

const CategorySelection: React.FC = () => {
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/api/categories');
        setCategories(response.data);
      } catch (error) {
        console.error('Error fetching categories:', error);
      }
    };
    fetchCategories();
  }, []);

  const toggleCategory = (category: string) => {
    setSelectedCategories(prev =>
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const handleNext = () => {
    if (selectedCategories.length > 0) {
      navigate('/difficulty', { state: { categories: selectedCategories } });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">
          Select Interview Categories
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {categories.map(category => (
            <button
              key={category}
              onClick={() => toggleCategory(category)}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedCategories.includes(category)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300'
              }`}
            >
              <div className="flex items-center space-x-2">
                <CheckCircle
                  className={`w-5 h-5 ${
                    selectedCategories.includes(category)
                      ? 'text-blue-500'
                      : 'text-gray-300'
                  }`}
                />
                <span className="text-lg font-medium">{category}</span>
              </div>
            </button>
          ))}
        </div>
        <button
          onClick={handleNext}
          disabled={selectedCategories.length === 0}
          className={`w-full md:w-auto px-8 py-3 rounded-lg font-semibold text-white transition-colors ${
            selectedCategories.length > 0
              ? 'bg-blue-500 hover:bg-blue-600'
              : 'bg-gray-300 cursor-not-allowed'
          }`}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default CategorySelection;
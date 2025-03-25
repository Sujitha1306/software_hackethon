import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import CategorySelection from './components/CategorySelection';
import DifficultySelection from './components/DifficultySelection';
import Interview from './components/Interview';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/categories" element={<CategorySelection />} />
        <Route path="/difficulty" element={<DifficultySelection />} />
        <Route path="/interview" element={<Interview />} />
      </Routes>
    </Router>
  );
}

export default App;
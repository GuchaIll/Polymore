import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import PolyForge from './pages/PolyForge';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PolyForge />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;

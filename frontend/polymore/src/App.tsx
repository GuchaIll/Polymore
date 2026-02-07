import React, { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import PolyForge from './pages/PolyForge';
import { initRDKit } from './util';

/**
 * Main App component
 * Initializes RDKit.js WebAssembly module on mount for early loading
 */
const App: React.FC = () => {
  const [rdkitReady, setRdkitReady] = useState(false);
  const [rdkitError, setRdkitError] = useState<string | null>(null);

  useEffect(() => {
    // Preload RDKit.js WebAssembly module
    initRDKit()
      .then(() => {
        setRdkitReady(true);
      })
      .catch((error) => {
        console.error("Failed to initialize RDKit.js:", error);
        setRdkitError("Chemistry engine failed to load");
      });
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PolyForge rdkitReady={rdkitReady} rdkitError={rdkitError} />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;

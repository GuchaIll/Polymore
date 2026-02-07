/**
 * Module: ResultsPageToolbarButton
 * Purpose: Toolbar button for Results page navigation
 * Usage: Add to Toolbar for Results toggle
 */

import React from 'react';

interface ResultsPageToolbarButtonProps {
  isActive: boolean;
  onClick: () => void;
}

const ResultsPageToolbarButton: React.FC<ResultsPageToolbarButtonProps> = ({ isActive, onClick }) => (
  <button
    className={`px-3 py-2 rounded-lg font-semibold text-sm transition-colors ${isActive ? 'bg-poly-accent text-white' : 'bg-poly-accent/10 text-poly-accent hover:bg-poly-accent/20'}`}
    onClick={onClick}
    title="View Results"
    style={{ minWidth: 80 }}
  >
    Results
  </button>
);

export default ResultsPageToolbarButton;

import React, { useState } from 'react';
import { Molecule } from '../../types';

interface MoleculeCardProps {
  molecule: Molecule;
  onSelect: (molecule: Molecule) => void;
  onDragStart: (molecule: Molecule) => void;
  onDragEnd: () => void;
}

const MoleculeCard: React.FC<MoleculeCardProps> = ({
  molecule,
  onSelect,
  onDragStart,
  onDragEnd
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);

  const handleDragStart = (e: React.DragEvent) => {
    setIsDragging(true);
    setShowTooltip(false);
    onDragStart(molecule);
    e.dataTransfer.effectAllowed = 'copy';
  };

  const handleDragEnd = () => {
    setIsDragging(false);
    onDragEnd();
  };

  // Build tooltip content from extended molecule fields
  const hasExtendedInfo = molecule.description || molecule.commonItems || molecule.sustainabilityImpact;

  return (
    <div
      className={`
        relative bg-poly-light-bg dark:bg-poly-bg border-2 border-poly-light-border dark:border-poly-border rounded-xl p-3 cursor-grab
        transition-all text-center hover:border-poly-light-accent dark:hover:border-poly-accent hover:-translate-y-0.5
        hover:shadow-lg hover:shadow-poly-light-accent/30 dark:hover:shadow-poly-accent/30 active:cursor-grabbing
        ${isDragging ? 'opacity-50' : ''}
      `}
      draggable
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onClick={() => onSelect(molecule)}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      {/* Icon with molecule color background */}
      <div 
        className="text-lg font-bold mb-2 w-10 h-10 mx-auto rounded-full flex items-center justify-center text-white"
        style={{ backgroundColor: molecule.color }}
      >
        {molecule.icon}
      </div>
      <div className="text-poly-light-text dark:text-white text-sm font-semibold mb-1 truncate">{molecule.name}</div>
      <div className="text-poly-light-muted dark:text-gray-500 text-xs">{molecule.formula}</div>
      
      {/* Extended info tooltip */}
      {showTooltip && hasExtendedInfo && (
        <div className="absolute z-50 left-full ml-2 top-0 w-64 p-3 bg-poly-light-sidebar dark:bg-poly-card border border-poly-light-border dark:border-poly-border rounded-lg shadow-xl text-left pointer-events-none">
          <div className="text-poly-light-text dark:text-white text-sm font-bold mb-1">{molecule.name}</div>
          <div className="text-poly-light-muted dark:text-gray-400 text-xs mb-2">{molecule.smiles}</div>
          
          {molecule.description && (
            <p className="text-poly-light-text dark:text-gray-300 text-xs mb-2">{molecule.description}</p>
          )}
          
          {molecule.resultingPolymer && (
            <div className="text-xs mb-1">
              <span className="text-poly-light-accent dark:text-poly-accent font-semibold">Polymer: </span>
              <span className="text-poly-light-text dark:text-gray-300">{molecule.resultingPolymer}</span>
            </div>
          )}
          
          {molecule.mechanicalEffect && (
            <div className="text-xs mb-1">
              <span className="text-poly-light-accent dark:text-poly-accent font-semibold">Effect: </span>
              <span className="text-poly-light-text dark:text-gray-300">{molecule.mechanicalEffect}</span>
            </div>
          )}
          
          {molecule.sustainabilityImpact && (
            <div className="text-xs mb-1">
              <span className="text-emerald-500 font-semibold">Sustainability: </span>
              <span className="text-poly-light-text dark:text-gray-300">{molecule.sustainabilityImpact}</span>
            </div>
          )}
          
          {molecule.commonItems && molecule.commonItems.length > 0 && (
            <div className="text-xs">
              <span className="text-blue-500 font-semibold">Common Uses: </span>
              <span className="text-poly-light-text dark:text-gray-300">{molecule.commonItems.slice(0, 3).join(', ')}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MoleculeCard;
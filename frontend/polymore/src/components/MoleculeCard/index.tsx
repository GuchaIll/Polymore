import React, { useState, useRef } from 'react';
import { createPortal } from 'react-dom';
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
  const [tooltipPos, setTooltipPos] = useState({ top: 0, left: 0 });
  const cardRef = useRef<HTMLDivElement>(null);

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

  const handleMouseEnter = () => {
    if (cardRef.current) {
      const rect = cardRef.current.getBoundingClientRect();
      setTooltipPos({
        top: rect.top,
        left: rect.right + 8
      });
    }
    setShowTooltip(true);
  };

  // Build tooltip content from extended molecule fields
  const hasExtendedInfo = molecule.description || molecule.commonItems || molecule.sustainabilityImpact;

  return (
    <div
      ref={cardRef}
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
      onMouseEnter={handleMouseEnter}
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
      <div className="text-poly-light-muted dark:text-poly-muted text-xs">{molecule.formula}</div>
      
      {/* Extended info tooltip - rendered via portal to escape overflow */}
      {showTooltip && hasExtendedInfo && createPortal(
        <div 
          className="fixed z-[70] w-64 p-3 bg-poly-light-sidebar dark:bg-poly-card border border-poly-light-border dark:border-poly-border rounded-lg shadow-xl text-left pointer-events-none"
          style={{ top: tooltipPos.top, left: tooltipPos.left }}
        >
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
        </div>,
        document.body
      )}
    </div>
  );
};

export default MoleculeCard;
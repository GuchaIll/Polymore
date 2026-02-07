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

  const handleDragStart = (e: React.DragEvent) => {
    setIsDragging(true);
    onDragStart(molecule);
    e.dataTransfer.effectAllowed = 'copy';
  };

  const handleDragEnd = () => {
    setIsDragging(false);
    onDragEnd();
  };

  return (
    <div
      className={`
        bg-poly-light-bg dark:bg-poly-bg border-2 border-poly-light-border dark:border-poly-border rounded-xl p-3 cursor-grab
        transition-all text-center hover:border-poly-light-accent dark:hover:border-poly-accent hover:-translate-y-0.5
        hover:shadow-lg hover:shadow-poly-light-accent/30 dark:hover:shadow-poly-accent/30 active:cursor-grabbing
        ${isDragging ? 'opacity-50' : ''}
      `}
      draggable
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onClick={() => onSelect(molecule)}
    >
      <div className="text-3xl mb-2">{molecule.icon}</div>
      <div className="text-poly-light-text dark:text-white text-sm font-semibold mb-1">{molecule.name}</div>
      <div className="text-poly-light-muted dark:text-gray-500 text-xs">{molecule.formula}</div>
    </div>
  );
};

export default MoleculeCard;
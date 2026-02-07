import React from 'react';
import MoleculeCard from '../MoleculeCard';
import { moleculeLibrary } from '../../data/moleculeGroupLibrary';
import { Molecule, ToolType } from '../../types';

interface Tool {
  id: ToolType;
  icon: string;
  label: string;
}

const tools: Tool[] = [
  { id: 'select', icon: '👆', label: 'Select' },
  { id: 'add', icon: '➕', label: 'Add' },
  { id: 'remove', icon: '✂️', label: 'Remove' },
  { id: 'connect', icon: '🔗', label: 'Bond' }
];

interface SidebarProps {
  currentTool: ToolType;
  onToolChange: (tool: ToolType) => void;
  onMoleculeSelect: (molecule: Molecule) => void;
  onDragStart: (molecule: Molecule) => void;
  onDragEnd: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  currentTool,
  onToolChange,
  onMoleculeSelect,
  onDragStart,
  onDragEnd
}) => {
  return (
    <div className="w-[280px] bg-poly-sidebar border-r-2 border-poly-border flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-5 bg-gradient-to-br from-poly-accent to-poly-accent2 text-white">
        <h1 className="text-2xl font-bold mb-1">🧬 PolyForge</h1>
        <p className="text-sm opacity-90">Polymer Builder</p>
      </div>

      {/* Tools */}
      <div className="p-4 border-b border-poly-border">
        <h3 className="text-poly-danger text-sm font-bold mb-3 uppercase tracking-wider">
          🔧 Tools
        </h3>
        <div className="flex gap-2 flex-wrap">
          {tools.map(tool => (
            <button
              key={tool.id}
              className={`
                flex-1 min-w-[70px] py-2.5 px-2 border-2 rounded-lg
                text-white cursor-pointer transition-all text-xs text-center
                ${currentTool === tool.id
                  ? 'border-poly-danger bg-poly-danger'
                  : 'border-poly-border bg-poly-bg hover:border-poly-accent hover:bg-poly-border'
                }
              `}
              onClick={() => onToolChange(tool.id)}
            >
              <span className="text-lg block mb-1">{tool.icon}</span>
              {tool.label}
            </button>
          ))}
        </div>
      </div>

      {/* Molecules */}
      <div className="flex-1 p-4 overflow-y-auto">
        <h3 className="text-poly-danger text-sm font-bold mb-3 uppercase tracking-wider">
          📦 Molecules
        </h3>

        <div className="text-poly-accent text-xs mb-2 pb-1 border-b border-poly-border">
          Basic Units
        </div>
        <div className="grid grid-cols-2 gap-2 mb-4">
          {moleculeLibrary.basic.map(mol => (
            <MoleculeCard
              key={mol.name}
              molecule={mol}
              onSelect={onMoleculeSelect}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
            />
          ))}
        </div>

        <div className="text-poly-accent text-xs mb-2 pb-1 border-b border-poly-border">
          Functional Groups
        </div>
        <div className="grid grid-cols-2 gap-2 mb-4">
          {moleculeLibrary.functional.map(mol => (
            <MoleculeCard
              key={mol.name}
              molecule={mol}
              onSelect={onMoleculeSelect}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
            />
          ))}
        </div>

        <div className="text-poly-accent text-xs mb-2 pb-1 border-b border-poly-border">
          Polymer Monomers
        </div>
        <div className="grid grid-cols-2 gap-2">
          {moleculeLibrary.monomers.map(mol => (
            <MoleculeCard
              key={mol.name}
              molecule={mol}
              onSelect={onMoleculeSelect}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
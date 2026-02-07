import React, { ComponentType } from 'react';
import MoleculeCard from '../MoleculeCard';
import { moleculeLibrary } from '../../data/moleculeGroupLibrary';
import { Molecule, ToolType } from '../../types';
import { Lasso, Plus, Trash, Link2, Move, Dna } from 'lucide-react';

interface Tool {
  id: ToolType;
  icon: ComponentType<any>;
  label: string;
}

const tools: Tool[] = [
  { id: 'select', icon: Lasso, label: 'Select' },
  { id: 'add', icon: Plus, label: 'Add' },
  { id: 'remove', icon: Trash, label: 'Remove' },
  { id: 'connect', icon: Link2, label: 'Bond' },
  { id: 'move', icon: Move, label: 'Move' }
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
    <div className="w-[280px] bg-poly-light-sidebar dark:bg-poly-sidebar border-r-2 border-poly-light-border dark:border-poly-border flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-5 bg-gradient-to-br from-poly-light-accent to-emerald-400 dark:from-poly-accent dark:to-poly-accent2 text-white">
        <h1 className="text-2xl font-bold mb-1 flex items-center gap-2">
          <Dna className="w-6 h-6" />
          PolyForge
        </h1>
        <p className="text-sm opacity-90">Polymer Builder</p>
      </div>

      {/* Tools */}
      <div className="p-4 border-b border-poly-light-border dark:border-poly-border">
        <h3 className="text-poly-light-accent dark:text-poly-danger text-sm font-bold mb-3 uppercase tracking-wider">
          Tools
        </h3>
        <div className="flex gap-2 flex-wrap">
          {tools.map(tool => (
            <button
              key={tool.id}
              className={`
                flex-1 min-w-[70px] py-2.5 px-2 border-2 rounded-lg
                cursor-pointer transition-all text-xs text-center
                flex flex-col items-center justify-center
                ${currentTool === tool.id
                  ? 'border-poly-light-accent dark:border-poly-danger bg-poly-light-accent dark:bg-poly-danger text-white'
                  : 'border-poly-light-border dark:border-poly-border bg-poly-light-bg dark:bg-poly-bg text-poly-light-text dark:text-white hover:border-poly-light-accent dark:hover:border-poly-accent hover:bg-poly-light-border dark:hover:bg-poly-border'
                }
              `}
              onClick={() => onToolChange(tool.id)}
            >
              <tool.icon className="w-5 h-5 mb-1" />
              {tool.label}
            </button>
          ))}
        </div>
      </div>

      {/* Molecules */}
      <div className="flex-1 p-4 overflow-y-auto">
        <h3 className="text-poly-light-accent dark:text-poly-danger text-sm font-bold mb-3 uppercase tracking-wider">
          Molecules
        </h3>

        <div className="text-poly-light-accent dark:text-poly-accent text-xs mb-2 pb-1 border-b border-poly-light-border dark:border-poly-border">
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

        <div className="text-poly-light-accent dark:text-poly-accent text-xs mb-2 pb-1 border-b border-poly-light-border dark:border-poly-border">
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

        <div className="text-poly-light-accent dark:text-poly-accent text-xs mb-2 pb-1 border-b border-poly-light-border dark:border-poly-border">
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
import React from 'react';
import { Trash2, Undo2, Redo2, Save, Sun, Moon } from 'lucide-react';

interface ToolbarProps {
  gridSnap: boolean;
  isDark: boolean;
  onClear: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onToggleSnap: () => void;
  onOptimize: () => void;
  onPredict: () => void;
  onExport: () => void;
  onToggleTheme: () => void;
  onValidate?: () => void;
  rdkitReady?: boolean;
}

const Toolbar: React.FC<ToolbarProps> = ({
  gridSnap,
  isDark,
  onClear,
  onUndo,
  onRedo,
  onToggleSnap,
  onOptimize,
  onPredict,
  onExport,
  onToggleTheme,
  onValidate,
  rdkitReady = false
}) => {
  const buttonClass = "py-2 px-4 border-none rounded-md bg-poly-light-border dark:bg-poly-border text-poly-light-text dark:text-white cursor-pointer text-sm transition-all hover:bg-poly-light-accent hover:text-white dark:hover:bg-poly-accent";
  const dangerButtonClass = "py-2 px-4 border-none rounded-md bg-poly-light-border dark:bg-poly-border text-poly-light-text dark:text-white cursor-pointer text-sm transition-all hover:bg-poly-light-danger hover:text-white dark:hover:bg-poly-danger";

  return (
    <div className="h-[50px] bg-poly-light-sidebar dark:bg-poly-sidebar border-b-2 border-poly-light-border dark:border-poly-border flex items-center px-5 gap-4">
      <div className="flex items-center gap-2">
        <button className={`${dangerButtonClass} flex items-center gap-1.5`} onClick={onClear}>
          <Trash2 className="w-4 h-4" /> Clear
        </button>
        <button className={`${buttonClass} flex items-center gap-1.5`} onClick={onUndo}>
          <Undo2 className="w-4 h-4" /> Undo
        </button>
        <button className={`${buttonClass} flex items-center gap-1.5`} onClick={onRedo}>
          <Redo2 className="w-4 h-4" /> Redo
        </button>
      </div>

      <div className="w-px h-[30px] bg-poly-light-border dark:bg-poly-border" />

      <div className="flex items-center gap-2">
        <span className="text-poly-light-muted dark:text-gray-500 text-xs">Grid Snap:</span>
        <button className={buttonClass} onClick={onToggleSnap}>
          {gridSnap ? 'ON' : 'OFF'}
        </button>
      </div>

      <div className="w-px h-[30px] bg-poly-light-border dark:bg-poly-border" />

      <div className="flex items-center gap-2">
        <button className={buttonClass} onClick={onOptimize}>Optimize</button>
        {onValidate && (
          <button 
            className={`${buttonClass} ${!rdkitReady ? 'opacity-50 cursor-not-allowed' : ''}`} 
            onClick={onValidate}
            disabled={!rdkitReady}
            title={rdkitReady ? 'Validate SMILES with RDKit.js' : 'Chemistry engine loading...'}
          >
            Validate
          </button>
        )}
        <button className={buttonClass} onClick={onPredict}>Predict</button>
      </div>

      <div className="w-px h-[30px] bg-poly-light-border dark:bg-poly-border" />

      <div className="flex items-center gap-2">
        <button className={`${buttonClass} flex items-center gap-1.5`} onClick={onExport}>
          <Save className="w-4 h-4" /> Export
        </button>
      </div>

      <div className="w-px h-[30px] bg-poly-light-border dark:bg-poly-border" />

      {/* Theme Toggle Switch */}
      <div className="ml-auto flex items-center gap-3">
        <Sun className="w-5 h-5 text-yellow-500" />
        <button
          onClick={onToggleTheme}
          className={`
            relative w-14 h-7 rounded-full transition-all duration-300 ease-in-out
            ${isDark ? 'bg-poly-accent' : 'bg-poly-light-border'}
            focus:outline-none focus:ring-2 focus:ring-poly-accent/50
          `}
          aria-label="Toggle theme"
        >
          <span
            className={`
              absolute top-1 w-5 h-5 rounded-full bg-white shadow-md
              transition-all duration-300 ease-in-out
              ${isDark ? 'left-8' : 'left-1'}
            `}
          />
        </button>
        <Moon className="w-5 h-5 text-blue-400" />
      </div>
    </div>
  );
};

export default Toolbar;
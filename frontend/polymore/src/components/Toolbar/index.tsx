import React, { useState, useRef, useEffect } from 'react';
import { Trash2, Undo2, Redo2, Save, Sun, Moon, ChevronDown, FileJson, FileCode, Wand2, Move, Link2, Activity, PenTool } from 'lucide-react';

interface ToolbarProps {
  gridSnap: boolean;
  isDark: boolean;
  onClear: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onToggleSnap: () => void;
  onOptimize: () => void;
  onOptimizePositions?: () => void;
  onOptimizeConnections?: () => void;
  onPredict: () => void;
  onExport: () => void;
  onExportJSON?: () => void;
  onExportSMILES?: () => void;
  onToggleTheme: () => void;
  onValidate?: () => void;
  onSimulate?: () => void;
  rdkitReady?: boolean;
  simulationQueueCount?: number;
  isSimulationView?: boolean;
  onResults?: () => void;
  isResultsView?: boolean;
}

const Toolbar: React.FC<ToolbarProps> = ({
  gridSnap,
  isDark,
  onClear,
  onUndo,
  onRedo,
  onToggleSnap,
  onOptimize,
  onOptimizePositions,
  onOptimizeConnections,
  onPredict,
  onExport,
  onExportJSON,
  onExportSMILES,
  onToggleTheme,
  onValidate,
  onSimulate,
  rdkitReady = false,
  simulationQueueCount,
  isSimulationView = false,
  onResults,
  isResultsView = false
}) => {
  const [exportDropdownOpen, setExportDropdownOpen] = useState(false);
  const [optimizeDropdownOpen, setOptimizeDropdownOpen] = useState(false);
  const exportDropdownRef = useRef<HTMLDivElement>(null);
  const optimizeDropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (exportDropdownRef.current && !exportDropdownRef.current.contains(event.target as Node)) {
        setExportDropdownOpen(false);
      }
      if (optimizeDropdownRef.current && !optimizeDropdownRef.current.contains(event.target as Node)) {
        setOptimizeDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
        <span className="text-poly-light-muted dark:text-poly-muted text-xs">Grid Snap:</span>
        <button className={buttonClass} onClick={onToggleSnap}>
          {gridSnap ? 'ON' : 'OFF'}
        </button>
      </div>

      <div className="w-px h-[30px] bg-poly-light-border dark:bg-poly-border" />

      <div className="flex items-center gap-2">
        {/* Optimize dropdown */}
        <div className="relative" ref={optimizeDropdownRef}>
          <button 
            className={`${buttonClass} flex items-center gap-1.5`}
            onClick={() => setOptimizeDropdownOpen(!optimizeDropdownOpen)}
          >
            <Wand2 className="w-4 h-4" /> Optimize <ChevronDown className={`w-3 h-3 transition-transform ${optimizeDropdownOpen ? 'rotate-180' : ''}`} />
          </button>
          
          {optimizeDropdownOpen && (
            <div className="absolute top-full left-0 mt-1 w-48 bg-poly-light-sidebar dark:bg-poly-sidebar border border-poly-light-border dark:border-poly-border rounded-lg shadow-lg z-50 overflow-hidden">
              <button
                className="w-full px-3 py-2 text-left text-sm text-poly-light-text dark:text-white hover:bg-poly-light-border dark:hover:bg-poly-border flex items-center gap-2 transition-colors"
                onClick={() => {
                  onOptimize();
                  setOptimizeDropdownOpen(false);
                }}
              >
                <Wand2 className="w-4 h-4 text-purple-500" /> Optimize All
              </button>
              {onOptimizePositions && (
                <button
                  className="w-full px-3 py-2 text-left text-sm text-poly-light-text dark:text-white hover:bg-poly-light-border dark:hover:bg-poly-border flex items-center gap-2 transition-colors"
                  onClick={() => {
                    onOptimizePositions();
                    setOptimizeDropdownOpen(false);
                  }}
                >
                  <Move className="w-4 h-4 text-blue-500" /> Optimize Positions
                </button>
              )}
              {onOptimizeConnections && (
                <button
                  className="w-full px-3 py-2 text-left text-sm text-poly-light-text dark:text-white hover:bg-poly-light-border dark:hover:bg-poly-border flex items-center gap-2 transition-colors"
                  onClick={() => {
                    onOptimizeConnections();
                    setOptimizeDropdownOpen(false);
                  }}
                >
                  <Link2 className="w-4 h-4 text-emerald-500" /> Auto-Connect
                </button>
              )}
            </div>
          )}
        </div>
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
        {/* Results page toggle button */}
       
        {onSimulate && (
          <button 
            className={`${buttonClass} flex items-center gap-1.5 relative ${isSimulationView ? 'bg-poly-accent text-white' : ''}`} 
            onClick={onSimulate}
            title={isSimulationView ? 'Return to Editor' : 'Open Simulation'}
          >
            {isSimulationView ? (
              <>
                <PenTool className="w-4 h-4" /> 
                Editor
              </>
            ) : (
              <>
                <Activity className="w-4 h-4" /> 
                Simulate
                {simulationQueueCount !== undefined && simulationQueueCount > 0 && (
                  <span className="absolute -top-1 -right-1 w-4 h-4 bg-poly-accent text-white text-[10px] rounded-full flex items-center justify-center">
                    {simulationQueueCount > 9 ? '9+' : simulationQueueCount}
                  </span>
                )}
              </>
            )}
          </button>
        )}
         {onResults && (
          <button
            className={`${buttonClass} flex items-center gap-1.5 ${isResultsView ? 'bg-poly-accent text-white' : ''}`}
            onClick={onResults}
            title={isResultsView ? 'Return to Editor' : 'View Results'}
          >
            {isResultsView ? (
              <>
                <PenTool className="w-4 h-4" /> Editor
              </>
            ) : (
              <>
                <Activity className="w-4 h-4" /> Results
              </>
            )}
          </button>
        )}
      </div>

      <div className="w-px h-[30px] bg-poly-light-border dark:bg-poly-border" />

      <div className="flex items-center gap-2">
        <div className="relative" ref={exportDropdownRef}>
          <button 
            className={`${buttonClass} flex items-center gap-1.5`} 
            onClick={() => setExportDropdownOpen(!exportDropdownOpen)}
          >
            <Save className="w-4 h-4" /> Export <ChevronDown className={`w-3 h-3 transition-transform ${exportDropdownOpen ? 'rotate-180' : ''}`} />
          </button>
          
          {exportDropdownOpen && (
            <div className="absolute top-full left-0 mt-1 w-40 bg-poly-light-sidebar dark:bg-poly-sidebar border border-poly-light-border dark:border-poly-border rounded-lg shadow-lg z-50 overflow-hidden">
              <button
                className="w-full px-3 py-2 text-left text-sm text-poly-light-text dark:text-white hover:bg-poly-light-border dark:hover:bg-poly-border flex items-center gap-2 transition-colors"
                onClick={() => {
                  onExportJSON ? onExportJSON() : onExport();
                  setExportDropdownOpen(false);
                }}
              >
                <FileJson className="w-4 h-4 text-blue-500" /> As JSON
              </button>
              <button
                className="w-full px-3 py-2 text-left text-sm text-poly-light-text dark:text-white hover:bg-poly-light-border dark:hover:bg-poly-border flex items-center gap-2 transition-colors"
                onClick={() => {
                  onExportSMILES ? onExportSMILES() : onExport();
                  setExportDropdownOpen(false);
                }}
              >
                <FileCode className="w-4 h-4 text-emerald-500" /> As SMILES
              </button>
            </div>
          )}
        </div>
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
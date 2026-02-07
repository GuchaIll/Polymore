import React from 'react';

interface ToolbarProps {
  gridSnap: boolean;
  onClear: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onToggleSnap: () => void;
  onOptimize: () => void;
  onPredict: () => void;
  onExport: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({
  gridSnap,
  onClear,
  onUndo,
  onRedo,
  onToggleSnap,
  onOptimize,
  onPredict,
  onExport
}) => {
  const buttonClass = "py-2 px-4 border-none rounded-md bg-poly-border text-white cursor-pointer text-sm transition-all hover:bg-poly-accent";
  const dangerButtonClass = "py-2 px-4 border-none rounded-md bg-poly-border text-white cursor-pointer text-sm transition-all hover:bg-poly-danger";

  return (
    <div className="h-[50px] bg-poly-sidebar border-b-2 border-poly-border flex items-center px-5 gap-4">
      <div className="flex items-center gap-2">
        <button className={dangerButtonClass} onClick={onClear}>🗑️ Clear</button>
        <button className={buttonClass} onClick={onUndo}>↩️ Undo</button>
        <button className={buttonClass} onClick={onRedo}>↪️ Redo</button>
      </div>

      <div className="w-px h-[30px] bg-poly-border" />

      <div className="flex items-center gap-2">
        <span className="text-gray-500 text-xs">Grid Snap:</span>
        <button className={buttonClass} onClick={onToggleSnap}>
          {gridSnap ? 'ON' : 'OFF'}
        </button>
      </div>

      <div className="w-px h-[30px] bg-poly-border" />

      <div className="flex items-center gap-2">
        <button className={buttonClass} onClick={onOptimize}>⚡ Optimize</button>
        <button className={buttonClass} onClick={onPredict}>🔬 Predict</button>
      </div>

      <div className="w-px h-[30px] bg-poly-border" />

      <div className="flex items-center gap-2">
        <button className={buttonClass} onClick={onExport}>💾 Export</button>
      </div>
    </div>
  );
};

export default Toolbar;
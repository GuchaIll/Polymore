import {useState, useCallback} from 'react';
import { PolyForgeState, Molecule, PlacedMolecule, Position, Toast, ToolType, ViewMode, PredictedProperties } from '../types';
import { 
  optimizeStructure as runOptimization, 
  getOptimizationSuggestions,
  OptimizationSuggestion
} from '../util/optimization';
import { PolymerValidationResult } from '../util';
import { useTierOneAnalysis } from './useTierOneAnalysis';


// Grid boundary constants (grid is 20x20 centered at origin)
const GRID_HALF_SIZE = 10;

const initialState: PolyForgeState = {
  tool: 'select',
  gridSnap: true,
  gridSize: 2,
  placedMolecules: [],
  selectedMolecule: null,
  selectedObject: null,
  connectStart: null,
  movingMoleculeId: null,
  history: [],
  historyIndex: -1,
  viewMode: 'both'
};

export function usePolyForgeState() {
  const [state, setState] = useState<PolyForgeState>(initialState);
  const [toast, setToast] = useState<Toast>({ message: '', visible: false });
  const tierOneAnalysis = useTierOneAnalysis();

  const showToast = useCallback((message: string) => {
    setToast({ message, visible: true });
    setTimeout(() => setToast({ message: '', visible: false }), 2500);
  }, []);

  const setTool = useCallback((tool: ToolType) => {
    setState(prev => ({ ...prev, tool, connectStart: null, movingMoleculeId: null }));
    const toolNames: Record<ToolType, string> = {
      select: 'Select tool - Click to select molecules',
      add: 'Add tool - Click to place selected molecule',
      remove: 'Remove tool - Click molecules to delete',
      connect: 'Bond tool - Click two molecules to connect',
      move: 'Move tool - Click molecule to move, click again to place'
    };
    showToast(toolNames[tool]);
  }, [showToast]);

  // Check if position is within grid boundaries
  const isWithinGrid = useCallback((position: Position): boolean => {
    return Math.abs(position.x) <= GRID_HALF_SIZE && Math.abs(position.z) <= GRID_HALF_SIZE;
  }, []);

  // Clamp position to grid boundaries
  const clampToGrid = useCallback((position: Position): Position => {
    return {
      x: Math.max(-GRID_HALF_SIZE, Math.min(GRID_HALF_SIZE, position.x)),
      y: position.y,
      z: Math.max(-GRID_HALF_SIZE, Math.min(GRID_HALF_SIZE, position.z))
    };
  }, []);

  // Start moving a molecule
  const startMove = useCallback((id: number) => {
    const mol = state.placedMolecules.find(m => m.id === id);
    if (mol) {
      setState(prev => ({ ...prev, movingMoleculeId: id }));
      showToast(`Moving ${mol.name} - click to place`);
    }
  }, [state.placedMolecules, showToast]);

  // Cancel move operation
  const cancelMove = useCallback(() => {
    setState(prev => ({ ...prev, movingMoleculeId: null }));
  }, []);

  // Move molecule to new position (live preview during mouse move)
  const updateMovePosition = useCallback((position: Position) => {
    setState(prev => {
      if (!prev.movingMoleculeId) return prev;

      const clampedPosition = clampToGrid(position);
      const snappedPosition: Position = prev.gridSnap
        ? {
            x: Math.round(clampedPosition.x / prev.gridSize) * prev.gridSize,
            y: 0,
            z: Math.round(clampedPosition.z / prev.gridSize) * prev.gridSize
          }
        : { ...clampedPosition, y: 0 };

      const newMolecules = prev.placedMolecules.map(m =>
        m.id === prev.movingMoleculeId
          ? { ...m, position: snappedPosition }
          : m
      );

      return { ...prev, placedMolecules: newMolecules };
    });
  }, [clampToGrid]);

  // Note: confirmMove is defined after saveToHistory below

  const setSelectedMolecule = useCallback((molecule: Molecule) => {
    setState(prev => ({ ...prev, selectedMolecule: molecule, tool: 'add' }));
    showToast(`Selected ${molecule.name} - Click on canvas to place`);
  }, [showToast]);

  const toggleSnap = useCallback(() => {
    setState(prev => {
      const newSnap = !prev.gridSnap;
      showToast('Grid snap: ' + (newSnap ? 'ON' : 'OFF'));
      return { ...prev, gridSnap: newSnap };
    });
  }, [showToast]);

  const setViewMode = useCallback((mode: ViewMode) => {
    setState(prev => ({ ...prev, viewMode: mode }));
  }, []);

  const saveToHistory = useCallback((molecules: PlacedMolecule[]) => {
    setState(prev => {
      const newHistory = prev.history.slice(0, prev.historyIndex + 1);
      newHistory.push(JSON.parse(JSON.stringify(molecules)));
      const newIndex = newHistory.length - 1;

      if (newHistory.length > 50) {
        newHistory.shift();
        return { ...prev, history: newHistory, historyIndex: newIndex - 1 };
      }

      return { ...prev, history: newHistory, historyIndex: newIndex };
    });
  }, []);

  // Confirm molecule placement after move
  const confirmMove = useCallback(() => {
    setState(prev => {
      if (!prev.movingMoleculeId) return prev;
      
      const mol = prev.placedMolecules.find(m => m.id === prev.movingMoleculeId);
      if (mol) {
        saveToHistory(prev.placedMolecules);
        showToast(`Placed ${mol.name}`);
      }
      
      return { ...prev, movingMoleculeId: null };
    });
  }, [saveToHistory, showToast]);

  const placeMolecule = useCallback((mol: Molecule, position: Position) => {
    setState(prev => {
      // Clamp position to grid boundaries
      const clampedX = Math.max(-GRID_HALF_SIZE, Math.min(GRID_HALF_SIZE, position.x));
      const clampedZ = Math.max(-GRID_HALF_SIZE, Math.min(GRID_HALF_SIZE, position.z));
      
      const snappedPosition: Position = prev.gridSnap
        ? {
            x: Math.round(clampedX / prev.gridSize) * prev.gridSize,
            y: 0,
            z: Math.round(clampedZ / prev.gridSize) * prev.gridSize
          }
        : { x: clampedX, y: 0, z: clampedZ };

      // Spread all molecule properties including extended fields
      const moleculeData: PlacedMolecule = {
        ...mol,
        id: Date.now(),
        position: snappedPosition,
        connections: []
      };

      const newMolecules = [...prev.placedMolecules, moleculeData];
      saveToHistory(newMolecules);
      showToast(`Added ${mol.name}`);

      return { ...prev, placedMolecules: newMolecules };
    });
  }, [saveToHistory, showToast]);

  const removeMolecule = useCallback((id: number) => {
    setState(prev => {
      const mol = prev.placedMolecules.find(m => m.id === id);
      if (!mol) return prev;

      const newMolecules = prev.placedMolecules
        .filter(m => m.id !== id)
        .map(m => ({
          ...m,
          connections: m.connections.filter(c => c !== id)
        }));

      saveToHistory(newMolecules);
      showToast(`Removed ${mol.name}`);

      return { ...prev, placedMolecules: newMolecules };
    });
  }, [saveToHistory, showToast]);

  const connectMolecules = useCallback((id1: number, id2: number) => {
    setState(prev => {
      const mol1 = prev.placedMolecules.find(m => m.id === id1);
      const mol2 = prev.placedMolecules.find(m => m.id === id2);

      if (!mol1 || !mol2) return prev;
      if (mol1.connections.includes(id2)) {
        showToast('Already connected!');
        return prev;
      }

      const newMolecules = prev.placedMolecules.map(m => {
        if (m.id === id1) {
          return { ...m, connections: [...m.connections, id2] };
        }
        if (m.id === id2) {
          return { ...m, connections: [...m.connections, id1] };
        }
        return m;
      });

      saveToHistory(newMolecules);
      showToast(`Connected ${mol1.name} to ${mol2.name}`);

      return { ...prev, placedMolecules: newMolecules, connectStart: null };
    });
  }, [saveToHistory, showToast]);

  const setConnectStart = useCallback((id: number) => {
    setState(prev => ({ ...prev, connectStart: id }));
    showToast('Now click another molecule to connect');
  }, [showToast]);

  const setSelectedObject = useCallback((id: number) => {
    setState(prev => {
      const mol = prev.placedMolecules.find(m => m.id === id);
      if (mol) showToast(`Selected: ${mol.name}`);
      return { ...prev, selectedObject: id };
    });
  }, [showToast]);

  const clearCanvas = useCallback(() => {
    if (state.placedMolecules.length === 0) return;

    setState(prev => {
      saveToHistory([]);
      showToast('Canvas cleared');
      return { ...prev, placedMolecules: [] };
    });
  }, [state.placedMolecules.length, saveToHistory, showToast]);

  const importMolecules = useCallback((moleculesToImport: PlacedMolecule[]) => {
    if (moleculesToImport.length === 0) return;

    setState(prev => {
      // Offset new molecules to avoid overlap with existing ones
      const maxId = prev.placedMolecules.reduce((max, m) => Math.max(max, m.id), 0);
      const offsetX = prev.placedMolecules.length > 0 ? 10 : 0;
      
      const newMolecules = moleculesToImport.map((mol, idx) => ({
        ...mol,
        id: maxId + mol.id + 1,
        position: {
          x: mol.position.x + offsetX,
          y: mol.position.y,
          z: mol.position.z
        },
        connections: mol.connections.map(connId => maxId + connId + 1)
      }));
      
      const allMolecules = [...prev.placedMolecules, ...newMolecules];
      saveToHistory(allMolecules);
      return { ...prev, placedMolecules: allMolecules };
    });
  }, [saveToHistory]);

  const undoAction = useCallback(() => {
    setState(prev => {
      if (prev.historyIndex <= 0) {
        showToast('Nothing to undo');
        return prev;
      }

      const newIndex = prev.historyIndex - 1;
      const restoredMolecules = JSON.parse(JSON.stringify(prev.history[newIndex]));
      showToast('Undone');

      return { ...prev, placedMolecules: restoredMolecules, historyIndex: newIndex };
    });
  }, [showToast]);

  const redoAction = useCallback(() => {
    setState(prev => {
      if (prev.historyIndex >= prev.history.length - 1) {
        showToast('Nothing to redo');
        return prev;
      }

      const newIndex = prev.historyIndex + 1;
      const restoredMolecules = JSON.parse(JSON.stringify(prev.history[newIndex]));
      showToast('Redone');

      return { ...prev, placedMolecules: restoredMolecules, historyIndex: newIndex };
    });
  }, [showToast]);

  const exportAsJSON = useCallback(() => {
    const data = {
      molecules: state.placedMolecules,
      timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'polymer_structure.json';
    a.click();
    URL.revokeObjectURL(url);

    showToast('Exported as JSON!');
  }, [state.placedMolecules, showToast]);

  const exportAsSMILES = useCallback(() => {
    if (state.placedMolecules.length === 0) {
      showToast('No molecules to export!');
      return;
    }

    // Generate standard .smi format: SMILES [tab] NAME per line
    // This format is compatible with RDKit, OpenBabel, and ML pipelines
    const lines = state.placedMolecules.map(m => `${m.smiles}\t${m.name}`);
    
    // Also add the combined polymer SMILES as first entry
    const combinedSMILES = state.placedMolecules.map(m => m.smiles).join('.');
    const polymerName = `Polymer_${state.placedMolecules.length}units`;
    
    const content = `${combinedSMILES}\t${polymerName}\n${lines.join('\n')}`;

    const blob = new Blob([content], { type: 'chemical/x-daylight-smiles' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'polymer_structure.smi';
    a.click();
    URL.revokeObjectURL(url);

    showToast('Exported as SMILES!');
  }, [state.placedMolecules, showToast]);

  // Legacy export function that defaults to JSON
  const exportStructure = useCallback(() => {
    exportAsJSON();
  }, [exportAsJSON]);

  // Full optimization with all steps
  const optimizeStructure = useCallback((validationResult?: PolymerValidationResult) => {
    if (state.placedMolecules.length === 0) {
      showToast('Add molecules first!');
      return;
    }

    const result = runOptimization(state.placedMolecules, {
      autoConnect: false,
      validationResult
    });

    if (result.changes.length > 0) {
      setState(prev => {
        saveToHistory(result.molecules);
        return { ...prev, placedMolecules: result.molecules };
      });
      showToast(result.summary);
    } else {
      showToast('Structure already optimized');
    }
  }, [state.placedMolecules, saveToHistory, showToast]);

  // Optimize positions only (bond lengths and spacing)
  const optimizePositions = useCallback(() => {
    if (state.placedMolecules.length === 0) {
      showToast('Add molecules first!');
      return;
    }

    const result = runOptimization(state.placedMolecules, { autoConnect: false });
    
    if (result.changes.filter(c => c.type === 'position').length > 0) {
      setState(prev => {
        saveToHistory(result.molecules);
        return { ...prev, placedMolecules: result.molecules };
      });
      showToast('Positions optimized');
    } else {
      showToast('Positions already optimal');
    }
  }, [state.placedMolecules, saveToHistory, showToast]);

  // Auto-connect disconnected fragments
  const optimizeConnections = useCallback(() => {
    if (state.placedMolecules.length < 2) {
      showToast('Need at least 2 molecules');
      return;
    }

    const result = runOptimization(state.placedMolecules, { autoConnect: true });
    
    if (result.changes.filter(c => c.type === 'connection').length > 0) {
      setState(prev => {
        saveToHistory(result.molecules);
        return { ...prev, placedMolecules: result.molecules };
      });
      showToast('Fragments connected');
    } else {
      showToast('All molecules already connected');
    }
  }, [state.placedMolecules, saveToHistory, showToast]);

  // Get optimization suggestions
  const getSuggestions = useCallback((validationResult?: PolymerValidationResult): OptimizationSuggestion[] => {
    return getOptimizationSuggestions(state.placedMolecules, validationResult);
  }, [state.placedMolecules]);

  const predictProperties = useCallback(async (): Promise<PredictedProperties | null> => {
    if (state.placedMolecules.length === 0) {
      showToast('Add some molecules first!');
      return null;
    }

    const smiles = state.placedMolecules.map(m => m.smiles).join('.');
    showToast('Predicting properties...');

    try {
      // Call backend API via useTierOneAnalysis hook
      const result = await tierOneAnalysis.analyze(smiles);
      
      if (result) {
        // Backend returns 0-10 scale, multiply by 10 to get 0-100 for display
        const normalize = (val: number | undefined) => {
          if (val === undefined || val === null) return 0;
          return val * 10;
        };
        
        const properties: PredictedProperties = {
          strength: Math.round(normalize(result.strength) * 10) / 10,
          flexibility: Math.round(normalize(result.flexibility) * 10) / 10,
          degradability: Math.round(normalize(result.degradability) * 10) / 10,
          sustainability: Math.round(normalize(result.sustainability) * 10) / 10
        };
        
        showToast('Properties predicted!');
        return properties;
      } else {
        showToast('No prediction data returned');
        return null;
      }
    } catch (error: any) {
      console.error('Prediction error:', error);
      showToast(error.message || 'Prediction failed');
      return null;
    }
  }, [state.placedMolecules, showToast, tierOneAnalysis]);

  return {
    state,
    toast,
    showToast,
    setTool,
    setSelectedMolecule,
    toggleSnap,
    setViewMode,
    placeMolecule,
    removeMolecule,
    connectMolecules,
    setConnectStart,
    setSelectedObject,
    clearCanvas,
    importMolecules,
    undoAction,
    redoAction,
    exportStructure,
    exportAsJSON,
    exportAsSMILES,
    optimizeStructure,
    optimizePositions,
    optimizeConnections,
    getSuggestions,
    predictProperties,
    // Move tool functions
    startMove,
    cancelMove,
    updateMovePosition,
    confirmMove,
    isWithinGrid
  };
}
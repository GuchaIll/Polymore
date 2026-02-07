import {useState, useCallback} from 'react';
import { PolyForgeState, Molecule, PlacedMolecule, Position, Toast, ToolType, ViewMode, PredictedProperties } from '../types';


const initialState: PolyForgeState = {
  tool: 'select',
  gridSnap: true,
  gridSize: 2,
  placedMolecules: [],
  selectedMolecule: null,
  selectedObject: null,
  connectStart: null,
  history: [],
  historyIndex: -1,
  viewMode: 'both'
};

export function usePolyForgeState() {
  const [state, setState] = useState<PolyForgeState>(initialState);
  const [toast, setToast] = useState<Toast>({ message: '', visible: false });

  const showToast = useCallback((message: string) => {
    setToast({ message, visible: true });
    setTimeout(() => setToast({ message: '', visible: false }), 2500);
  }, []);

  const setTool = useCallback((tool: ToolType) => {
    setState(prev => ({ ...prev, tool, connectStart: null }));
    const toolNames: Record<ToolType, string> = {
      select: 'Select tool - Click to select molecules',
      add: 'Add tool - Click to place selected molecule',
      remove: 'Remove tool - Click molecules to delete',
      connect: 'Bond tool - Click two molecules to connect'
    };
    showToast(toolNames[tool]);
  }, [showToast]);

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

  const placeMolecule = useCallback((mol: Molecule, position: Position) => {
    setState(prev => {
      const snappedPosition: Position = prev.gridSnap
        ? {
            x: Math.round(position.x / prev.gridSize) * prev.gridSize,
            y: 0,
            z: Math.round(position.z / prev.gridSize) * prev.gridSize
          }
        : { ...position, y: 0 };

      const moleculeData: PlacedMolecule = {
        id: Date.now(),
        name: mol.name,
        formula: mol.formula,
        smiles: mol.smiles,
        icon: mol.icon,
        weight: mol.weight,
        color: mol.color,
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

  const exportStructure = useCallback(() => {
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

    showToast('Structure exported!');
  }, [state.placedMolecules, showToast]);

  const optimizeStructure = useCallback(() => {
    showToast('Optimizing structure... (simulated)');
  }, [showToast]);

  const predictProperties = useCallback(async (): Promise<PredictedProperties | null> => {
    if (state.placedMolecules.length === 0) {
      showToast('Add some molecules first!');
      return null;
    }

    const smiles = state.placedMolecules.map(m => m.smiles).join('.');
    showToast('Predicting properties...');

    // Helper function to generate random properties based on molecule composition
    const generateRandomProperties = (): PredictedProperties => {
      // Use molecule data to influence the random values for more realistic simulation
      const totalWeight = state.placedMolecules.reduce((sum, m) => sum + m.weight, 0);
      const moleculeCount = state.placedMolecules.length;
      const bondCount = state.placedMolecules.reduce((sum, m) => sum + m.connections.length, 0) / 2;

      // Base values influenced by structure
      const weightFactor = Math.min(totalWeight / 500, 1); // Normalize weight influence
      const bondFactor = bondCount > 0 ? Math.min(bondCount / moleculeCount, 1) : 0;

      // Generate properties with some randomness but influenced by structure
      const strength = Math.min(100, Math.max(10,
        30 + (weightFactor * 40) + (bondFactor * 20) + (Math.random() * 20 - 10)
      ));

      const flexibility = Math.min(100, Math.max(10,
        50 - (bondFactor * 30) + (Math.random() * 30 - 15)
      ));

      const degradability = Math.min(100, Math.max(10,
        60 - (weightFactor * 30) + (Math.random() * 30 - 15)
      ));

      const sustainability = Math.min(100, Math.max(10,
        40 + (degradability * 0.3) + (Math.random() * 20 - 10)
      ));

      return {
        strength: Math.round(strength * 10) / 10,
        flexibility: Math.round(flexibility * 10) / 10,
        degradability: Math.round(degradability * 10) / 10,
        sustainability: Math.round(sustainability * 10) / 10
      };
    };

    try {
      // Try to call backend API first
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles })
      });

      const data = await response.json();

      if (response.ok) {
        showToast('Properties predicted!');
        return data.properties;
      } else {
        throw new Error(data.error);
      }
    } catch (error) {
      // Fallback to random generation if backend is not available
      console.log('Backend not available, using simulated properties');

      // Simulate a small delay like a real API call
      await new Promise(resolve => setTimeout(resolve, 500));

      const properties = generateRandomProperties();
      showToast('Properties predicted! (simulated)');
      return properties;
    }
  }, [state.placedMolecules, showToast]);

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
    undoAction,
    redoAction,
    exportStructure,
    optimizeStructure,
    predictProperties
  };
}
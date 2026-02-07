import React, { useState, useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import Sidebar from '../../components/Sidebar';
import Toolbar from '../../components/Toolbar';
import Canvas3D from '../../components/Canvas3D';
import PropertiesPanel from '../../components/PropertiesPanel';
import ValidationErrorPopup from '../../components/ValidationErrorPopup';
import SimulationPage from '../SimulationPage';
import ResultsPage from '../ResultsPage';
import { usePolyForgeState } from '../../hooks/usePolyForgeState';
import { useTheme } from '../../hooks/useTheme';
import { Molecule, PredictedProperties, SimulationTask, SimulationResult } from '../../types';
import { 
  validatePolymerComprehensive,
  predictPropertiesFromBackend,
  parseSmilestoMolecules,
  PolymerValidationResult,
  repairSmiles,
  SmilesRepairResult,
  generateSmiles
} from '../../util';

interface PolyForgeProps {
  rdkitReady: boolean;
  rdkitError: string | null;
}

const PolyForge: React.FC<PolyForgeProps> = ({ rdkitReady, rdkitError }) => {
  const {
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
    predictProperties,
    // Move tool functions
    startMove,
    updateMovePosition,
    confirmMove
  } = usePolyForgeState();

  const [draggedMolecule, setDraggedMolecule] = useState<Molecule | null>(null);
  const [predictedProperties, setPredictedProperties] = useState<PredictedProperties | null>(null);
  const [validationResult, setValidationResult] = useState<PolymerValidationResult | null>(null);
  // Results page state
  const [resultsPageOpen, setResultsPageOpen] = useState(false);
  const [showValidationPopup, setShowValidationPopup] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [repairResult, setRepairResult] = useState<SmilesRepairResult | null>(null);
  const [isRepairing, setIsRepairing] = useState(false);
  const { isDark, toggleTheme } = useTheme();

  // Simulation Queue State
  const [simulationPageOpen, setSimulationPageOpen] = useState(false);
  const [simulationQueue, setSimulationQueue] = useState<SimulationTask[]>([]);
  const [runningTask, setRunningTask] = useState<SimulationTask | null>(null);
  const [completedTasks, setCompletedTasks] = useState<SimulationTask[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [currentSmiles, setCurrentSmiles] = useState('');
  const simulationIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pendingQueueAddRef = useRef<Set<string>>(new Set());

  const handleDragStart = useCallback((molecule: Molecule) => {
    setDraggedMolecule(molecule);
  }, []);

  const handleDragEnd = useCallback(() => {
    setDraggedMolecule(null);
  }, []);

  const handleDrop = useCallback((molecule: Molecule, position: { x: number; y: number; z: number }) => {
    placeMolecule(molecule, position);
    setDraggedMolecule(null);
  }, [placeMolecule]);

  const handlePlaneClick = useCallback((intersection: THREE.Vector3) => {
    // If we're in move mode and have a molecule selected, confirm the move
    if (state.movingMoleculeId !== null) {
      confirmMove();
      return;
    }
    
    // Otherwise, place a new molecule in add mode
    if (state.tool === 'add' && state.selectedMolecule) {
      placeMolecule(state.selectedMolecule, {
        x: intersection.x,
        y: intersection.y,
        z: intersection.z
      });
    }
  }, [state.tool, state.selectedMolecule, state.movingMoleculeId, placeMolecule, confirmMove]);

  // Handle pointer movement for move tool preview
  const handlePointerMove = useCallback((intersection: THREE.Vector3) => {
    if (state.movingMoleculeId !== null) {
      updateMovePosition({
        x: intersection.x,
        y: 0,
        z: intersection.z
      });
    }
  }, [state.movingMoleculeId, updateMovePosition]);

  const handleMoleculeClick = useCallback((id: number) => {
    switch (state.tool) {
      case 'select':
        setSelectedObject(id);
        break;
      case 'remove':
        removeMolecule(id);
        break;
      case 'connect':
        if (!state.connectStart) {
          setConnectStart(id);
        } else if (state.connectStart !== id) {
          connectMolecules(state.connectStart, id);
        }
        break;
      case 'move':
        // If not currently moving, start moving this molecule
        if (state.movingMoleculeId === null) {
          startMove(id);
        } else if (state.movingMoleculeId === id) {
          // Clicking the same molecule confirms its placement
          confirmMove();
        } else {
          // Clicking a different molecule switches to moving that one instead
          confirmMove();
          startMove(id);
        }
        break;
    }
  }, [state.tool, state.connectStart, state.movingMoleculeId, setSelectedObject, removeMolecule, setConnectStart, connectMolecules, startMove, confirmMove]);

  // Validate polymer configuration using comprehensive RDKit.js validation
  const handleValidate = useCallback(async () => {
    if (!rdkitReady) {
      showToast('Chemistry engine loading...');
      return;
    }
    
    if (state.placedMolecules.length === 0) {
      showToast('No molecules to validate');
      return;
    }

    setIsLoading(true);
    setRepairResult(null); // Clear previous repair results
    
    try {
      const result = await validatePolymerComprehensive(state.placedMolecules);
      setValidationResult(result);
      
      if (result.isValid) {
        showToast(`Valid ${result.polymerType} polymer: ${result.canonicalSmiles}`);
      } else {
        // Auto-trigger repair attempt when validation fails
        try {
          const smiles = await generateSmiles(state.placedMolecules);
          if (smiles) {
            const repair = await repairSmiles(smiles);
            setRepairResult(repair);
            
            if (repair.success && repair.wasModified) {
              showToast('Auto-repair found fixes - see suggestions');
            }
          }
        } catch {
          // Repair failed silently, still show validation popup
        }
        
        // Show popup with detailed errors
        setShowValidationPopup(true);
      }
    } catch (error) {
      showToast(`Validation error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [rdkitReady, state.placedMolecules, showToast]);

  const handlePredict = useCallback(async () => {
    if (!rdkitReady) {
      showToast('Chemistry engine loading...');
      return;
    }

    if (state.placedMolecules.length === 0) {
      showToast('No molecules to predict');
      return;
    }

    setIsLoading(true);
    setRepairResult(null); // Clear previous repair results
    
    try {
      // Comprehensive validation first
      const result = await validatePolymerComprehensive(state.placedMolecules);
      setValidationResult(result);
      
      if (!result.isValid) {
        // Auto-trigger repair attempt when validation fails
        try {
          const smiles = await generateSmiles(state.placedMolecules);
          if (smiles) {
            const repair = await repairSmiles(smiles);
            setRepairResult(repair);
            
            if (repair.success && repair.wasModified) {
              showToast('Auto-repair found fixes - see suggestions');
            }
          }
        } catch {
          // Repair failed silently
        }
        
        // Show popup with validation errors
        setShowValidationPopup(true);
        return;
      }
      
      // Use canonical SMILES for prediction
      const smilesForPrediction = result.canonicalSmiles || result.smiles;
      
      if (!smilesForPrediction) {
        showToast('Could not generate SMILES for prediction');
        return;
      }
      
      // Call backend API for property prediction
      showToast('Predicting properties...');
      const predictionResult = await predictPropertiesFromBackend(smilesForPrediction);
      
      if (predictionResult.success && predictionResult.properties) {
        // Map backend properties to our PredictedProperties format
        const props: PredictedProperties = {
          strength: (predictionResult.properties.strength ?? 0.8) * 100,
          flexibility: (predictionResult.properties.flexibility ?? 0.6) * 100,
          degradability: (predictionResult.properties.degradability ?? 0.4) * 100,
          sustainability: (predictionResult.properties.sustainability ?? 0.55) * 100,
        };
        setPredictedProperties(props);
        showToast('Properties predicted successfully!');
      } else {
        // Fallback to local prediction if backend fails
        showToast(predictionResult.error || 'Backend unavailable, using local prediction');
        const props = await predictProperties();
        if (props) {
          setPredictedProperties(props);
        }
      }
    } catch (error) {
      showToast(`Prediction error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [rdkitReady, state.placedMolecules, predictProperties, showToast]);

  // Manual auto-repair trigger from validation popup
  const handleAutoRepair = useCallback(async () => {
    if (state.placedMolecules.length === 0) return;
    
    setIsRepairing(true);
    try {
      const smiles = await generateSmiles(state.placedMolecules);
      if (smiles) {
        const repair = await repairSmiles(smiles);
        setRepairResult(repair);
        
        if (repair.success) {
          if (repair.wasModified) {
            showToast(`SMILES repaired: ${repair.repairSteps.join(', ')}`);
          } else {
            showToast('SMILES is already valid');
          }
        } else {
          showToast(repair.error || 'Could not repair SMILES');
        }
      }
    } catch (error) {
      showToast(`Repair error: ${error}`);
    } finally {
      setIsRepairing(false);
    }
  }, [state.placedMolecules, showToast]);

  // Update current SMILES when molecules change
  useEffect(() => {
    const updateSmiles = async () => {
      if (state.placedMolecules.length > 0) {
        try {
          const smiles = await generateSmiles(state.placedMolecules);
          setCurrentSmiles(smiles || '');
        } catch {
          setCurrentSmiles('');
        }
      } else {
        setCurrentSmiles('');
      }
    };
    updateSmiles();
  }, [state.placedMolecules]);

  // Add task to simulation queue (validates for duplicates)
  const handleAddToQueue = useCallback((smiles: string, name: string): boolean => {
    // Check synchronously if already pending addition (prevents race condition)
    if (pendingQueueAddRef.current.has(smiles)) {
      return false;
    }

    // Check if SMILES already in queue or running
    const allSmiles = [
      ...simulationQueue.map(t => t.smiles),
      runningTask?.smiles,
      ...completedTasks.map(t => t.smiles)
    ].filter(Boolean);

    if (allSmiles.includes(smiles)) {
      return false; // Duplicate
    }

    // Mark as pending synchronously before async state update
    pendingQueueAddRef.current.add(smiles);

    const newTask: SimulationTask = {
      id: `sim-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      smiles,
      name,
      status: 'pending',
      createdAt: new Date(),
      progress: 0
    };

    setSimulationQueue(prev => {
      // Clear from pending ref once state is updated
      pendingQueueAddRef.current.delete(smiles);
      return [...prev, newTask];
    });
    showToast('Added to simulation queue');
    return true;
  }, [simulationQueue, runningTask, completedTasks, showToast]);

  // Remove task from queue
  const handleRemoveFromQueue = useCallback((taskId: string) => {
    setSimulationQueue(prev => prev.filter(t => t.id !== taskId));
    setCompletedTasks(prev => prev.filter(t => t.id !== taskId));
  }, []);

  // Clear all completed tasks
  const handleClearCompleted = useCallback(() => {
    setCompletedTasks([]);
  }, []);

  // Toggle pause state
  const handleTogglePause = useCallback(() => {
    setIsPaused(prev => !prev);
  }, []);

  // Mock simulation runner - processes queue one at a time
  useEffect(() => {
    // If paused, running, or empty queue, do nothing
    if (isPaused || runningTask || simulationQueue.length === 0) {
      return;
    }

    // Start processing the first task in queue
    const nextTask = simulationQueue[0];
    setSimulationQueue(prev => prev.slice(1));
    setRunningTask({ ...nextTask, status: 'running', startedAt: new Date() });
  }, [isPaused, runningTask, simulationQueue]);

  // Simulate progress for running task (mock simulation)
  useEffect(() => {
    if (!runningTask || isPaused) {
      if (simulationIntervalRef.current) {
        clearInterval(simulationIntervalRef.current);
        simulationIntervalRef.current = null;
      }
      return;
    }

    simulationIntervalRef.current = setInterval(() => {
      setRunningTask(prev => {
        if (!prev) return null;

        const newProgress = Math.min(prev.progress + Math.random() * 15 + 5, 100);
        
        if (newProgress >= 100) {
          // First transition to processing status
          if (prev.status === 'running') {
            return { ...prev, status: 'processing', progress: 100 };
          }
          
          // Then complete with results (simulates API response)
          const completedTask: SimulationTask = {
            ...prev,
            status: 'completed',
            progress: 100,
            completedAt: new Date(),
            result: {
              predictedProperties: {
                strength: Math.random() * 0.4 + 0.5,
                flexibility: Math.random() * 0.4 + 0.3,
                degradability: Math.random() * 0.5 + 0.2,
                sustainability: Math.random() * 0.4 + 0.4
              },
              simulationTime: Date.now() - prev.startedAt!.getTime(),
              iterations: Math.floor(Math.random() * 500 + 200),
              convergenceScore: Math.random() * 0.2 + 0.8
            } as SimulationResult
          };

          setCompletedTasks(prevCompleted => [completedTask, ...prevCompleted]);
          return null; // Clear running task
        }

        return { ...prev, progress: Math.round(newProgress) };
      });
    }, 500);

    return () => {
      if (simulationIntervalRef.current) {
        clearInterval(simulationIntervalRef.current);
        simulationIntervalRef.current = null;
      }
    };
  }, [runningTask, isPaused]);

  const handleImportSmiles = useCallback(async (smiles: string) => {
    if (!rdkitReady) {
      showToast('Chemistry engine loading...');
      return;
    }

    setIsLoading(true);
    try {
      const result = await parseSmilestoMolecules(smiles);
      
      if (result.success && result.molecules.length > 0) {
        importMolecules(result.molecules);
        showToast(`Imported ${result.atomCount} atoms with ${result.bondCount} bonds`);
      } else {
        showToast(result.error || 'Failed to parse SMILES');
      }
    } catch (error) {
      showToast(`Import error: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [rdkitReady, importMolecules, showToast]);

  const handleClear = useCallback(() => {
    if (state.placedMolecules.length === 0) return;
    if (window.confirm('Clear all molecules?')) {
      clearCanvas();
      setPredictedProperties(null);
      setValidationResult(null);
      setShowValidationPopup(false);
    }
  }, [state.placedMolecules.length, clearCanvas]);

  const handleResetCamera = useCallback(() => {
    showToast('Camera reset');
  }, [showToast]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-poly-light-bg dark:bg-poly-bg">
      {/* RDKit.js loading indicator */}
      {!rdkitReady && !rdkitError && (
        <div className="absolute top-16 left-1/2 transform -translate-x-1/2 z-50 
                        bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg text-sm">
          Loading chemistry engine...
        </div>
      )}
      
      {/* RDKit.js error indicator */}
      {rdkitError && (
        <div className="absolute top-16 left-1/2 transform -translate-x-1/2 z-50 
                        bg-amber-500 text-white px-4 py-2 rounded-lg shadow-lg text-sm">
          {rdkitError}
        </div>
      )}

      <Sidebar
        currentTool={state.tool}
        onToolChange={setTool}
        onMoleculeSelect={setSelectedMolecule}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        onImportSmiles={handleImportSmiles}
        rdkitReady={rdkitReady}
      />

      <div className="flex-1 flex flex-col relative">
        {!simulationPageOpen && !resultsPageOpen && (
          <Toolbar
            gridSnap={state.gridSnap}
            isDark={isDark}
            onClear={handleClear}
            onUndo={undoAction}
            onRedo={redoAction}
            onToggleSnap={toggleSnap}
            onOptimize={optimizeStructure}
            onOptimizePositions={optimizePositions}
            onOptimizeConnections={optimizeConnections}
            onPredict={handlePredict}
            onExport={exportStructure}
            onExportJSON={exportAsJSON}
            onExportSMILES={exportAsSMILES}
            onToggleTheme={toggleTheme}
            onValidate={handleValidate}
            onSimulate={() => {
              setSimulationPageOpen(prev => !prev);
              setResultsPageOpen(false);
            }}
            rdkitReady={rdkitReady}
            simulationQueueCount={simulationQueue.length + (runningTask ? 1 : 0)}
            isSimulationView={simulationPageOpen}
            onResults={() => {
              setResultsPageOpen(prev => !prev);
              setSimulationPageOpen(false);
            }}
            isResultsView={resultsPageOpen}
          />
        )}

        {/* Conditionally render Editor, Simulation, or Results Page */}
        {simulationPageOpen ? (
          <SimulationPage
            isOpen={simulationPageOpen}
            currentSmiles={currentSmiles}
            currentName={state.placedMolecules.length > 0 
              ? `Polymer (${state.placedMolecules.length} units)` 
              : 'Unnamed'}
            queue={simulationQueue}
            runningTask={runningTask}
            completedTasks={completedTasks}
            onAddToQueue={handleAddToQueue}
            onRemoveFromQueue={handleRemoveFromQueue}
            onClearCompleted={handleClearCompleted}
            isPaused={isPaused}
            onTogglePause={handleTogglePause}
            onAutoQueueAttempted={(_success: boolean, message: string) => showToast(message)}
            onClose={() => setSimulationPageOpen(false)}
          />
        ) : resultsPageOpen ? (
          <>
            {/* ResultsPage integration */}
            {/* TODO: Map predictedProperties and mock applications to ResultsPage props */}
            <ResultsPage
              onClose={() => setResultsPageOpen(false)}
              properties={{
                strength: predictedProperties?.strength ?? 80,
                elasticity: 0,
                thermal: 0,
                flexibility: predictedProperties?.flexibility ?? 60,
                ecoScore: 0,
                biodegradable: 0,
                degradability: predictedProperties?.degradability ?? 40,
                sustainability: predictedProperties?.sustainability ?? 55,
              }}
              applications={[
                {
                  name: 'Biodegradable Packaging',
                  description: 'Eco-friendly packaging for food and retail.',
                  suitability: 85,
                  icon: <span role="img" aria-label="package"></span>,
                },
                {
                  name: 'Medical Devices',
                  description: 'Flexible, strong, and safe for medical use.',
                  suitability: 70,
                  icon: <span role="img" aria-label="medical"></span>,
                },
                {
                  name: 'Textiles',
                  description: 'Durable and sustainable fibers for clothing.',
                  suitability: 65,
                  icon: <span role="img" aria-label="textile"></span>,
                },
                {
                  name: 'Automotive Parts',
                  description: 'High-strength, lightweight components.',
                  suitability: 60,
                  icon: <span role="img" aria-label="car"></span>,
                },
              ]}
            />
          </>
        ) : (
          <>
            <Canvas3D
              molecules={state.placedMolecules}
              viewMode={state.viewMode}
              tool={state.tool}
              selectedObject={state.selectedObject}
              connectStart={state.connectStart}
              movingMoleculeId={state.movingMoleculeId}
              draggedMolecule={draggedMolecule}
              toast={toast}
              isDark={isDark}
              onMoleculeClick={handleMoleculeClick}
              onPlaneClick={handlePlaneClick}
              onPointerMove={handlePointerMove}
              onDrop={handleDrop}
              onViewModeChange={setViewMode}
              onResetCamera={handleResetCamera}
            />

            <PropertiesPanel
              molecules={state.placedMolecules}
              properties={predictedProperties}
              validationResult={validationResult}
              onRemoveMolecule={removeMolecule}
            />
          </>
        )}
      </div>

      {/* Validation Error Popup */}
      <ValidationErrorPopup
        isOpen={showValidationPopup}
        onClose={() => setShowValidationPopup(false)}
        errors={validationResult?.errors || []}
        warnings={validationResult?.warnings || []}
        title={`Validation Failed (${validationResult?.errors?.length || 0} errors)`}
        repairResult={repairResult}
        onAutoRepair={handleAutoRepair}
        isRepairing={isRepairing}
      />

      {/* Loading overlay */}
      {isLoading && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/20">
          <div className="bg-white dark:bg-poly-card px-6 py-4 rounded-lg shadow-lg flex items-center gap-3">
            <div className="w-5 h-5 border-2 border-poly-accent border-t-transparent rounded-full animate-spin" />
            <span className="text-gray-700 dark:text-gray-200">Processing...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default PolyForge;
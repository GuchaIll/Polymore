import React, { useState, useCallback } from 'react';
import * as THREE from 'three';
import Sidebar from '../../components/Sidebar';
import Toolbar from '../../components/Toolbar';
import Canvas3D from '../../components/Canvas3D';
import PropertiesPanel from '../../components/PropertiesPanel';
import ValidationErrorPopup from '../../components/ValidationErrorPopup';
import { usePolyForgeState } from '../../hooks/usePolyForgeState';
import { useTheme } from '../../hooks/useTheme';
import { Molecule, PredictedProperties } from '../../types';
import { 
  validatePolymerComprehensive,
  predictPropertiesFromBackend,
  PolymerValidationResult
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
    undoAction,
    redoAction,
    exportStructure,
    optimizeStructure,
    predictProperties
  } = usePolyForgeState();

  const [draggedMolecule, setDraggedMolecule] = useState<Molecule | null>(null);
  const [predictedProperties, setPredictedProperties] = useState<PredictedProperties | null>(null);
  const [validationResult, setValidationResult] = useState<PolymerValidationResult | null>(null);
  const [showValidationPopup, setShowValidationPopup] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { isDark, toggleTheme } = useTheme();

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
    if (state.tool === 'add' && state.selectedMolecule) {
      placeMolecule(state.selectedMolecule, {
        x: intersection.x,
        y: intersection.y,
        z: intersection.z
      });
    }
  }, [state.tool, state.selectedMolecule, placeMolecule]);

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
    }
  }, [state.tool, state.connectStart, setSelectedObject, removeMolecule, setConnectStart, connectMolecules]);

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
    try {
      const result = await validatePolymerComprehensive(state.placedMolecules);
      setValidationResult(result);
      
      if (result.isValid) {
        showToast(`Valid ${result.polymerType} polymer: ${result.canonicalSmiles}`);
      } else {
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
    try {
      // Comprehensive validation first
      const result = await validatePolymerComprehensive(state.placedMolecules);
      setValidationResult(result);
      
      if (!result.isValid) {
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
                        bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg text-sm">
          {rdkitError}
        </div>
      )}

      <Sidebar
        currentTool={state.tool}
        onToolChange={setTool}
        onMoleculeSelect={setSelectedMolecule}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      />

      <div className="flex-1 flex flex-col relative">
        <Toolbar
          gridSnap={state.gridSnap}
          isDark={isDark}
          onClear={handleClear}
          onUndo={undoAction}
          onRedo={redoAction}
          onToggleSnap={toggleSnap}
          onOptimize={optimizeStructure}
          onPredict={handlePredict}
          onExport={exportStructure}
          onToggleTheme={toggleTheme}
          onValidate={handleValidate}
          rdkitReady={rdkitReady}
        />

        <Canvas3D
          molecules={state.placedMolecules}
          viewMode={state.viewMode}
          tool={state.tool}
          selectedObject={state.selectedObject}
          connectStart={state.connectStart}
          draggedMolecule={draggedMolecule}
          toast={toast}
          isDark={isDark}
          onMoleculeClick={handleMoleculeClick}
          onPlaneClick={handlePlaneClick}
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
      </div>

      {/* Validation Error Popup */}
      <ValidationErrorPopup
        isOpen={showValidationPopup}
        onClose={() => setShowValidationPopup(false)}
        errors={validationResult?.errors || []}
        warnings={validationResult?.warnings || []}
        title={`Validation Failed (${validationResult?.errors?.length || 0} errors)`}
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
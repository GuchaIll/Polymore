import React, { useState, useCallback } from 'react';
import * as THREE from 'three';
import Sidebar from '../../components/Sidebar';
import Toolbar from '../../components/Toolbar';
import Canvas3D from '../../components/Canvas3D';
import PropertiesPanel from '../../components/PropertiesPanel';
import { usePolyForgeState } from '../../hooks/usePolyForgeState';
import { Molecule, PredictedProperties } from '../../types';

const PolyForge: React.FC = () => {
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

  const handlePredict = useCallback(async () => {
    const props = await predictProperties();
    if (props) {
      setPredictedProperties(props);
    }
  }, [predictProperties]);

  const handleClear = useCallback(() => {
    if (state.placedMolecules.length === 0) return;
    if (window.confirm('Clear all molecules?')) {
      clearCanvas();
      setPredictedProperties(null);
    }
  }, [state.placedMolecules.length, clearCanvas]);

  const handleResetCamera = useCallback(() => {
    showToast('Camera reset');
  }, [showToast]);

  return (
    <div className="flex h-screen w-screen overflow-hidden">
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
          onClear={handleClear}
          onUndo={undoAction}
          onRedo={redoAction}
          onToggleSnap={toggleSnap}
          onOptimize={optimizeStructure}
          onPredict={handlePredict}
          onExport={exportStructure}
        />

        <Canvas3D
          molecules={state.placedMolecules}
          viewMode={state.viewMode}
          tool={state.tool}
          selectedObject={state.selectedObject}
          connectStart={state.connectStart}
          draggedMolecule={draggedMolecule}
          toast={toast}
          onMoleculeClick={handleMoleculeClick}
          onPlaneClick={handlePlaneClick}
          onDrop={handleDrop}
          onViewModeChange={setViewMode}
          onResetCamera={handleResetCamera}
        />

        <PropertiesPanel
          molecules={state.placedMolecules}
          properties={predictedProperties}
          onRemoveMolecule={removeMolecule}
        />
      </div>
    </div>
  );
};

export default PolyForge;
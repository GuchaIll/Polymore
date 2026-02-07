import React, { useRef, useState, useCallback, useMemo } from 'react';
import { Canvas, useThree, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { PlacedMolecule, Molecule, ViewMode, ToolType, Toast } from '../../types';

// Molecule 3D component
interface Molecule3DProps {
  molecule: PlacedMolecule;
  isSelected: boolean;
  isHighlighted: boolean;
  onClick: (e: ThreeEvent<MouseEvent>) => void;
}

const Molecule3D: React.FC<Molecule3DProps> = ({ molecule, isSelected, isHighlighted, onClick }) => {
  const color = new THREE.Color(molecule.color);

  return (
    <group position={[molecule.position.x, molecule.position.y, molecule.position.z]}>
      <mesh onClick={onClick}>
        <sphereGeometry args={[0.8, 32, 32]} />
        <meshPhongMaterial
          color={color}
          shininess={80}
          transparent
          opacity={0.9}
          emissive={isSelected || isHighlighted ? new THREE.Color(0x667eea) : new THREE.Color(0x000000)}
        />
      </mesh>
      <mesh>
        <sphereGeometry args={[1.0, 32, 32]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.2}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
};

// Bond component
interface BondProps {
  start: { x: number; y: number; z: number };
  end: { x: number; y: number; z: number };
}

const Bond: React.FC<BondProps> = ({ start, end }) => {
  const direction = new THREE.Vector3().subVectors(
    new THREE.Vector3(end.x, end.y, end.z),
    new THREE.Vector3(start.x, start.y, start.z)
  );
  const length = direction.length();
  const midpoint = new THREE.Vector3()
    .addVectors(
      new THREE.Vector3(start.x, start.y, start.z),
      new THREE.Vector3(end.x, end.y, end.z)
    )
    .multiplyScalar(0.5);

  const quaternion = new THREE.Quaternion();
  quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());

  return (
    <mesh position={midpoint} quaternion={quaternion}>
      <cylinderGeometry args={[0.1, 0.1, length, 8]} />
      <meshPhongMaterial color={0xffffff} transparent opacity={0.6} />
    </mesh>
  );
};

// Click handler
interface ClickHandlerProps {
  onPlaneClick: (intersection: THREE.Vector3) => void;
  tool: ToolType;
}

const ClickHandler: React.FC<ClickHandlerProps> = ({ onPlaneClick, tool }) => {
  const { camera, raycaster, pointer } = useThree();

  const handleClick = useCallback(() => {
    if (tool !== 'add') return;

    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersection = new THREE.Vector3();

    raycaster.setFromCamera(pointer, camera);
    raycaster.ray.intersectPlane(plane, intersection);

    if (intersection) {
      onPlaneClick(intersection);
    }
  }, [camera, raycaster, pointer, onPlaneClick, tool]);

  return (
    <mesh
      position={[0, -0.01, 0]}
      rotation={[-Math.PI / 2, 0, 0]}
      onClick={handleClick}
      visible={false}
    >
      <planeGeometry args={[100, 100]} />
      <meshBasicMaterial transparent opacity={0} />
    </mesh>
  );
};

// Scene content
interface SceneContentProps {
  molecules: PlacedMolecule[];
  viewMode: ViewMode;
  tool: ToolType;
  selectedObject: number | null;
  connectStart: number | null;
  onMoleculeClick: (id: number) => void;
  onPlaneClick: (intersection: THREE.Vector3) => void;
  onCameraReady: (camera: THREE.Camera) => void;
}

// Component to capture camera reference
const CameraCapture: React.FC<{ onCameraReady: (camera: THREE.Camera) => void }> = ({ onCameraReady }) => {
  const { camera } = useThree();

  React.useEffect(() => {
    onCameraReady(camera);
  }, [camera, onCameraReady]);

  return null;
};

const SceneContent: React.FC<SceneContentProps> = ({
  molecules,
  viewMode,
  tool,
  selectedObject,
  connectStart,
  onMoleculeClick,
  onPlaneClick,
  onCameraReady
}) => {
  const bonds = useMemo(() => {
    const bondList: { start: any; end: any; key: string }[] = [];
    const drawnBonds = new Set<string>();

    molecules.forEach(mol => {
      mol.connections.forEach(connId => {
        const bondKey = [mol.id, connId].sort().join('-');
        if (drawnBonds.has(bondKey)) return;
        drawnBonds.add(bondKey);

        const connMol = molecules.find(m => m.id === connId);
        if (connMol) {
          bondList.push({ start: mol.position, end: connMol.position, key: bondKey });
        }
      });
    });

    return bondList;
  }, [molecules]);

  const showStructure = viewMode === 'both' || viewMode === 'structure';

  return (
    <>
      <CameraCapture onCameraReady={onCameraReady} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 20, 10]} intensity={0.8} />

      <Grid
        args={[20, 20]}
        cellSize={2}
        cellThickness={1}
        cellColor="#0f3460"
        sectionSize={10}
        sectionThickness={1.5}
        sectionColor="#0f3460"
        fadeDistance={50}
        fadeStrength={1}
        followCamera={false}
        position={[0, -0.1, 0]}
      />

      <ClickHandler onPlaneClick={onPlaneClick} tool={tool} />

      {showStructure && molecules.map(mol => (
        <Molecule3D
          key={mol.id}
          molecule={mol}
          isSelected={selectedObject === mol.id}
          isHighlighted={connectStart === mol.id}
          onClick={(e) => {
            e.stopPropagation();
            onMoleculeClick(mol.id);
          }}
        />
      ))}

      {showStructure && bonds.map(bond => (
        <Bond key={bond.key} start={bond.start} end={bond.end} />
      ))}

      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={5}
        maxDistance={100}
      />
    </>
  );
};

// Main Canvas3D component
interface Canvas3DProps {
  molecules: PlacedMolecule[];
  viewMode: ViewMode;
  tool: ToolType;
  selectedObject: number | null;
  connectStart: number | null;
  draggedMolecule: Molecule | null;
  toast: Toast;
  onMoleculeClick: (id: number) => void;
  onPlaneClick: (intersection: THREE.Vector3) => void;
  onDrop: (molecule: Molecule, position: { x: number; y: number; z: number }) => void;
  onViewModeChange: (mode: ViewMode) => void;
  onResetCamera: () => void;
}

const Canvas3D: React.FC<Canvas3DProps> = ({
  molecules,
  viewMode,
  tool,
  selectedObject,
  connectStart,
  draggedMolecule,
  toast,
  onMoleculeClick,
  onPlaneClick,
  onDrop,
  onViewModeChange,
  onResetCamera
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const [dropIndicator, setDropIndicator] = useState({ visible: false, x: 0, y: 0 });

  // Callback to capture camera reference from inside Canvas
  const handleCameraReady = useCallback((camera: THREE.Camera) => {
    cameraRef.current = camera;
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setDropIndicator({
        visible: true,
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  }, []);

  const handleDragLeave = useCallback(() => {
    setDropIndicator(prev => ({ ...prev, visible: false }));
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDropIndicator(prev => ({ ...prev, visible: false }));

    if (draggedMolecule && containerRef.current && cameraRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      // Convert screen coordinates to normalized device coordinates (-1 to +1)
      const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      // Create a raycaster and cast from camera through the mouse position
      const raycaster = new THREE.Raycaster();
      const mouseVector = new THREE.Vector2(ndcX, ndcY);
      raycaster.setFromCamera(mouseVector, cameraRef.current);

      // Intersect with the ground plane (y = 0)
      const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const intersection = new THREE.Vector3();
      raycaster.ray.intersectPlane(groundPlane, intersection);

      if (intersection) {
        onDrop(draggedMolecule, { x: intersection.x, y: 0, z: intersection.z });
      }
    }
  }, [draggedMolecule, onDrop]);

  const viewModes: { mode: ViewMode; icon: string; title: string }[] = [
    { mode: 'both', icon: '🔮', title: 'Volume + Structure' },
    { mode: 'structure', icon: '⚛️', title: 'Structure Only' },
    { mode: 'volume', icon: '☁️', title: 'Volume Only' }
  ];

  return (
    <div
      ref={containerRef}
      className="flex-1 relative overflow-hidden"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="w-full h-full">
        <Canvas
          camera={{ position: [15, 15, 15], fov: 60 }}
          style={{ background: '#0a0a1a' }}
        >
          <SceneContent
            molecules={molecules}
            viewMode={viewMode}
            tool={tool}
            selectedObject={selectedObject}
            connectStart={connectStart}
            onMoleculeClick={onMoleculeClick}
            onPlaneClick={onPlaneClick}
            onCameraReady={handleCameraReady}
          />
        </Canvas>
      </div>

      {/* Grid Overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-50"
        style={{
          backgroundImage: 'linear-gradient(rgba(102, 126, 234, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(102, 126, 234, 0.1) 1px, transparent 1px)',
          backgroundSize: '50px 50px'
        }}
      />

      {/* Drop Indicator */}
      {dropIndicator.visible && (
        <div
          className="absolute w-[60px] h-[60px] border-[3px] border-dashed border-poly-accent rounded-full pointer-events-none animate-pulse"
          style={{
            left: dropIndicator.x,
            top: dropIndicator.y,
            transform: 'translate(-50%, -50%)'
          }}
        />
      )}

      {/* View Controls */}
      <div className="absolute top-5 left-5 flex flex-col gap-2">
        {viewModes.map(({ mode, icon, title }) => (
          <button
            key={mode}
            className={`
              w-10 h-10 border-2 rounded-lg text-white cursor-pointer text-lg transition-all
              ${viewMode === mode
                ? 'border-poly-danger bg-poly-danger'
                : 'border-poly-border bg-poly-sidebar/90 hover:border-poly-accent hover:bg-poly-border'
              }
            `}
            onClick={() => onViewModeChange(mode)}
            title={title}
          >
            {icon}
          </button>
        ))}
        <button
          className="w-10 h-10 border-2 border-poly-border rounded-lg bg-poly-sidebar/90 text-white cursor-pointer text-lg transition-all hover:border-poly-accent hover:bg-poly-border"
          onClick={onResetCamera}
          title="Reset View"
        >
          🎯
        </button>
      </div>

      {/* Toast */}
      <div
        className={`
          absolute top-5 left-1/2 -translate-x-1/2 bg-poly-accent/95 text-white
          py-2.5 px-5 rounded-lg text-sm pointer-events-none z-50 transition-opacity
          ${toast.visible ? 'opacity-100' : 'opacity-0'}
        `}
      >
        {toast.message}
      </div>
    </div>
  );
};

export default Canvas3D;
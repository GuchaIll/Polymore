import React, { useRef, useState, useCallback, useMemo } from 'react';
import { Canvas, useThree, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Grid, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Boxes, CircleDot, Cloud, Crosshair, Beaker, GitBranch, Hexagon, Leaf, FlaskRound, LinkIcon, FlaskConical, Atom, Sparkles, Diamond, Zap, DropletIcon } from 'lucide-react';
import { PlacedMolecule, Molecule, ViewMode, ToolType, Toast } from '../../types';

// Helper function to get icon component based on molecule icon name or category
const getMoleculeIcon = (molecule: PlacedMolecule) => {
  const iconMap: Record<string, React.ReactNode> = {
    'Beaker': <Beaker className="w-3 h-3" />,
    'GitBranch': <GitBranch className="w-3 h-3" />,
    'Hexagon': <Hexagon className="w-3 h-3" />,
    'Leaf': <Leaf className="w-3 h-3" />,
    'FlaskRound': <FlaskRound className="w-3 h-3" />,
    'Link': <LinkIcon className="w-3 h-3" />,
    'FlaskConical': <FlaskConical className="w-3 h-3" />,
    'Atom': <Atom className="w-3 h-3" />,
    'Sparkles': <Sparkles className="w-3 h-3" />,
    'Diamond': <Diamond className="w-3 h-3" />,
    'Zap': <Zap className="w-3 h-3" />,
    'Droplet': <DropletIcon className="w-3 h-3" />,
  };
  return iconMap[molecule.icon] || <Beaker className="w-3 h-3" />;
};

// Helper function to get bond color based on linkage type
const getBondStyle = (linkageType?: string): { color: number; thickness: number } => {
  if (!linkageType) return { color: 0xffffff, thickness: 0.1 };
  
  const lt = linkageType.toLowerCase();
  
  // Ester linkages - green
  if (lt.includes('ester')) {
    return { color: 0x10b981, thickness: 0.12 };
  }
  // Amide/peptide bonds - purple
  if (lt.includes('amide') || lt.includes('peptide')) {
    return { color: 0x8b5cf6, thickness: 0.14 };
  }
  // Protein/disulfide bonds - gold
  if (lt.includes('protein') || lt.includes('disulfide')) {
    return { color: 0xf59e0b, thickness: 0.16 };
  }
  // Urethane - blue
  if (lt.includes('urethane')) {
    return { color: 0x3b82f6, thickness: 0.12 };
  }
  // Double bonds - thicker white
  if (lt.includes('double') || lt.includes('vinyl')) {
    return { color: 0xe0e0e0, thickness: 0.18 };
  }
  // C-C backbone - normal white
  if (lt.includes('c-c') || lt.includes('backbone')) {
    return { color: 0xffffff, thickness: 0.1 };
  }
  
  return { color: 0xffffff, thickness: 0.1 };
};

// Molecule 3D component
interface Molecule3DProps {
  molecule: PlacedMolecule;
  isSelected: boolean;
  isHighlighted: boolean;
  isHovered: boolean;
  onClick: (e: ThreeEvent<MouseEvent>) => void;
  onPointerEnter: () => void;
  onPointerLeave: () => void;
}

const Molecule3D: React.FC<Molecule3DProps> = ({ 
  molecule, 
  isSelected, 
  isHighlighted, 
  isHovered,
  onClick,
  onPointerEnter,
  onPointerLeave 
}) => {
  const color = new THREE.Color(molecule.color);

  return (
    <group position={[molecule.position.x, molecule.position.y, molecule.position.z]}>
      {/* Main sphere */}
      <mesh 
        onClick={onClick}
        onPointerEnter={onPointerEnter}
        onPointerLeave={onPointerLeave}
      >
        <sphereGeometry args={[0.8, 32, 32]} />
        <meshPhongMaterial
          color={color}
          shininess={80}
          transparent
          opacity={0.9}
          emissive={isSelected || isHighlighted ? new THREE.Color(0x667eea) : isHovered ? new THREE.Color(0x444466) : new THREE.Color(0x000000)}
        />
      </mesh>
      
      {/* Outer glow sphere */}
      <mesh>
        <sphereGeometry args={[1.0, 32, 32]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={isHovered ? 0.35 : 0.2}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Floating label */}
      <Html
        position={[0, 1.4, 0]}
        center
        distanceFactor={15}
        zIndexRange={[10, 0]}
        style={{
          pointerEvents: 'none',
          userSelect: 'none',
          overflow: 'visible'
        }}
      >
        <div 
          className="flex items-center gap-1 text-xs font-medium text-center"
          style={{ 
            color: '#1a1a2e',
            textShadow: '0 0 3px rgba(255,255,255,0.8)',
            maxWidth: '100px',
            wordWrap: 'break-word',
            whiteSpace: 'normal'
          }}
        >
          <span>{molecule.name}</span>
        </div>
      </Html>
    </group>
  );
};

// Bond component with linkage type styling
interface BondProps {
  start: { x: number; y: number; z: number };
  end: { x: number; y: number; z: number };
  linkageType?: string;
}

const Bond: React.FC<BondProps> = ({ start, end, linkageType }) => {
  const { color, thickness } = getBondStyle(linkageType);
  
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
      <cylinderGeometry args={[thickness, thickness, length, 8]} />
      <meshPhongMaterial color={color} transparent opacity={0.75} />
    </mesh>
  );
};

// Click and move handler for placing molecules and moving
interface ClickHandlerProps {
  onPlaneClick: (intersection: THREE.Vector3) => void;
  onPointerMove: (intersection: THREE.Vector3) => void;
  tool: ToolType;
  isMoving: boolean;
}

const ClickHandler: React.FC<ClickHandlerProps> = ({ onPlaneClick, onPointerMove, tool, isMoving }) => {
  const { camera, raycaster, pointer } = useThree();
  
  // Track pointer down state to distinguish clicks from drags
  const pointerDownRef = useRef<{ time: number; x: number; y: number } | null>(null);
  const CLICK_THRESHOLD_MS = 200; // Max time for a click (vs drag)
  const MOVE_THRESHOLD = 0.02; // Max pointer movement for a click (in NDC units)

  const getPlaneIntersection = useCallback(() => {
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersection = new THREE.Vector3();
    raycaster.setFromCamera(pointer, camera);
    raycaster.ray.intersectPlane(plane, intersection);
    return intersection;
  }, [camera, raycaster, pointer]);

  const handlePointerDown = useCallback(() => {
    pointerDownRef.current = {
      time: Date.now(),
      x: pointer.x,
      y: pointer.y
    };
  }, [pointer]);

  const handleClick = useCallback(() => {
    if (tool !== 'add' && !isMoving) return;

    // Check if this was a quick click (not a drag)
    if (pointerDownRef.current) {
      const elapsed = Date.now() - pointerDownRef.current.time;
      const dx = Math.abs(pointer.x - pointerDownRef.current.x);
      const dy = Math.abs(pointer.y - pointerDownRef.current.y);
      
      // If held too long or moved too much, treat as drag (camera control)
      if (elapsed > CLICK_THRESHOLD_MS || dx > MOVE_THRESHOLD || dy > MOVE_THRESHOLD) {
        pointerDownRef.current = null;
        return;
      }
    }
    pointerDownRef.current = null;

    const intersection = getPlaneIntersection();
    if (intersection) {
      onPlaneClick(intersection);
    }
  }, [tool, isMoving, getPlaneIntersection, onPlaneClick, pointer]);

  const handlePointerMoveEvent = useCallback(() => {
    if (!isMoving) return;

    const intersection = getPlaneIntersection();
    if (intersection) {
      onPointerMove(intersection);
    }
  }, [isMoving, getPlaneIntersection, onPointerMove]);

  return (
    <mesh
      position={[0, -0.01, 0]}
      rotation={[-Math.PI / 2, 0, 0]}
      onPointerDown={handlePointerDown}
      onClick={handleClick}
      onPointerMove={handlePointerMoveEvent}
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
  movingMoleculeId: number | null;
  hoveredMoleculeId: number | null;
  onMoleculeClick: (id: number) => void;
  onPlaneClick: (intersection: THREE.Vector3) => void;
  onPointerMove: (intersection: THREE.Vector3) => void;
  onCameraReady: (camera: THREE.Camera) => void;
  onMoleculeHover: (id: number | null) => void;
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
  movingMoleculeId,
  hoveredMoleculeId,
  onMoleculeClick,
  onPlaneClick,
  onPointerMove,
  onCameraReady,
  onMoleculeHover
}) => {
  // Generate bonds with linkage type information
  const bonds = useMemo(() => {
    const bondList: { start: any; end: any; key: string; linkageType?: string }[] = [];
    const drawnBonds = new Set<string>();

    molecules.forEach(mol => {
      mol.connections.forEach(connId => {
        const bondKey = [mol.id, connId].sort().join('-');
        if (drawnBonds.has(bondKey)) return;
        drawnBonds.add(bondKey);

        const connMol = molecules.find(m => m.id === connId);
        if (connMol) {
          // Determine linkage type from either molecule's linkageType or bondFormed property
          const linkageType = mol.linkageType || connMol.linkageType || mol.bondFormed || connMol.bondFormed;
          bondList.push({ 
            start: mol.position, 
            end: connMol.position, 
            key: bondKey,
            linkageType 
          });
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

      <ClickHandler 
        onPlaneClick={onPlaneClick} 
        onPointerMove={onPointerMove}
        tool={tool} 
        isMoving={movingMoleculeId !== null}
      />

      {showStructure && molecules.map(mol => (
        <Molecule3D
          key={mol.id}
          molecule={mol}
          isSelected={selectedObject === mol.id}
          isHighlighted={connectStart === mol.id || movingMoleculeId === mol.id}
          isHovered={hoveredMoleculeId === mol.id}
          onClick={(e) => {
            e.stopPropagation();
            onMoleculeClick(mol.id);
          }}
          onPointerEnter={() => onMoleculeHover(mol.id)}
          onPointerLeave={() => onMoleculeHover(null)}
        />
      ))}

      {showStructure && bonds.map(bond => (
        <Bond key={bond.key} start={bond.start} end={bond.end} linkageType={bond.linkageType} />
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
  movingMoleculeId: number | null;
  draggedMolecule: Molecule | null;
  toast: Toast;
  isDark: boolean;
  onMoleculeClick: (id: number) => void;
  onPlaneClick: (intersection: THREE.Vector3) => void;
  onPointerMove: (intersection: THREE.Vector3) => void;
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
  movingMoleculeId,
  draggedMolecule,
  toast,
  isDark,
  onMoleculeClick,
  onPlaneClick,
  onPointerMove,
  onDrop,
  onViewModeChange,
  onResetCamera
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const [dropIndicator, setDropIndicator] = useState({ visible: false, x: 0, y: 0 });
  const [hoveredMoleculeId, setHoveredMoleculeId] = useState<number | null>(null);

  // Get the hovered molecule data for overlay panel
  const hoveredMolecule = useMemo(() => {
    if (hoveredMoleculeId === null) return null;
    return molecules.find(m => m.id === hoveredMoleculeId) || null;
  }, [hoveredMoleculeId, molecules]);

  // Callback to capture camera reference from inside Canvas
  const handleCameraReady = useCallback((camera: THREE.Camera) => {
    cameraRef.current = camera;
  }, []);

  // Callback to handle molecule hover
  const handleMoleculeHover = useCallback((id: number | null) => {
    setHoveredMoleculeId(id);
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

  const viewModes: { mode: ViewMode; icon: React.ReactNode; title: string }[] = [
    { mode: 'both', icon: <Boxes className="w-5 h-5" />, title: 'Volume + Structure' },
    { mode: 'structure', icon: <CircleDot className="w-5 h-5" />, title: 'Structure Only' },
    { mode: 'volume', icon: <Cloud className="w-5 h-5" />, title: 'Volume Only' }
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
          style={{ 
            background: isDark ? '#0a0a1a' : '#f8fafb',
            transition: 'background 0.3s ease-in-out'
          }}
        >
          <SceneContent
            molecules={molecules}
            viewMode={viewMode}
            tool={tool}
            selectedObject={selectedObject}
            connectStart={connectStart}
            movingMoleculeId={movingMoleculeId}
            hoveredMoleculeId={hoveredMoleculeId}
            onMoleculeClick={onMoleculeClick}
            onPlaneClick={onPlaneClick}
            onPointerMove={onPointerMove}
            onCameraReady={handleCameraReady}
            onMoleculeHover={handleMoleculeHover}
          />
        </Canvas>
      </div>

      {/* Grid Overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-50 transition-opacity duration-300"
        style={{
          backgroundImage: isDark 
            ? 'linear-gradient(rgba(102, 126, 234, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(102, 126, 234, 0.1) 1px, transparent 1px)'
            : 'linear-gradient(rgba(16, 185, 129, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(16, 185, 129, 0.1) 1px, transparent 1px)',
          backgroundSize: '50px 50px'
        }}
      />

      {/* Drop Indicator */}
      {dropIndicator.visible && (
        <div
          className="absolute w-[60px] h-[60px] border-[3px] border-dashed border-poly-light-accent dark:border-poly-accent rounded-full pointer-events-none animate-pulse"
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
              w-10 h-10 border-2 rounded-lg cursor-pointer transition-all
              flex items-center justify-center
              ${viewMode === mode
                ? 'border-poly-light-accent dark:border-poly-danger bg-poly-light-accent dark:bg-poly-danger text-white'
                : 'border-poly-light-border dark:border-poly-border bg-poly-light-sidebar/90 dark:bg-poly-sidebar/90 text-poly-light-text dark:text-white hover:border-poly-light-accent dark:hover:border-poly-accent hover:bg-poly-light-border dark:hover:bg-poly-border'
              }
            `}
            onClick={() => onViewModeChange(mode)}
            title={title}
          >
            {icon}
          </button>
        ))}
        <button
          className="w-10 h-10 border-2 border-poly-light-border dark:border-poly-border rounded-lg bg-poly-light-sidebar/90 dark:bg-poly-sidebar/90 text-poly-light-text dark:text-white cursor-pointer transition-all hover:border-poly-light-accent dark:hover:border-poly-accent hover:bg-poly-light-border dark:hover:bg-poly-border flex items-center justify-center"
          onClick={onResetCamera}
          title="Reset View"
        >
          <Crosshair className="w-5 h-5" />
        </button>
      </div>

      {/* Hover Info Overlay Panel */}
      {hoveredMolecule && (
        <div 
          className="absolute top-5 left-20 w-64 pointer-events-none transition-opacity duration-200 z-[50]"
        >
          {/* Header with color indicator */}
          <div className="flex items-center gap-3 mb-2">
            <div 
              className="w-8 h-8 rounded-lg flex items-center justify-center text-white shadow-sm"
              style={{ backgroundColor: hoveredMolecule.color }}
            >
              {getMoleculeIcon(hoveredMolecule)}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-poly-light-text dark:text-white font-semibold text-sm drop-shadow-sm">
                {hoveredMolecule.name}
              </h3>
              <p className="text-poly-light-muted dark:text-poly-muted text-xs">
                {hoveredMolecule.category ? hoveredMolecule.category.charAt(0).toUpperCase() + hoveredMolecule.category.slice(1) : 'Molecule'}
              </p>
              {hoveredMolecule.description && (
                <p className="text-poly-light-muted dark:text-poly-muted text-xs mt-1 leading-relaxed">
                  {hoveredMolecule.description}
                </p>
              )}
              {hoveredMolecule.polymerizationType && (
                <p className="text-blue-500 dark:text-blue-400 text-xs mt-0.5">
                  {hoveredMolecule.polymerizationType} polymerization
                </p>
              )}
              {hoveredMolecule.mechanicalEffect && (
                <p className="text-emerald-500 dark:text-emerald-400 text-xs mt-0.5">
                  {hoveredMolecule.mechanicalEffect}
                </p>
              )}
              {hoveredMolecule.sustainabilityImpact && (
                <p className="text-green-500 dark:text-green-400 text-xs mt-0.5">
                  {hoveredMolecule.sustainabilityImpact}
                </p>
              )}
              {hoveredMolecule.commonItems && hoveredMolecule.commonItems.length > 0 && (
                <p className="text-poly-light-muted dark:text-poly-muted text-xs mt-0.5">
                  Found in: {hoveredMolecule.commonItems.join(', ')}
                </p>
              )}
            </div>
          </div>

          {/* Properties grid */}
          <div className="space-y-1.5 text-sm">
            <div className="flex justify-between">
              <span className="text-poly-light-muted dark:text-poly-muted">Formula</span>
              <span className="text-poly-light-text dark:text-white font-mono">{hoveredMolecule.formula}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-poly-light-muted dark:text-poly-muted">SMILES</span>
              <span className="text-poly-light-text dark:text-white font-mono text-xs max-w-[140px] truncate" title={hoveredMolecule.smiles}>
                {hoveredMolecule.smiles}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-poly-light-muted dark:text-poly-muted">Weight</span>
              <span className="text-poly-light-text dark:text-white">{hoveredMolecule.weight.toFixed(2)} g/mol</span>
            </div>
            
            {hoveredMolecule.linkageType && (
              <div className="flex justify-between">
                <span className="text-poly-light-muted dark:text-poly-muted">Linkage</span>
                <span className="text-poly-light-text dark:text-white text-xs">{hoveredMolecule.linkageType}</span>
              </div>
            )}

            {hoveredMolecule.polymerizationType && (
              <div className="flex justify-between">
                <span className="text-poly-light-muted dark:text-poly-muted">Polymerization</span>
                <span className="text-blue-500 dark:text-blue-400 text-xs">{hoveredMolecule.polymerizationType}</span>
              </div>
            )}

            {hoveredMolecule.mechanicalEffect && (
              <div className="flex justify-between">
                <span className="text-poly-light-muted dark:text-poly-muted">Effect</span>
                <span className="text-emerald-500 dark:text-emerald-400 text-xs">{hoveredMolecule.mechanicalEffect}</span>
              </div>
            )}

            {hoveredMolecule.sustainabilityImpact && (
              <div className="flex justify-between items-start">
                <span className="text-poly-light-muted dark:text-poly-muted">Sustainability</span>
                <span className="text-green-500 dark:text-green-400 text-xs max-w-[140px] text-right">{hoveredMolecule.sustainabilityImpact}</span>
              </div>
            )}

            {hoveredMolecule.description && (
              <div className="flex justify-between items-start">
                <span className="text-poly-light-muted dark:text-poly-muted">Description</span>
                <span className="text-poly-light-text dark:text-white text-xs max-w-[140px] text-right">{hoveredMolecule.description}</span>
              </div>
            )}

            {hoveredMolecule.commonItems && hoveredMolecule.commonItems.length > 0 && (
              <div className="flex justify-between items-start">
                <span className="text-poly-light-muted dark:text-poly-muted">Found in</span>
                <span className="text-poly-light-text dark:text-white text-xs max-w-[140px] text-right">{hoveredMolecule.commonItems.join(', ')}</span>
              </div>
            )}
          </div>

        </div>
      )}

      {/* Toast */}
      <div
        className={`
          absolute top-5 left-1/2 -translate-x-1/2 bg-poly-light-accent/95 dark:bg-poly-accent/95 text-white
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
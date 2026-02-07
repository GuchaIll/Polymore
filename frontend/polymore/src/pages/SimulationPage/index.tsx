/**
 * Module: SimulationPage
 * Purpose: Full-page simulation queue management for polymer property prediction
 * Features: WebGL ray marching animation, black/white theme, dark gray queue cards
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  Pause, 
  Trash2, 
  CheckCircle, 
  Clock, 
  AlertCircle,
  Loader2,
  Plus,
  ListOrdered,
  X,
  Atom,
  ArrowLeft
} from 'lucide-react';
import { SimulationTask, SimulationStatus } from '../../types';

interface SimulationPageProps {
  isOpen: boolean;
  currentSmiles: string;
  currentName: string;
  queue: SimulationTask[];
  runningTask: SimulationTask | null;
  completedTasks: SimulationTask[];
  onAddToQueue: (smiles: string, name: string) => boolean;
  onRemoveFromQueue: (taskId: string) => void;
  onClearCompleted: () => void;
  isPaused: boolean;
  onTogglePause: () => void;
  onAutoQueueAttempted?: (success: boolean, message: string) => void;
  onClose?: () => void;
}

const getStatusIcon = (status: SimulationStatus) => {
  switch (status) {
    case 'pending':
      return <Clock className="w-4 h-4 text-yellow-400" />;
    case 'running':
      return <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />;
    case 'processing':
      return <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />;
    case 'completed':
      return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    case 'failed':
      return <AlertCircle className="w-4 h-4 text-amber-400" />;
  }
};

const getStatusLabel = (status: SimulationStatus): string => {
  switch (status) {
    case 'pending': return 'Pending';
    case 'running': return 'Running';
    case 'processing': return 'Processing';
    case 'completed': return 'Completed';
    case 'failed': return 'Failed';
  }
};

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
};

/**
 * Volumetric Ray Marching Animation with Polymer/Helix Structures
 * Features: Double helix SDF, polymer chain, Beer's law absorption, 
 * Henyey-Greenstein phase function, FBM noise
 */
const VolumetricPolymerAnimation: React.FC<{ isRunning: boolean }> = ({ isRunning }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const startTimeRef = useRef<number>(Date.now());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
    if (!gl) return;

    // Vertex shader - fullscreen quad
    const vertexShaderSource = `
      attribute vec2 a_position;
      varying vec2 v_uv;
      void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    // Fragment shader - volumetric ray marching with polymer structures
    const fragmentShaderSource = `
      precision highp float;
      varying vec2 v_uv;
      uniform float u_time;
      uniform vec2 u_resolution;
      uniform float u_speed;
      uniform int u_frame;

      #define MAX_STEPS 30
      #define MAX_STEPS_LIGHTS 3
      #define ABSORPTION_COEFFICIENT 1.5
      #define SCATTERING_ANISO 0.4
      #define PI 3.14159265359
      #define MARCH_SIZE 0.0.2
      #define BASE_AMBIENT 0.1

      const vec3 SUN_POSITION = vec3(2.0, 1.5, 2.0);

      // ============ SDF Primitives ============
      float sdSphere(vec3 p, float radius) {
        return length(p) - radius;
      }

      float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
        vec3 ab = b - a;
        vec3 ap = p - a;
        float t = clamp(dot(ab, ap) / dot(ab, ab), 0.0, 1.0);
        vec3 c = a + t * ab;
        return length(p - c) - r;
      }

      // Torus for helix cross-section
      float sdTorus(vec3 p, vec2 t) {
        vec2 q = vec2(length(p.xz) - t.x, p.y);
        return length(q) - t.y;
      }

      // ============ Hash functions for procedural noise ============
      float hash(float n) { return fract(sin(n) * 43758.5453123); }
      float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      
      vec3 hash3(vec3 p) {
        p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
                 dot(p, vec3(269.5, 183.3, 246.1)),
                 dot(p, vec3(113.5, 271.9, 124.6)));
        return fract(sin(p) * 43758.5453123);
      }

      // ============ 3D Noise ============
      float noise(vec3 x) {
        vec3 p = floor(x);
        vec3 f = fract(x);
        f = f * f * (3.0 - 2.0 * f);

        float n = p.x + p.y * 57.0 + 113.0 * p.z;
        return mix(
          mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
              mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
          mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
              mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y),
          f.z
        );
      }

      // ============ Fractional Brownian Motion (more noise) ============
      float fbm(vec3 p) {
        float t = u_time * u_speed;
        vec3 q = p + t * 0.2 * vec3(1.0, -0.2, -0.5);
        float f = 0.0;
        float scale = 0.6;
        float factor = 2.0;
        for (int i = 0; i < 4; i++) {
          f += scale * (noise(q) * 2.0 - 1.0);
          q *= factor;
          scale *= 0.5;
        }
        return f;
      }

      // ============ Beer's Law ============
      float BeersLaw(float dist, float absorption) {
        return exp(-dist * absorption);
      }

      // ============ Henyey-Greenstein Phase Function ============
      float HenyeyGreenstein(float g, float mu) {
        float gg = g * g;
        return (1.0 / (4.0 * PI)) * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
      }

      // ============ Double Helix SDF ============
      float sdDoubleHelix(vec3 p, float time) {
        float helixRadius = 1.0;
        float tubeRadius = 0.6;
        float pitch = 0.6;
        float twist = p.y * PI / pitch + time * 0.5;
        
        // First strand
        vec3 p1 = vec3(
          p.x - helixRadius * cos(twist),
          p.y,
          p.z - helixRadius * sin(twist)
        );
        float d1 = length(p1.xz) - tubeRadius;
        
        // Second strand (offset by PI)
        vec3 p2 = vec3(
          p.x - helixRadius * cos(twist + PI),
          p.y,
          p.z - helixRadius * sin(twist + PI)
        );
        float d2 = length(p2.xz) - tubeRadius;
        
        // Cross bars connecting strands (like DNA base pairs)
        float barSpacing = 0.2;
        float barY = mod(p.y + time * 0.2, barSpacing) - barSpacing * 0.5;
        float barAngle = floor((p.y + time * 0.2) / barSpacing) * PI * 0.5 + twist;
        
        vec3 barStart = vec3(helixRadius * cos(barAngle), 0.0, helixRadius * sin(barAngle));
        vec3 barEnd = vec3(helixRadius * cos(barAngle + PI), 0.0, helixRadius * sin(barAngle + PI));
        vec3 pBar = vec3(p.x, barY, p.z);
        float dBar = sdCapsule(pBar, barStart, barEnd, 0.06);
        
        return min(min(d1, d2), dBar);
      }

      // ============ Polymer Chain SDF ============
      float sdPolymerChain(vec3 p, float time) {
        float chainDist = 1000.0;
        float segmentLength = 0.45;
        float sphereRadius = 0.12;
        
        // Create a wavy polymer chain
        for (int i = 0; i < 8; i++) {
          float fi = float(i);
          float phase = fi * 0.7 + time * 0.4;
          
          vec3 spherePos = vec3(
            sin(phase) * 0.25,
            fi * segmentLength - 1.8,
            cos(phase * 1.3) * 0.25
          );
          
          float d = sdSphere(p - spherePos, sphereRadius + sin(time + fi) * 0.015);
          chainDist = min(chainDist, d);
          
          // Connect spheres with capsules
          if (i > 0) {
            float prevPhase = (fi - 1.0) * 0.7 + time * 0.4;
            vec3 prevPos = vec3(
              sin(prevPhase) * 0.25,
              (fi - 1.0) * segmentLength - 1.8,
              cos(prevPhase * 1.3) * 0.25
            );
            float capsule = sdCapsule(p, prevPos, spherePos, 0.05);
            chainDist = min(chainDist, capsule);
          }
        }
        
        return chainDist;
      }

      // ============ Combined Scene SDF ============
      float scene(vec3 p) {
        float t = u_time * u_speed;
        
        // Diagonal tilt: rotate so helix goes from bottom-left to top-right
        float tiltAngle = 0.6;
        float cs = cos(tiltAngle), sn = sin(tiltAngle);
        p.xy = mat2(cs, -sn, sn, cs) * p.xy;
        
        // Slight rotation around X for depth
        float tiltX = 0.3;
        float cx = cos(tiltX), sx = sin(tiltX);
        p.yz = mat2(cx, -sx, sx, cx) * p.yz;
        
        // Slow rotation around Y axis
        float rotY = t * 0.12;
        float cy = cos(rotY), sy = sin(rotY);
        p.xz = mat2(cy, -sy, sy, cy) * p.xz;
        
        // Double helix (centered) - sharp SDF
        float helix = sdDoubleHelix(p, t);
        
        // Offset polymer chains on sides
        float chain1 = sdPolymerChain(p - vec3(1.3, 0.0, 0.0), t * 0.8);
        float chain2 = sdPolymerChain(p - vec3(-1.3, 0.0, 0.0), t * 1.2);
        
        // Combine structures - minimal noise for sharp look
        float structures = min(helix, min(chain1, chain2));
        
        // Very subtle surface detail only
        float detail = fbm(p * 3.0) * 0.12;
        
        // Sharp envelope - structures are solid
        return -(structures - 0.05 + detail);
      }

      // ============ Light March (Volumetric Shadow - optimized) ============
      float lightmarch(vec3 position) {
        vec3 sunDirection = normalize(SUN_POSITION);
        float totalDensity = 0.0;
        
        for (int step = 0; step < MAX_STEPS_LIGHTS; step++) {
          position += sunDirection * 0.15 * float(step + 1);
          totalDensity += max(0.0, scene(position));
        }
        
        return BeersLaw(totalDensity, ABSORPTION_COEFFICIENT);
      }

      // ============ Main Ray March ============
      vec4 raymarch(vec3 rayOrigin, vec3 rayDirection, float offset) {
        float depth = 0.0;
        depth += MARCH_SIZE * offset;
        vec3 p = rayOrigin + depth * rayDirection;
        vec3 sunDirection = normalize(SUN_POSITION);

        float totalTransmittance = 1.0;
        vec3 accumulated = vec3(0.0);

        float phase = HenyeyGreenstein(SCATTERING_ANISO, dot(rayDirection, sunDirection));

        for (int i = 0; i < MAX_STEPS; i++) {
          float density = scene(p);

          if (density > 0.0) {
            float lightTransmittance = lightmarch(p);
            float luminance = BASE_AMBIENT + density * phase * 2.5;

            // Crisp white/blue color
            vec3 localColor = vec3(0.95, 0.97, 1.0);
            
            accumulated += totalTransmittance * luminance * lightTransmittance * localColor;
            totalTransmittance *= BeersLaw(density * MARCH_SIZE, ABSORPTION_COEFFICIENT);
            
            if (totalTransmittance < 0.02) break;
          }

          depth += MARCH_SIZE;
          p = rayOrigin + depth * rayDirection;
          
          if (depth > 7.0) break;
        }

        return vec4(accumulated, 1.0 - totalTransmittance);
      }

      // ============ Blue noise approximation ============
      float blueNoise(vec2 coord) {
        return fract(sin(dot(coord, vec2(12.9898, 78.233))) * 43758.5453 + 
                     float(u_frame) * 0.0618);
      }

      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
        
        // Camera setup with orbital movement
        float t = u_time * u_speed * 0.1;
        float camDist = 5.5;
        vec3 ro = vec3(
          sin(t * 0.3) * camDist,
          sin(t * 0.2) * 0.5,
          cos(t * 0.3) * camDist
        );
        
        // Look at center
        vec3 target = vec3(0.0, 0.0, 0.0);
        vec3 forward = normalize(target - ro);
        vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
        vec3 up = cross(forward, right);
        
        // Ray direction
        float fov = 0.8;
        vec3 rd = normalize(forward + fov * (uv.x * right + uv.y * up));
        
        // Blue noise dithering for temporal sampling
        float offset = blueNoise(gl_FragCoord.xy);
        
        // Sky gradient (dark)
        vec3 skyColor = vec3(0.02, 0.02, 0.04);
        skyColor += vec3(0.01, 0.02, 0.03) * (1.0 - uv.y);
        
        // Sun glow
        vec3 sunDir = normalize(SUN_POSITION);
        float sun = pow(max(0.0, dot(rd, sunDir)), 32.0);
        vec3 sunColor = vec3(0.3, 0.2, 0.15) * sun;
        
        // Ray march
        vec4 cloudResult = raymarch(ro, rd, offset);
        
        // Composite
        vec3 color = skyColor + sunColor;
        color = mix(color, cloudResult.rgb, cloudResult.a);
        
        // Subtle bloom on bright areas
        float brightness = dot(color, vec3(0.299, 0.587, 0.114));
        color += vec3(0.05, 0.08, 0.12) * smoothstep(0.3, 0.8, brightness);
        
        // Vignette
        float vignette = 1.0 - length(uv) * 0.4;
        color *= vignette;
        
        // Gamma correction
        color = pow(color, vec3(1.0 / 2.2));
        
        gl_FragColor = vec4(color, 1.0);
      }
    `;

    // Compile shaders
    const compileShader = (source: string, type: number): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertexShader = compileShader(vertexShaderSource, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER);
    
    if (!vertexShader || !fragmentShader) return;

    const program = gl.createProgram();
    if (!program) return;
    
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      return;
    }
    
    gl.useProgram(program);

    // Create fullscreen quad
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    
    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    // Get uniform locations
    const timeLoc = gl.getUniformLocation(program, 'u_time');
    const resolutionLoc = gl.getUniformLocation(program, 'u_resolution');
    const speedLoc = gl.getUniformLocation(program, 'u_speed');
    const frameLoc = gl.getUniformLocation(program, 'u_frame');

    let frameCount = 0;

    // Resize handler
    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);
    };
    resize();
    window.addEventListener('resize', resize);

    // Animation loop
    const animate = () => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;
      const speed = isRunning ? 1.0 : 0.3;
      frameCount++;
      
      gl.uniform1f(timeLoc, elapsed);
      gl.uniform2f(resolutionLoc, canvas.width, canvas.height);
      gl.uniform1f(speedLoc, speed);
      gl.uniform1i(frameLoc, frameCount);
      
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      
      animationRef.current = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationRef.current);
    };
  }, [isRunning]);

  return (
    <canvas 
      ref={canvasRef} 
      className="absolute inset-0 w-full h-full"
    />
  );
};

/**
 * SimulationPage Component
 */
const SimulationPage: React.FC<SimulationPageProps> = ({
  isOpen,
  currentSmiles,
  currentName,
  queue,
  runningTask,
  completedTasks,
  onAddToQueue,
  onRemoveFromQueue,
  onClearCompleted,
  isPaused,
  onTogglePause,
  onAutoQueueAttempted,
  onClose
}) => {
  const [addError, setAddError] = useState<string | null>(null);
  const autoQueueRef = useRef<string | null>(null);
  const isProcessingRef = useRef(false);

  // Auto-queue when page opens - use refs to prevent duplicate queuing
  useEffect(() => {
    if (isOpen && currentSmiles && autoQueueRef.current !== currentSmiles && !isProcessingRef.current) {
      // Set refs BEFORE async operation to prevent race condition
      isProcessingRef.current = true;
      autoQueueRef.current = currentSmiles;
      
      const success = onAddToQueue(currentSmiles, currentName || 'Unnamed Polymer');
      
      if (onAutoQueueAttempted) {
        onAutoQueueAttempted(success, success ? 'Added to simulation queue' : 'Already in queue or running');
      }
      
      // Reset processing flag after a short delay
      setTimeout(() => {
        isProcessingRef.current = false;
      }, 100);
    }
  }, [isOpen, currentSmiles, currentName, onAddToQueue, onAutoQueueAttempted]);

  useEffect(() => {
    if (!isOpen) {
      autoQueueRef.current = null;
      isProcessingRef.current = false;
      setAddError(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleAddToQueue = () => {
    setAddError(null);
    if (!currentSmiles || currentSmiles.trim() === '') {
      setAddError('No valid SMILES to add. Create a molecule first.');
      return;
    }
    const success = onAddToQueue(currentSmiles, currentName || 'Unnamed Polymer');
    if (!success) {
      setAddError('This SMILES is already in the queue or currently running.');
    }
  };

  const totalPending = queue.length;
  const totalCompleted = completedTasks.length;
  const hasActivity = runningTask !== null || totalPending > 0;

  return (
    <div className="flex-1 flex relative overflow-hidden bg-black">
      {/* Volumetric Polymer Animation Background */}
      <VolumetricPolymerAnimation isRunning={runningTask !== null && !isPaused} />

      {/* Main Content */}
      <div className="relative z-10 flex-1 flex flex-col p-6 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 bg-white/10 hover:bg-white/20 rounded-lg border border-white/20 transition-colors"
                title="Back to Editor"
              >
                <ArrowLeft className="w-5 h-5 text-white" />
              </button>
            )}
            <div className="p-3 bg-white/10 rounded-xl border border-white/20">
              <Atom className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">
                Polymer Simulation
              </h1>
              <p className="text-sm text-white/60">
                Queue and run molecular simulations
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-yellow-500/20 border border-yellow-500/30 rounded-lg">
              <Clock className="w-4 h-4 text-yellow-400" />
              <span className="text-sm font-medium text-yellow-400">{totalPending} Pending</span>
            </div>
            {runningTask && (
              <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/20 border border-cyan-500/30 rounded-lg">
                <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                <span className="text-sm font-medium text-cyan-400">Running</span>
              </div>
            )}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/20 border border-emerald-500/30 rounded-lg">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <span className="text-sm font-medium text-emerald-400">{totalCompleted} Done</span>
            </div>
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="flex-1 flex gap-6 overflow-hidden">
          {/* Left Column */}
          <div className="w-80 flex flex-col gap-4">
            {/* Add to Queue Card */}
            <div className="bg-cyan-950/80 rounded-xl border border-cyan-800/50 p-4">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Plus className="w-4 h-4 text-white/70" />
                Add to Queue
              </h3>
              
              <div className="mb-3">
                <p className="text-xs text-white/50 mb-1">Current Molecule</p>
                <p className="font-mono text-xs text-white bg-black/50 p-2 rounded border border-cyan-800/50 truncate">
                  {currentSmiles || 'No molecule on canvas'}
                </p>
              </div>
              
              <button
                onClick={handleAddToQueue}
                disabled={!currentSmiles}
                className={`
                  w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium transition-all
                  ${currentSmiles 
                    ? 'bg-cyan-400 text-black hover:bg-cyan-300' 
                    : 'bg-neutral-700 text-neutral-400 cursor-not-allowed'}
                `}
              >
                <Plus className="w-4 h-4" />
                Add to Queue
              </button>
              
              {addError && (
                <p className="mt-2 text-xs text-amber-400 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" />
                  {addError}
                </p>
              )}
            </div>

            {/* Currently Running Card */}
            {runningTask && (
              <div className="bg-cyan-950/50 backdrop-blur-sm rounded-xl border border-cyan-500/30 p-4">
                <h3 className="text-sm font-semibold text-cyan-400 mb-3 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Currently Running
                </h3>
                <TaskCard task={runningTask} onRemove={onRemoveFromQueue} isRunning />
              </div>
            )}

            {/* Queue Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={onTogglePause}
                className={`
                  flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all border
                  ${isPaused 
                    ? 'bg-emerald-950/50 text-emerald-400 border-emerald-500/30 hover:bg-emerald-950/70' 
                    : 'bg-yellow-950/50 text-yellow-400 border-yellow-500/30 hover:bg-yellow-950/70'}
                `}
              >
                {isPaused ? (
                  <>
                    <Play className="w-4 h-4" />
                    Resume Queue
                  </>
                ) : (
                  <>
                    <Pause className="w-4 h-4" />
                    Pause Queue
                  </>
                )}
              </button>
              
              {totalCompleted > 0 && (
                <button
                  onClick={onClearCompleted}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-amber-950/50 text-amber-400 border border-amber-500/30 hover:bg-amber-950/70 transition-all"
                  title="Clear completed tasks"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* Right Column - Queue and Completed */}
          <div className="flex-1 flex flex-col overflow-hidden bg-cyan-950/70 backdrop-blur-sm rounded-xl border border-cyan-800/50">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* Pending Queue */}
              {queue.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2 sticky top-0 bg-cyan-950/95 py-2 -mt-2 backdrop-blur-sm z-10">
                    <Clock className="w-4 h-4 text-yellow-400" />
                    Pending Queue ({queue.length})
                  </h3>
                  <div className="space-y-2">
                    {queue.map((task, index) => (
                      <TaskCard 
                        key={task.id} 
                        task={task} 
                        onRemove={onRemoveFromQueue}
                        position={index + 1}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Completed Tasks */}
              {completedTasks.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2 sticky top-0 bg-cyan-950/95 py-2 -mt-2 backdrop-blur-sm z-10">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    Completed ({completedTasks.length})
                  </h3>
                  <div className="space-y-2">
                    {completedTasks.map((task) => (
                      <TaskCard 
                        key={task.id} 
                        task={task} 
                        onRemove={onRemoveFromQueue}
                        showResult
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Empty State */}
              {!hasActivity && completedTasks.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center py-12">
                  <ListOrdered className="w-16 h-16 text-cyan-800 mb-4" />
                  <p className="text-lg font-medium text-white mb-2">
                    Queue is empty
                  </p>
                  <p className="text-sm text-white/50">
                    Add a molecule from the editor to start simulation
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * TaskCard component
 */
interface TaskCardProps {
  task: SimulationTask;
  onRemove: (taskId: string) => void;
  isRunning?: boolean;
  position?: number;
  showResult?: boolean;
}

const TaskCard: React.FC<TaskCardProps> = ({ 
  task, 
  onRemove, 
  isRunning = false,
  position,
  showResult = false
}) => {
  return (
    <div className={`
      p-3 rounded-lg border transition-all
      ${isRunning 
        ? 'border-cyan-500/50 bg-cyan-900/50' 
        : task.status === 'completed'
          ? 'border-emerald-500/30 bg-cyan-900/40'
          : task.status === 'failed'
            ? 'border-amber-500/30 bg-amber-950/30'
            : 'border-cyan-700/50 bg-cyan-900/40'}
    `}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {position && (
              <span className="text-xs font-medium text-white/40">
                #{position}
              </span>
            )}
            {getStatusIcon(task.status)}
            <span className="font-medium text-sm text-white truncate">
              {task.name}
            </span>
            <span className={`
              text-xs px-1.5 py-0.5 rounded border
              ${task.status === 'completed' ? 'bg-emerald-950/50 text-emerald-400 border-emerald-500/30' :
                task.status === 'running' ? 'bg-cyan-950/50 text-cyan-400 border-cyan-500/30' :
                task.status === 'failed' ? 'bg-amber-950/50 text-amber-400 border-amber-500/30' :
                'bg-yellow-950/50 text-yellow-400 border-yellow-500/30'}
            `}>
              {getStatusLabel(task.status)}
            </span>
          </div>
          <p className="font-mono text-xs text-white/50 truncate">
            {task.smiles}
          </p>
          
          {/* Progress bar for running task */}
          {isRunning && (
            <div className="mt-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-white/60">Progress</span>
                <span className="text-cyan-400 font-medium">{task.progress}%</span>
              </div>
              <div className="h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-cyan-400 rounded-full transition-all duration-300"
                  style={{ width: `${task.progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Result summary for completed tasks */}
          {showResult && task.result && (
            <div className="mt-2 grid grid-cols-4 gap-1.5 text-xs">
              <div className="bg-cyan-950/80 p-1.5 rounded border border-cyan-800/50 text-center">
                <span className="text-white/60 text-[10px]">STR</span>
                <p className="font-medium text-white">
                  {(task.result.predictedProperties.strength * 100).toFixed(0)}%
                </p>
              </div>
              <div className="bg-cyan-950/80 p-1.5 rounded border border-cyan-800/50 text-center">
                <span className="text-white/60 text-[10px]">FLX</span>
                <p className="font-medium text-white">
                  {(task.result.predictedProperties.flexibility * 100).toFixed(0)}%
                </p>
              </div>
              <div className="bg-cyan-950/80 p-1.5 rounded border border-cyan-800/50 text-center">
                <span className="text-white/60 text-[10px]">DEG</span>
                <p className="font-medium text-white">
                  {(task.result.predictedProperties.degradability * 100).toFixed(0)}%
                </p>
              </div>
              <div className="bg-cyan-950/80 p-1.5 rounded border border-cyan-800/50 text-center">
                <span className="text-white/60 text-[10px]">Time</span>
                <p className="font-medium text-white">
                  {formatDuration(task.result.simulationTime)}
                </p>
              </div>
            </div>
          )}

          {/* Error message */}
          {task.status === 'failed' && task.error && (
            <p className="mt-2 text-xs text-amber-400">
              Error: {task.error}
            </p>
          )}
        </div>

        {/* Remove button */}
        {task.status === 'pending' && (
          <button
            onClick={() => onRemove(task.id)}
            className="p-1 hover:bg-amber-500/20 rounded transition-colors text-amber-400"
            title="Remove from queue"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  );
};

export default SimulationPage;

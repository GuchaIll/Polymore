/**
 * Module: ResultsPage
 * Purpose: Displays polymer property profile and potential applications
 * Features: Radar chart with hover interactions, animated cards, tooltips
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { ArrowLeft, TrendingUp, Leaf, Recycle, FlaskConical, ChevronDown } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';

// Material type for comparison cards
interface MaterialData {
  name: string;
  desc: string;
  strength: number;
  elasticity: number;
  thermal: number;
  flexibility: number;
  ecoScore: number;
  biodegradable: number;
  degradability: number;
  sustainability: number;
}

// Full property set for display
interface FullProperties {
  strength: number;
  elasticity: number;
  thermal: number;
  flexibility: number;
  ecoScore: number;
  biodegradable: number;
  degradability: number;
  sustainability: number;
}

/**
 * Helper: Generate derived properties from the 4 core backend properties
 * Backend returns: strength, flexibility, degradability, sustainability (0-100)
 * We derive: elasticity, thermal, ecoScore, biodegradable
 */
const deriveFullProperties = (props: {
  strength?: number;
  flexibility?: number;
  degradability?: number;
  sustainability?: number;
}): FullProperties => {
  const s = props.strength ?? 0;
  const f = props.flexibility ?? 0;
  const d = props.degradability ?? 0;
  const sus = props.sustainability ?? 0;

  // Elasticity: inverse relationship with strength + flexibility contribution
  // High flexibility + moderate strength = high elasticity
  const elasticity = Math.min(100, Math.max(0, 
    (f * 0.6) + ((100 - s) * 0.3) + (d * 0.1)
  ));

  // Thermal resistance: correlated with strength, inverse with flexibility
  // Strong rigid polymers tend to have better thermal resistance
  const thermal = Math.min(100, Math.max(0,
    (s * 0.5) + ((100 - f) * 0.3) + (sus * 0.2)
  ));

  // Eco Score: weighted combination of sustainability and degradability
  const ecoScore = Math.min(100, Math.max(0,
    (sus * 0.5) + (d * 0.4) + ((100 - s) * 0.1)
  ));

  // Biodegradable: primarily from degradability with sustainability boost
  const biodegradable = Math.min(100, Math.max(0,
    (d * 0.7) + (sus * 0.3)
  ));

  return {
    strength: s,
    elasticity: Math.round(elasticity),
    thermal: Math.round(thermal),
    flexibility: f,
    ecoScore: Math.round(ecoScore),
    biodegradable: Math.round(biodegradable),
    degradability: d,
    sustainability: sus
  };
};

/**
 * Helper: Calculate application suitability scores based on material properties
 */
const calculateApplicationSuitability = (
  props: FullProperties,
  appName: string
): number => {
  // Application-specific weight profiles
  const weights: Record<string, { str: number; flex: number; sus: number; deg: number; eco: number; therm: number }> = {
    'Packaging': { str: 0.15, flex: 0.2, sus: 0.25, deg: 0.25, eco: 0.1, therm: 0.05 },
    'Medical Devices': { str: 0.3, flex: 0.15, sus: 0.15, deg: 0.2, eco: 0.1, therm: 0.1 },
    'Construction': { str: 0.4, flex: 0.05, sus: 0.15, deg: 0.1, eco: 0.1, therm: 0.2 },
    'Textiles': { str: 0.1, flex: 0.35, sus: 0.2, deg: 0.15, eco: 0.15, therm: 0.05 },
    'Electronics': { str: 0.2, flex: 0.1, sus: 0.15, deg: 0.1, eco: 0.1, therm: 0.35 },
    'Agriculture': { str: 0.1, flex: 0.15, sus: 0.3, deg: 0.3, eco: 0.1, therm: 0.05 },
    'default': { str: 0.2, flex: 0.15, sus: 0.25, deg: 0.2, eco: 0.1, therm: 0.1 }
  };

  const w = weights[appName] || weights['default'];
  const score = (
    props.strength * w.str +
    props.flexibility * w.flex +
    props.sustainability * w.sus +
    props.degradability * w.deg +
    props.ecoScore * w.eco +
    props.thermal * w.therm
  );

  return Math.round(Math.min(100, Math.max(0, score)));
};

// Hook to detect when element is visible in viewport
const useScrollVisibility = (threshold: number = 0.2) => {
  const ref = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isVisible) {
          setIsVisible(true);
        }
      },
      { threshold, rootMargin: '0px 0px -50px 0px' }
    );

    observer.observe(element);
    return () => observer.disconnect();
  }, [threshold, isVisible]);

  return { ref, isVisible };
};

interface ResultsPageProps {
  onClose?: () => void;
  properties: {
    strength: number;
    elasticity: number;
    thermal: number;
    flexibility: number;
    ecoScore: number;
    biodegradable: number;
    degradability: number;
    sustainability: number;
  };
  applications: Array<{
    name: string;
    description: string;
    suitability: number;
    icon: React.ReactNode;
  }>;
}

// Animated counter hook - resets and re-animates when value changes
const useAnimatedCounter = (end: number, duration: number = 1500, delay: number = 0) => {
  const [count, setCount] = useState(0);
  const prevEndRef = useRef(end);

  useEffect(() => {
    // Reset count when end value changes significantly
    if (Math.abs(prevEndRef.current - end) > 0.5) {
      setCount(0);
    }
    prevEndRef.current = end;

    const timeout = setTimeout(() => {
      let start = 0;
      const increment = end / (duration / 16);
      const timer = setInterval(() => {
        start += increment;
        if (start >= end) {
          setCount(end);
          clearInterval(timer);
        } else {
          setCount(Math.floor(start));
        }
      }, 16);
      return () => clearInterval(timer);
    }, delay);
    return () => clearTimeout(timeout);
  }, [end, duration, delay]);
  return count;
};

// 8-axis radar chart for all measurement metrics
const radarFields = [
  { key: 'strength', label: 'Strength', color: '#ef4444' },
  { key: 'elasticity', label: 'Elasticity', color: '#f97316' },
  { key: 'thermal', label: 'Thermal', color: '#eab308' },
  { key: 'flexibility', label: 'Flexibility', color: '#22c55e' },
  { key: 'ecoScore', label: 'Eco-Score', color: '#14b8a6' },
  { key: 'biodegradable', label: 'Biodegradable', color: '#06b6d4' },
  { key: 'degradability', label: 'Degradability', color: '#8b5cf6' },
  { key: 'sustainability', label: 'Sustainability', color: '#ec4899' },
];

interface RadarTooltip {
  visible: boolean;
  x: number;
  y: number;
  label: string;
  value: number;
  color: string;
}

const RadarChart: React.FC<{ properties: FullProperties; isDark: boolean; animationKey?: number }> = ({ properties, isDark, animationKey = 0 }) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<RadarTooltip>({ visible: false, x: 0, y: 0, label: '', value: 0, color: '' });
  const [isAnimated, setIsAnimated] = useState(false);
  
  // Re-trigger animation when properties or animationKey changes
  useEffect(() => {
    setIsAnimated(false);
    const timer = setTimeout(() => setIsAnimated(true), 100);
    return () => clearTimeout(timer);
  }, [animationKey, properties.strength, properties.flexibility, properties.degradability, properties.sustainability]);

  const values = radarFields.map(f => Math.max(0, Math.min(1, (properties as any)[f.key] / 100)));
  const cx = 160, cy = 160, r = 110;
  const angleStep = (2 * Math.PI) / radarFields.length;
  
  const points = values.map((v, i) => {
    const angle = -Math.PI / 2 + i * angleStep;
    const animatedV = isAnimated ? v : 0;
    return [cx + Math.cos(angle) * r * animatedV, cy + Math.sin(angle) * r * animatedV];
  });
  
  const axisPoints = radarFields.map((_, i) => {
    const angle = -Math.PI / 2 + i * angleStep;
    return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r];
  });

  const handlePointHover = useCallback((index: number, x: number, y: number) => {
    setHoveredIndex(index);
    setTooltip({
      visible: true,
      x: x + 10,
      y: y - 30,
      label: radarFields[index].label,
      value: Math.round((properties as any)[radarFields[index].key]),
      color: radarFields[index].color
    });
  }, [properties]);

  const handlePointLeave = useCallback(() => {
    setHoveredIndex(null);
    setTooltip(prev => ({ ...prev, visible: false }));
  }, []);

  const gridLevels = [0.25, 0.5, 0.75, 1];
  
  return (
    <div className="w-full flex items-center justify-center relative" style={{ minHeight: '320px' }}>
      <svg width="300" height="300" viewBox="0 0 320 320" className="overflow-visible max-w-full">
        {/* Animated gradient definitions */}
        <defs>
          <linearGradient id="radarGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#a8e6cf" stopOpacity="0.5">
              <animate attributeName="stop-opacity" values="0.3;0.6;0.3" dur="3s" repeatCount="indefinite" />
            </stop>
            <stop offset="100%" stopColor="#56ab2f" stopOpacity="0.5">
              <animate attributeName="stop-opacity" values="0.5;0.8;0.5" dur="3s" repeatCount="indefinite" />
            </stop>
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          <filter id="pointGlow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Grid with hover effect */}
        {gridLevels.map((level, idx) => (
          <polygon
            key={idx}
            points={axisPoints.map(([x, y], i) => {
              const angle = -Math.PI / 2 + i * angleStep;
              const rx = cx + Math.cos(angle) * r * level;
              const ry = cy + Math.sin(angle) * r * level;
              return `${rx},${ry}`;
            }).join(' ')}
            fill="none"
            stroke={isDark ? '#374151' : '#e5e7eb'}
            strokeWidth="1"
            className="transition-all duration-300"
            style={{ opacity: hoveredIndex !== null ? 0.3 : 1 }}
          />
        ))}

        {/* Property polygon with animation */}
        <polygon 
          points={points.map(p => p.join(",")).join(" ")} 
          fill="url(#radarGradient)" 
          stroke="#38ada9" 
          strokeWidth="2.5"
          filter="url(#glow)"
          className="transition-all duration-700 ease-out"
          style={{ 
            strokeDasharray: isAnimated ? 'none' : '1000',
            strokeDashoffset: isAnimated ? '0' : '1000'
          }}
        />

        {/* Axis lines with hover highlight */}
        {axisPoints.map(([x, y], i) => (
          <line 
            key={i} 
            x1={cx} 
            y1={cy} 
            x2={x} 
            y2={y} 
            stroke={hoveredIndex === i ? radarFields[i].color : (isDark ? '#4b5563' : '#bbb')} 
            strokeWidth={hoveredIndex === i ? 2 : 1}
            className="transition-all duration-200"
          />
        ))}

        {/* Interactive data points */}
        {points.map(([x, y], i) => (
          <g key={i}>
            {/* Larger invisible hit area */}
            <circle
              cx={x}
              cy={y}
              r={15}
              fill="transparent"
              className="cursor-pointer"
              onMouseEnter={() => handlePointHover(i, x, y)}
              onMouseLeave={handlePointLeave}
            />
            {/* Visible point */}
            <circle
              cx={x}
              cy={y}
              r={hoveredIndex === i ? 8 : 5}
              fill={radarFields[i].color}
              stroke="white"
              strokeWidth="2"
              filter={hoveredIndex === i ? "url(#pointGlow)" : undefined}
              className="transition-all duration-200 cursor-pointer"
              style={{
                transform: hoveredIndex === i ? 'scale(1.2)' : 'scale(1)',
                transformOrigin: `${x}px ${y}px`
              }}
            />
            {/* Pulse animation on hover */}
            {hoveredIndex === i && (
              <circle
                cx={x}
                cy={y}
                r={12}
                fill="none"
                stroke={radarFields[i].color}
                strokeWidth="2"
                opacity="0.5"
              >
                <animate attributeName="r" from="8" to="20" dur="1s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="0.6" to="0" dur="1s" repeatCount="indefinite" />
              </circle>
            )}
          </g>
        ))}

        {/* Labels with hover effect */}
        {axisPoints.map(([x, y], i) => (
          <text
            key={radarFields[i].key}
            x={x + (x < cx ? -12 : x > cx ? 12 : 0)}
            y={y + (y < cy ? -10 : y > cy ? 20 : 0)}
            textAnchor={x < cx ? 'end' : x > cx ? 'start' : 'middle'}
            fontSize={hoveredIndex === i ? "14" : "12"}
            fontWeight={hoveredIndex === i ? "600" : "400"}
            fill={hoveredIndex === i ? radarFields[i].color : (isDark ? '#d1d5db' : '#444')}
            className="transition-all duration-200 cursor-pointer"
            onMouseEnter={() => handlePointHover(i, points[i][0], points[i][1])}
            onMouseLeave={handlePointLeave}
          >
            {radarFields[i].label}
          </text>
        ))}

        {/* Grid values */}
        {gridLevels.map((level, idx) => (
          <text
            key={"g"+idx}
            x={cx + 5}
            y={cy - r * level - 4}
            textAnchor="start"
            fontSize="9"
            fill={isDark ? '#6b7280' : '#bbb'}
          >
            {Math.round(level * 100)}
          </text>
        ))}
      </svg>

      {/* Tooltip */}
      {tooltip.visible && (
        <div 
          className="absolute pointer-events-none z-10 px-3 py-2 rounded-lg shadow-lg border backdrop-blur-sm animate-fadeIn"
          style={{ 
            left: tooltip.x + 160 - 160, 
            top: tooltip.y + 160 - 160,
            backgroundColor: isDark ? 'rgba(31, 41, 55, 0.95)' : 'rgba(255, 255, 255, 0.95)',
            borderColor: tooltip.color
          }}
        >
          <div className="text-xs font-medium" style={{ color: tooltip.color }}>{tooltip.label}</div>
          <div className="text-lg font-bold" style={{ color: isDark ? '#fff' : '#1f2937' }}>{tooltip.value}%</div>
        </div>
      )}
    </div>
  );
};

// Animated stat card component
const StatCard: React.FC<{
  value: number;
  label: string;
  suffix?: string;
  delay: number;
  gradientFrom: string;
  gradientTo: string;
  borderColor: string;
  textColor: string;
  labelColor: string;
  icon: React.ReactNode;
}> = ({ value, label, suffix = '%', delay, gradientFrom, gradientTo, borderColor, textColor, labelColor, icon }) => {
  const animatedValue = useAnimatedCounter(value, 1500, delay);
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div 
      className={`
        relative overflow-hidden rounded-xl px-5 py-4 flex flex-col items-center min-w-[140px] border
        transition-all duration-300 cursor-pointer group
        ${isHovered ? 'scale-105 shadow-lg' : 'scale-100 shadow'}
      `}
      style={{ 
        background: `linear-gradient(135deg, ${gradientFrom}, ${gradientTo})`,
        borderColor: borderColor
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Background pulse on hover */}
      <div 
        className={`
          absolute inset-0 bg-white/10 transition-opacity duration-300
          ${isHovered ? 'opacity-100' : 'opacity-0'}
        `}
      />
      
      {/* Icon */}
      <div className={`
        absolute top-2 right-2 transition-all duration-300
        ${isHovered ? 'scale-110 opacity-100' : 'scale-90 opacity-50'}
      `} style={{ color: textColor }}>
        {icon}
      </div>

      {/* Value with counter animation */}
      <span 
        className={`font-bold text-3xl transition-all duration-300 ${isHovered ? 'scale-110' : 'scale-100'}`}
        style={{ color: textColor }}
      >
        {animatedValue}{suffix}
      </span>
      
      {/* Label */}
      <span className="text-xs font-medium mt-1" style={{ color: labelColor }}>{label}</span>

      {/* Shine effect on hover */}
      <div 
        className={`
          absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent
          transition-transform duration-700 -skew-x-12
          ${isHovered ? 'translate-x-[200%]' : '-translate-x-[200%]'}
        `}
      />
    </div>
  );
};

// Application card component with hover effects
const ApplicationCard: React.FC<{
  app: { name: string; description: string; suitability: number; icon: React.ReactNode };
  index: number;
  barRef: (el: HTMLDivElement | null) => void;
}> = ({ app, index, barRef }) => {
  const [isHovered, setIsHovered] = useState(false);
  
  return (
    <div 
      className={`
        bg-gradient-to-br from-[#e8f5e9] to-[#c8e6c9] dark:from-emerald-900/30 dark:to-emerald-800/20 
        rounded-xl p-4 flex flex-col gap-2 border border-[#a5d6a7] dark:border-emerald-700/40
        transition-all duration-300 cursor-pointer
        ${isHovered ? 'scale-[1.02] shadow-lg shadow-emerald-500/20 -translate-y-1' : 'scale-100 shadow'}
      `}
      style={{ 
        animationDelay: `${index * 100}ms`,
        opacity: 0,
        animation: 'slideUp 0.5s ease-out forwards'
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex items-center gap-3">
        {/* Icon with bounce effect */}
        <div className={`
          bg-gradient-to-br from-[#4ecdc4] to-[#44a08d] rounded-xl w-12 h-12 
          flex items-center justify-center text-xl text-white flex-shrink-0
          transition-all duration-300 shadow-md
          ${isHovered ? 'scale-110 rotate-3 shadow-lg' : 'scale-100 rotate-0'}
        `}>
          {app.icon}
        </div>
        <div className="min-w-0 flex-1">
          <span className={`
            font-semibold text-sm text-[#1b5e20] dark:text-emerald-300 block truncate
            transition-all duration-200
            ${isHovered ? 'text-[#0d5214]' : ''}
          `}>
            {app.name}
          </span>
          <span className="text-[11px] text-gray-600 dark:text-gray-400 line-clamp-2 leading-relaxed">
            {app.description}
          </span>
        </div>
      </div>
      
      {/* Suitability bar */}
      <div className="mt-1">
        <div className="flex justify-between items-center mb-1">
          <span className="text-[10px] text-gray-600 dark:text-gray-400 font-medium">Suitability</span>
          <span className={`
            text-xs font-mono font-bold text-[#2d6a4f] dark:text-emerald-400
            transition-all duration-300
            ${isHovered ? 'scale-110' : 'scale-100'}
          `}>
            {app.suitability}%
          </span>
        </div>
        <div className="w-full h-2 bg-[#c8e6c9] dark:bg-emerald-900/50 rounded-full overflow-hidden">
          <div
            className={`
              h-full bg-gradient-to-r from-[#4ecdc4] to-[#44a08d] rounded-full 
              transition-all duration-700 ease-out
              ${isHovered ? 'shadow-inner' : ''}
            `}
            ref={barRef}
            style={{ width: '0%' }}
          />
        </div>
      </div>
    </div>
  );
};

// Plastic comparison card with hover and selection
const PlasticCard: React.FC<{ 
  plastic: MaterialData; 
  index: number; 
  isSelected: boolean;
  onClick: () => void;
}> = ({ plastic, index, isSelected, onClick }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div 
      className={`
        bg-gradient-to-br from-[#a8e6cf]/30 to-[#56ab2f]/20 dark:from-emerald-500/20 dark:to-emerald-700/10 
        rounded-xl px-4 py-3 flex flex-col items-center min-w-[120px] max-w-[220px] 
        border-2 transition-all duration-300 cursor-pointer
        ${isSelected 
          ? 'border-emerald-500 dark:border-emerald-400 scale-105 shadow-lg shadow-emerald-500/30 ring-2 ring-emerald-500/20' 
          : isHovered 
            ? 'border-emerald-500/60 scale-105 shadow-lg shadow-emerald-500/20' 
            : 'border-[#95e1d3]/40 dark:border-emerald-500/30 scale-100'
        }
      `}
      style={{ 
        animationDelay: `${index * 150}ms`,
        opacity: 0,
        animation: 'fadeIn 0.5s ease-out forwards'
      }}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Selection indicator */}
      {isSelected && (
        <div className="absolute -top-1 -right-1 w-5 h-5 bg-emerald-500 rounded-full flex items-center justify-center">
          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
          </svg>
        </div>
      )}
      <span className={`
        font-bold text-sm transition-all duration-200
        ${isSelected ? 'text-emerald-600 dark:text-emerald-300 scale-110' : 'text-[#2d6a4f] dark:text-emerald-400'}
        ${isHovered && !isSelected ? 'scale-110' : ''}
      `}>
        {plastic.name}
      </span>
      <span className={`
        text-[10px] text-center leading-tight mt-1 transition-all duration-300
        ${isSelected ? 'text-emerald-700 dark:text-emerald-300' : 'text-gray-600 dark:text-gray-400'}
        ${isHovered ? 'text-gray-700 dark:text-gray-300' : ''}
      `}>
        {plastic.desc}
      </span>
    </div>
  );
};

const ResultsPage: React.FC<ResultsPageProps> = ({ onClose, properties, applications }) => {
  const { isDark } = useTheme();
  const barRefs = useRef<(HTMLDivElement | null)[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const [showScrollIndicator, setShowScrollIndicator] = useState(true);
  const { ref: sustainabilityRef, isVisible: sustainabilityVisible } = useScrollVisibility(0.15);
  const [selectedMaterialIndex, setSelectedMaterialIndex] = useState(0); // 0 = Current Polymer

  // Derive full properties from backend's 4 core values for "Your Polymer"
  const derivedPolymerProps = useMemo(() => deriveFullProperties(properties), [properties]);

  // Build materials array with Current Polymer as first entry
  const materialsComparison: MaterialData[] = useMemo(() => [
    {
      name: 'Your Polymer',
      desc: 'Your designed polymer with predicted properties.',
      ...derivedPolymerProps
    },
    {
      name: 'PLA',
      desc: 'Biodegradable, compostable, used in packaging and 3D printing.',
      strength: 65,
      elasticity: 40,
      thermal: 35,
      flexibility: 45,
      ecoScore: 90,
      biodegradable: 95,
      degradability: 90,
      sustainability: 85
    },
    {
      name: 'PET',
      desc: 'Common in bottles, strong and recyclable but not biodegradable.',
      strength: 85,
      elasticity: 70,
      thermal: 80,
      flexibility: 65,
      ecoScore: 60,
      biodegradable: 5,
      degradability: 10,
      sustainability: 55
    },
    {
      name: 'HDPE',
      desc: 'Very durable and chemically resistant, widely used in containers.',
      strength: 80,
      elasticity: 85,
      thermal: 75,
      flexibility: 90,
      ecoScore: 40,
      biodegradable: 0,
      degradability: 5,
      sustainability: 35
    }
  ], [derivedPolymerProps]);

  // Get currently selected material's properties
  const selectedMaterial = materialsComparison[selectedMaterialIndex];
  const displayProperties: FullProperties = useMemo(() => ({
    strength: selectedMaterial.strength,
    elasticity: selectedMaterial.elasticity,
    thermal: selectedMaterial.thermal,
    flexibility: selectedMaterial.flexibility,
    ecoScore: selectedMaterial.ecoScore,
    biodegradable: selectedMaterial.biodegradable,
    degradability: selectedMaterial.degradability,
    sustainability: selectedMaterial.sustainability
  }), [selectedMaterial]);

  // Calculate application suitability based on selected material using helper
  const calculatedApplications = useMemo(() => {
    return applications.map(app => ({
      ...app,
      suitability: calculateApplicationSuitability(displayProperties, app.name)
    }));
  }, [displayProperties, applications]);
  
  // Reset and re-animate bars when material selection changes
  useEffect(() => {
    // First reset all bars to 0
    barRefs.current.forEach(ref => {
      if (ref) ref.style.width = '0%';
    });
    
    // Then animate to new values
    calculatedApplications.forEach((app, i) => {
      setTimeout(() => {
        const ref = barRefs.current[i];
        if (ref) {
          ref.style.width = `${app.suitability}%`;
        }
      }, 100 + i * 100);
    });
  }, [selectedMaterialIndex, calculatedApplications]);

  // Handle scroll to hide indicator
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (container.scrollTop > 50) {
        setShowScrollIndicator(false);
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSustainability = () => {
    sustainabilityRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

// Materials comparison is now generated in component with useMemo

  return (
    <div ref={containerRef} className="flex flex-col min-h-full w-full bg-gradient-to-br from-[#f0fff4] to-[#e6f7ed] dark:from-poly-bg dark:to-poly-sidebar p-4 md:p-6 pb-8 overflow-y-auto overflow-x-hidden relative scroll-smooth">
      {/* CSS Animations */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(40px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }
        @keyframes bounceDown {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(8px); }
        }
        @keyframes slideUpReveal {
          from { opacity: 0; transform: translateY(60px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .animate-fadeIn { animation: fadeIn 0.5s ease-out forwards; }
        .animate-slideUpReveal { animation: slideUpReveal 0.8s ease-out forwards; }
        .animate-bounceDown { animation: bounceDown 1.5s ease-in-out infinite; }
      `}</style>

      {/* Back button with hover effect */}
      {onClose && (
        <div className="flex items-center mb-4 flex-shrink-0">
          <button
            onClick={onClose}
            className="flex items-center gap-2 px-4 py-2.5 bg-[#2d6a4f]/10 dark:bg-emerald-500/10 
                       hover:bg-[#2d6a4f]/20 dark:hover:bg-emerald-500/20 rounded-xl 
                       border border-[#2d6a4f]/20 dark:border-emerald-500/30 
                       transition-all duration-300 text-[#2d6a4f] dark:text-emerald-400
                       hover:scale-105 hover:shadow-md active:scale-95 group"
          >
            <ArrowLeft className="w-4 h-4 transition-transform duration-200 group-hover:-translate-x-1" />
            <span className="text-sm font-medium">Back to Editor</span>
          </button>
        </div>
      )}

      {/* Material selection cards */}
      <div className="flex flex-col items-center mb-5 flex-shrink-0">
        <p className="text-center text-gray-600 dark:text-gray-400 text-xs mb-3 opacity-0 animate-fadeIn" style={{ animationDelay: '100ms' }}>
          Select a material to compare properties
        </p>
        <div className="flex gap-4 mb-3 flex-wrap justify-center">
          {materialsComparison.map((material, index) => (
            <PlasticCard 
              key={material.name} 
              plastic={material} 
              index={index}
              isSelected={selectedMaterialIndex === index}
              onClick={() => setSelectedMaterialIndex(index)}
            />
          ))}
        </div>
        <p className="text-center text-gray-600 dark:text-gray-400 text-xs max-w-xl opacity-0 animate-fadeIn" style={{ animationDelay: '500ms' }}>
          {selectedMaterialIndex === 0 
            ? 'Viewing your designed polymer properties and potential applications.'
            : `Comparing with ${selectedMaterial.name} - click "Your Polymer" to return to your design.`
          }
        </p>
      </div>

      {/* Title with animation */}
      <h2 className="text-xl font-bold mb-1 text-center flex-shrink-0 text-poly-light-text dark:text-poly-text opacity-0 animate-fadeIn" style={{ animationDelay: '200ms' }}>
        Environmental & Application Impact
      </h2>
      <p className="text-center text-gray-500 dark:text-gray-400 mb-5 text-sm flex-shrink-0 opacity-0 animate-fadeIn" style={{ animationDelay: '300ms' }}>
        Comprehensive analysis of your polymer's properties and potential uses
      </p>

      <div className="flex flex-col lg:flex-row gap-4 md:gap-6">
        {/* Property Profile */}
        <div className="flex-1 bg-white dark:bg-poly-card rounded-2xl p-4 md:p-5 shadow-lg border border-[#95e1d3]/30 dark:border-emerald-500/20 
                        transition-all duration-300 hover:shadow-xl hover:border-emerald-500/40 overflow-hidden">
          <h3 className="text-base font-semibold mb-2 text-[#2d6a4f] dark:text-emerald-400 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Property Profile {selectedMaterialIndex > 0 ? `(${selectedMaterial.name})` : ''}
          </h3>
          <div className="flex items-center justify-center overflow-visible">
            <RadarChart properties={displayProperties} isDark={isDark} animationKey={selectedMaterialIndex} />
          </div>
        </div>

        {/* Potential Applications */}
        <div className="flex-1 bg-white dark:bg-poly-card rounded-2xl p-4 md:p-5 shadow-lg border border-[#95e1d3]/30 dark:border-emerald-500/20
                        transition-all duration-300 hover:shadow-xl hover:border-emerald-500/40 overflow-hidden">
          <h3 className="text-base font-semibold mb-4 text-[#2d6a4f] dark:text-emerald-400 flex items-center gap-2">
            <FlaskConical className="w-4 h-4" />
            Potential Applications
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 md:gap-4">
            {calculatedApplications.length === 0 ? (
              <div className="col-span-2 text-center text-gray-400 dark:text-gray-500 py-8">
                No application data available
              </div>
            ) : (
              calculatedApplications.map((app, i) => (
                <ApplicationCard 
                  key={app.name} 
                  app={app} 
                  index={i}
                  barRef={(el) => { barRefs.current[i] = el; }}
                />
              ))
            )}
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      {showScrollIndicator && (
        <div 
          className="flex flex-col items-center mt-4 mb-2 cursor-pointer group"
          onClick={scrollToSustainability}
        >
          <span className="text-xs text-gray-500 dark:text-gray-400 mb-1 group-hover:text-emerald-500 transition-colors">
            Scroll for sustainability insights
          </span>
          <div className="animate-bounceDown">
            <ChevronDown className="w-6 h-6 text-emerald-500 dark:text-emerald-400 group-hover:text-emerald-400 dark:group-hover:text-emerald-300 transition-colors" />
          </div>
        </div>
      )}

      {/* Sustainability Impact Section - Scroll Triggered */}
      <div 
        ref={sustainabilityRef}
        className={`
          mt-4 md:mt-6 flex flex-col items-center flex-shrink-0 bg-white dark:bg-poly-card rounded-2xl p-4 md:p-5 shadow-lg border border-[#95e1d3]/30 dark:border-emerald-500/20
          transition-all duration-300 hover:shadow-xl
          ${sustainabilityVisible ? 'animate-slideUpReveal' : 'opacity-0 translate-y-10'}
        `}
      >
        <h3 className="text-base font-semibold mb-4 text-[#2d6a4f] dark:text-emerald-400 flex items-center gap-2">
          <Leaf className="w-4 h-4" />
          Sustainability Impact
        </h3>
        <div className="flex gap-3 md:gap-5 mb-4 flex-wrap justify-center">
          <StatCard
            value={displayProperties.sustainability || 0}
            label="Sustainability Score"
            delay={sustainabilityVisible ? 200 : 99999}
            gradientFrom={isDark ? 'rgba(16, 185, 129, 0.3)' : '#a8e6cf'}
            gradientTo={isDark ? 'rgba(6, 95, 70, 0.2)' : 'rgba(86, 171, 47, 0.3)'}
            borderColor={isDark ? 'rgba(16, 185, 129, 0.3)' : 'rgba(86, 171, 47, 0.4)'}
            textColor={isDark ? '#6ee7b7' : '#1b5e20'}
            labelColor={isDark ? '#34d399' : '#2d6a4f'}
            icon={<Leaf className="w-5 h-5" />}
          />
          <StatCard
            value={displayProperties.biodegradable || 0}
            label="Biodegradable Score"
            delay={sustainabilityVisible ? 350 : 99999}
            gradientFrom={isDark ? 'rgba(20, 184, 166, 0.3)' : '#b2dfdb'}
            gradientTo={isDark ? 'rgba(15, 118, 110, 0.2)' : 'rgba(38, 166, 154, 0.3)'}
            borderColor={isDark ? 'rgba(20, 184, 166, 0.3)' : 'rgba(38, 166, 154, 0.4)'}
            textColor={isDark ? '#5eead4' : '#00695c'}
            labelColor={isDark ? '#2dd4bf' : '#004d40'}
            icon={<Recycle className="w-5 h-5" />}
          />
          <StatCard
            value={displayProperties.degradability || 0}
            label="Degradability Score"
            delay={sustainabilityVisible ? 500 : 99999}
            gradientFrom={isDark ? 'rgba(6, 182, 212, 0.3)' : '#95e1d3'}
            gradientTo={isDark ? 'rgba(8, 145, 178, 0.2)' : 'rgba(56, 173, 169, 0.3)'}
            borderColor={isDark ? 'rgba(6, 182, 212, 0.3)' : 'rgba(56, 173, 169, 0.4)'}
            textColor={isDark ? '#67e8f9' : '#00695c'}
            labelColor={isDark ? '#22d3ee' : '#004d40'}
            icon={<TrendingUp className="w-5 h-5" />}
          />
          <StatCard
            value={Math.floor((displayProperties.ecoScore || 0) * 0.1)}
            label="Experiments Saved"
            suffix=""
            delay={sustainabilityVisible ? 650 : 99999}
            gradientFrom={isDark ? 'rgba(16, 185, 129, 0.3)' : '#4ecdc4'}
            gradientTo={isDark ? 'rgba(20, 184, 166, 0.2)' : 'rgba(68, 160, 141, 0.3)'}
            borderColor={isDark ? 'rgba(16, 185, 129, 0.3)' : 'rgba(68, 160, 141, 0.4)'}
            textColor={isDark ? '#6ee7b7' : '#00796b'}
            labelColor={isDark ? '#34d399' : '#004d40'}
            icon={<FlaskConical className="w-5 h-5" />}
          />
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 text-center max-w-lg opacity-0 animate-fadeIn" style={{ animationDelay: '600ms' }}>
          Higher sustainability and biodegradability scores indicate a greener polymer. 
          Experiments saved is an estimate based on model predictions vs. lab trials.
        </p>
      </div>
    </div>
  );
};

export default ResultsPage;

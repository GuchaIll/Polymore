/**
 * Module: ResultsPage
 * Purpose: Displays polymer property profile and potential applications
 * Features: Radar chart, application cards, summary fields
 */

import React, { useEffect, useRef } from 'react';
import { ArrowLeft } from 'lucide-react';

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

// 8-axis radar chart for all measurement metrics
const radarFields = [
  { key: 'strength', label: 'Strength' },
  { key: 'elasticity', label: 'Elasticity' },
  { key: 'thermal', label: 'Thermal' },
  { key: 'flexibility', label: 'Flexibility' },
  { key: 'ecoScore', label: 'Eco-Score' },
  { key: 'biodegradable', label: 'Biodegradable' },
  { key: 'degradability', label: 'Degradability' },
  { key: 'sustainability', label: 'Sustainability' },
];
const RadarChart: React.FC<{ properties: ResultsPageProps['properties'] }> = ({ properties }) => {
  const values = radarFields.map(f => Math.max(0, Math.min(1, (properties as any)[f.key] / 100)));
  const cx = 160, cy = 160, r = 110;
  const angleStep = (2 * Math.PI) / radarFields.length;
  const points = values.map((v, i) => {
    const angle = -Math.PI / 2 + i * angleStep;
    return [cx + Math.cos(angle) * r * v, cy + Math.sin(angle) * r * v];
  });
  const axisPoints = radarFields.map((_, i) => {
    const angle = -Math.PI / 2 + i * angleStep;
    return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r];
  });
  // Draw grid
  const gridLevels = [0.25, 0.5, 0.75, 1];
  return (
    <div className="w-full h-80 flex items-center justify-center">
      <svg width="320" height="320" viewBox="0 0 320 320">
        {/* Grid */}
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
            stroke="#e5e7eb"
            strokeWidth="1"
          />
        ))}
        {/* Property polygon - green gradient fill */}
        <defs>
          <linearGradient id="radarGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#a8e6cf" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#56ab2f" stopOpacity="0.4" />
          </linearGradient>
        </defs>
        <polygon points={points.map(p => p.join(",")).join(" ")} fill="url(#radarGradient)" stroke="#38ada9" strokeWidth="2" />
        {/* Axis lines */}
        {axisPoints.map(([x, y], i) => (
          <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#bbb" strokeWidth="1" />
        ))}
        {/* Labels */}
        {axisPoints.map(([x, y], i) => (
          <text
            key={radarFields[i].key}
            x={x + (x < cx ? -10 : x > cx ? 10 : 0)}
            y={y + (y < cy ? -8 : y > cy ? 18 : 0)}
            textAnchor={x < cx ? 'end' : x > cx ? 'start' : 'middle'}
            fontSize="13"
            fill="#444"
          >
            {radarFields[i].label}
          </text>
        ))}
        {/* Grid values */}
        {gridLevels.map((level, idx) => (
          <text
            key={"g"+idx}
            x={cx}
            y={cy - r * level - 4}
            textAnchor="middle"
            fontSize="10"
            fill="#bbb"
          >
            {Math.round(level * 100)}
          </text>
        ))}
      </svg>
    </div>
  );
};

const ResultsPage: React.FC<ResultsPageProps> = ({ onClose, properties, applications }) => {
  // Animate suitability bar widths
  const barRefs = useRef<(HTMLDivElement | null)[]>([]);
  useEffect(() => {
    applications.forEach((app, i) => {
      const ref = barRefs.current[i];
      if (ref) {
        ref.style.width = `${app.suitability}%`;
      }
    });
  }, [applications]);

  // Example: compare to PLA, PET, etc. (static for now)
  const plasticComparison = [
    { name: 'PLA', desc: 'Biodegradable, compostable, used in packaging and 3D printing.' },
    { name: 'PET', desc: 'Common in bottles, strong but not biodegradable.' },
    { name: 'HDPE', desc: 'Durable, used in containers, not biodegradable.' },
  ];

  return (
    <div className="flex flex-col h-full w-full bg-gradient-to-br from-[#f0fff4] to-[#e6f7ed] p-6 pb-8 overflow-y-auto">
      {/* Back button */}
      {onClose && (
        <div className="flex items-center mb-4 flex-shrink-0">
          <button
            onClick={onClose}
            className="flex items-center gap-2 px-3 py-2 bg-[#2d6a4f]/10 hover:bg-[#2d6a4f]/20 rounded-lg border border-[#2d6a4f]/20 transition-colors text-[#2d6a4f]"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm font-medium">Back to Editor</span>
          </button>
        </div>
      )}
      {/* Top plastic comparison */}
      <div className="flex flex-col items-center mb-4 flex-shrink-0">
        <div className="flex gap-3 mb-2 flex-wrap justify-center">
          {plasticComparison.map(plastic => (
            <div key={plastic.name} className="bg-gradient-to-br from-[#a8e6cf]/30 to-[#56ab2f]/20 rounded-lg px-3 py-2 flex flex-col items-center min-w-[100px] max-w-[200px] border border-[#95e1d3]/40">
              <span className="font-bold text-[#2d6a4f] text-sm">{plastic.name}</span>
              <span className="text-[10px] text-gray-600 text-center leading-tight">{plastic.desc}</span>
            </div>
          ))}
        </div>
        <p className="text-center text-gray-600 text-xs max-w-xl">Compare your polymer to common plastics for sustainability, degradability, and application fit.</p>
      </div>

      <h2 className="text-xl font-bold mb-1 text-center flex-shrink-0">Environmental & Application Impact</h2>
      <p className="text-center text-gray-500 mb-4 text-sm flex-shrink-0">Comprehensive analysis of your polymer's properties and potential uses</p>
      <div className="flex gap-6 flex-1 min-h-0">
        {/* Property Profile */}
        <div className="flex-1 bg-white rounded-xl p-4 shadow border border-[#95e1d3]/30 min-w-0">
          <h3 className="text-base font-semibold mb-2 text-[#2d6a4f]">Property Profile</h3>
          <RadarChart properties={properties} />
        </div>
        {/* Potential Applications */}
        <div className="flex-1 bg-white rounded-xl p-4 shadow border border-[#95e1d3]/30 min-w-0 ">
          <h3 className="text-base font-semibold mb-3 text-[#2d6a4f]">Potential Applications</h3>
          <div className="grid grid-cols-2 gap-3">
            {applications.length === 0 ? (
              <div className="col-span-2 text-center text-gray-400">No application data available</div>
            ) : (
              applications.map((app, i) => (
                <div key={app.name} className="bg-gradient-to-br from-[#e8f5e9] to-[#c8e6c9] rounded-lg p-3 flex flex-col gap-1 border border-[#a5d6a7]">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="bg-gradient-to-br from-[#4ecdc4] to-[#44a08d] rounded-lg w-10 h-10 flex items-center justify-center text-xl text-white flex-shrink-0">
                      {app.icon}
                    </div>
                    <div className="min-w-0">
                      <span className="font-semibold text-sm text-[#1b5e20] block truncate">{app.name}</span>
                      <span className="text-[10px] text-gray-600 line-clamp-2">{app.description}</span>
                    </div>
                  </div>
                  <span className="text-[10px] text-gray-600">Suitability</span>
                  <div className="w-full h-1.5 bg-[#c8e6c9] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-[#4ecdc4] to-[#44a08d] rounded-full transition-all duration-500"
                      ref={el => { barRefs.current[i] = el; }}
                      style={{ width: '0%' }}
                    />
                  </div>
                  <span className="text-[10px] font-mono text-right w-full block text-[#2d6a4f]">{app.suitability}%</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Sustainability Impact Section */}
      <div className="mt-6 flex flex-col items-center flex-shrink-0 bg-white rounded-xl p-4 shadow border border-[#95e1d3]/30">
        <h3 className="text-base font-semibold mb-3 text-[#2d6a4f]">Sustainability Impact</h3>
        <div className="flex gap-4 mb-3 flex-wrap justify-center">
          <div className="bg-gradient-to-br from-[#a8e6cf] to-[#56ab2f]/30 rounded-lg px-5 py-3 flex flex-col items-center min-w-[120px] border border-[#56ab2f]/40">
            <span className="text-[#1b5e20] font-bold text-2xl">{properties.sustainability || 0}%</span>
            <span className="text-xs text-[#2d6a4f] font-medium">Sustainability Score</span>
          </div>
          <div className="bg-gradient-to-br from-[#b2dfdb] to-[#26a69a]/30 rounded-lg px-5 py-3 flex flex-col items-center min-w-[120px] border border-[#26a69a]/40">
            <span className="text-[#00695c] font-bold text-2xl">{properties.biodegradable || 0}%</span>
            <span className="text-xs text-[#004d40] font-medium">Biodegradable Score</span>
          </div>
          <div className="bg-gradient-to-br from-[#95e1d3] to-[#38ada9]/30 rounded-lg px-5 py-3 flex flex-col items-center min-w-[120px] border border-[#38ada9]/40">
            <span className="text-[#00695c] font-bold text-2xl">{properties.degradability || 0}%</span>
            <span className="text-xs text-[#004d40] font-medium">Degradability Score</span>
          </div>
          <div className="bg-gradient-to-br from-[#4ecdc4] to-[#44a08d]/30 rounded-lg px-5 py-3 flex flex-col items-center min-w-[120px] border border-[#44a08d]/40">
            <span className="text-[#00796b] font-bold text-2xl">{Math.floor((properties.ecoScore || 0) * 0.1)}</span>
            <span className="text-xs text-[#004d40] font-medium">Experiments Saved</span>
          </div>
        </div>
        <p className="text-xs text-gray-500 text-center max-w-lg">Higher sustainability and biodegradability scores indicate a greener polymer. Experiments saved is an estimate based on model predictions vs. lab trials.</p>
      </div>
    </div>
  );
};

export default ResultsPage;

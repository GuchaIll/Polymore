import React from 'react';
import { PlacedMolecule, HeuristicPredictedProperties } from '../../types';

interface PropertiesPanelProps {
  molecules: PlacedMolecule[];
  properties: HeuristicPredictedProperties | null;
  onRemoveMolecule: (id: number) => void;
}

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({
  molecules,
  properties,
  onRemoveMolecule
}) => {
  const totalAtoms = molecules.length;
  const totalBonds = molecules.reduce((sum, m) => sum + m.connections.length, 0) / 2;
  const totalWeight = molecules.reduce((sum, m) => sum + m.weight, 0);

  const PropertyBar: React.FC<{ type: string; value: number }> = ({ type, value }) => (
    <div className="h-1.5 bg-poly-border rounded-sm mt-1 overflow-hidden">
      <div
        className={`property-fill ${type} h-full rounded-sm transition-all duration-500`}
        style={{ width: `${value}%` }}
      />
    </div>
  );

  return (
    <div className="absolute bottom-5 right-5 w-[300px] bg-poly-sidebar/95 border-2 border-poly-border rounded-xl p-4 backdrop-blur-sm">
      <h3 className="text-poly-danger text-sm font-bold mb-3 uppercase">📊 Properties</h3>

      <div className="flex justify-between items-center mb-2">
        <span className="text-gray-500 text-sm">Total Atoms</span>
        <span className="text-white font-semibold">{totalAtoms}</span>
      </div>

      <div className="flex justify-between items-center mb-2">
        <span className="text-gray-500 text-sm">Total Bonds</span>
        <span className="text-white font-semibold">{Math.floor(totalBonds)}</span>
      </div>

      <div className="flex justify-between items-center mb-4">
        <span className="text-gray-500 text-sm">Molecular Weight</span>
        <span className="text-white font-semibold">{totalWeight.toFixed(1)} g/mol</span>
      </div>

      {/* Predicted Properties */}
      <div className="mt-4">
        <div className="flex justify-between items-center">
          <span className="text-gray-500 text-sm">Strength</span>
          <span className="text-white font-semibold">{properties?.strength?.toFixed(1) || '-'}</span>
        </div>
        <PropertyBar type="strength" value={properties?.strength || 0} />
      </div>

      <div className="mt-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-500 text-sm">Flexibility</span>
          <span className="text-white font-semibold">{properties?.flexibility?.toFixed(1) || '-'}</span>
        </div>
        <PropertyBar type="flexibility" value={properties?.flexibility || 0} />
      </div>

      <div className="mt-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-500 text-sm">Degradability</span>
          <span className="text-white font-semibold">{properties?.degradability?.toFixed(1) || '-'}</span>
        </div>
        <PropertyBar type="degradability" value={properties?.degradability || 0} />
      </div>

      <div className="mt-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-500 text-sm">🌱 Sustainability</span>
          <span className="text-white font-semibold">{properties?.sustainability?.toFixed(1) || '-'}</span>
        </div>
        <PropertyBar type="sustainability" value={properties?.sustainability || 0} />
      </div>

      {/* Structure List */}
      {molecules.length > 0 && (
        <div className="max-h-[150px] overflow-y-auto mt-4 pt-3 border-t border-poly-border">
          {molecules.map(mol => (
            <div
              key={mol.id}
              className="flex justify-between items-center p-2 bg-poly-bg rounded-md mb-1 text-xs"
            >
              <span className="text-white">{mol.name}</span>
              <button
                className="bg-transparent border-none text-poly-danger cursor-pointer px-1.5 hover:bg-poly-danger/20 rounded"
                onClick={() => onRemoveMolecule(mol.id)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default PropertiesPanel;
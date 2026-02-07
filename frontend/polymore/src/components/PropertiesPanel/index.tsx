import React from 'react';
import { BarChart3, Leaf } from 'lucide-react';
import { PlacedMolecule, HeuristicPredictedProperties } from '../../types';
import { PolymerValidationResult } from '../../util';

interface PropertiesPanelProps {
  molecules: PlacedMolecule[];
  properties: HeuristicPredictedProperties | null;
  validationResult?: PolymerValidationResult | null;
  onRemoveMolecule: (id: number) => void;
}

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({
  molecules,
  properties,
  validationResult,
  onRemoveMolecule
}) => {
  const totalAtoms = molecules.length;
  const totalBonds = molecules.reduce((sum, m) => sum + m.connections.length, 0) / 2;
  const totalWeight = molecules.reduce((sum, m) => sum + m.weight, 0);

  const PropertyBar: React.FC<{ type: string; value: number }> = ({ type, value }) => (
    <div className="h-1.5 bg-poly-light-border dark:bg-poly-border rounded-sm mt-1 overflow-hidden">
      <div
        className={`property-fill ${type} h-full rounded-sm transition-all duration-500`}
        style={{ width: `${value}%` }}
      />
    </div>
  );

  return (
    <div className="absolute bottom-5 right-5 w-[300px] bg-poly-light-sidebar/95 dark:bg-poly-sidebar/95 border-2 border-poly-light-border dark:border-poly-border rounded-xl p-4 backdrop-blur-sm">
      <h3 className="text-poly-light-accent dark:text-poly-danger text-sm font-bold mb-3 uppercase flex items-center gap-2">
        <BarChart3 className="w-4 h-4" /> Properties
      </h3>

      <div className="flex justify-between items-center mb-2">
        <span className="text-poly-light-muted dark:text-gray-500 text-sm">Total Atoms</span>
        <span className="text-poly-light-text dark:text-white font-semibold">{totalAtoms}</span>
      </div>

      <div className="flex justify-between items-center mb-2">
        <span className="text-poly-light-muted dark:text-gray-500 text-sm">Total Bonds</span>
        <span className="text-poly-light-text dark:text-white font-semibold">{Math.floor(totalBonds)}</span>
      </div>

      <div className="flex justify-between items-center mb-4">
        <span className="text-poly-light-muted dark:text-gray-500 text-sm">Molecular Weight</span>
        <span className="text-poly-light-text dark:text-white font-semibold">{totalWeight.toFixed(1)} g/mol</span>
      </div>

      {/* Validation Results */}
      {validationResult && (
        <div className="mb-4 p-2 rounded-md bg-poly-light-bg dark:bg-poly-bg border border-poly-light-border dark:border-poly-border">
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-sm font-semibold ${validationResult.isValid ? 'text-green-500' : 'text-red-500'}`}>
              {validationResult.isValid ? 'Valid' : 'Invalid'}
            </span>
            <span className="text-xs text-poly-light-muted dark:text-gray-500">
              {validationResult.polymerType}
            </span>
          </div>
          {validationResult.canonicalSmiles && (
            <div className="text-xs text-poly-light-text dark:text-white font-mono break-all mb-1">
              {validationResult.canonicalSmiles.length > 40 
                ? validationResult.canonicalSmiles.slice(0, 40) + '...' 
                : validationResult.canonicalSmiles}
            </div>
          )}
          {validationResult.molecularWeight && (
            <div className="text-xs text-poly-light-muted dark:text-gray-500">
              MW: {validationResult.molecularWeight.toFixed(2)}
            </div>
          )}
          {validationResult.errors && validationResult.errors.length > 0 && (
            <div className="text-xs text-red-400 mt-1">
              {validationResult.errors.slice(0, 2).map(e => e.message).join('; ')}
            </div>
          )}
          {validationResult.warnings && validationResult.warnings.length > 0 && (
            <div className="text-xs text-yellow-400 mt-1">
              {validationResult.warnings.slice(0, 1).map(w => w.message).join('; ')}
            </div>
          )}
        </div>
      )}

      {/* Predicted Properties */}
      <div className="mt-4">
        <div className="flex justify-between items-center">
          <span className="text-poly-light-muted dark:text-gray-500 text-sm">Strength</span>
          <span className="text-poly-light-text dark:text-white font-semibold">{properties?.strength?.toFixed(1) || '-'}</span>
        </div>
        <PropertyBar type="strength" value={properties?.strength || 0} />
      </div>

      <div className="mt-3">
        <div className="flex justify-between items-center">
          <span className="text-poly-light-muted dark:text-gray-500 text-sm">Flexibility</span>
          <span className="text-poly-light-text dark:text-white font-semibold">{properties?.flexibility?.toFixed(1) || '-'}</span>
        </div>
        <PropertyBar type="flexibility" value={properties?.flexibility || 0} />
      </div>

      <div className="mt-3">
        <div className="flex justify-between items-center">
          <span className="text-poly-light-muted dark:text-gray-500 text-sm">Degradability</span>
          <span className="text-poly-light-text dark:text-white font-semibold">{properties?.degradability?.toFixed(1) || '-'}</span>
        </div>
        <PropertyBar type="degradability" value={properties?.degradability || 0} />
      </div>

      <div className="mt-3">
        <div className="flex justify-between items-center">
          <span className="text-poly-light-muted dark:text-gray-500 text-sm flex items-center gap-1.5">
            <Leaf className="w-3.5 h-3.5 text-green-500" /> Sustainability
          </span>
          <span className="text-poly-light-text dark:text-white font-semibold">{properties?.sustainability?.toFixed(1) || '-'}</span>
        </div>
        <PropertyBar type="sustainability" value={properties?.sustainability || 0} />
      </div>

      {/* Structure List */}
      {molecules.length > 0 && (
        <div className="max-h-[150px] overflow-y-auto mt-4 pt-3 border-t border-poly-light-border dark:border-poly-border">
          {molecules.map(mol => (
            <div
              key={mol.id}
              className="flex justify-between items-center p-2 bg-poly-light-bg dark:bg-poly-bg rounded-md mb-1 text-xs"
            >
              <span className="text-poly-light-text dark:text-white">{mol.name}</span>
              <button
                className="bg-transparent border-none text-poly-light-danger dark:text-poly-danger cursor-pointer px-1.5 hover:bg-poly-light-danger/20 dark:hover:bg-poly-danger/20 rounded"
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
/**
 * Module: util/optimization.ts
 * Purpose: Rule-based polymer structure optimization
 * Inputs: PlacedMolecule arrays and optional validation results
 * Outputs: Optimized molecule positions and suggested fixes
 * 
 * Optimization strategies:
 * 1. Spatial - Adjust bond lengths and molecule spacing
 * 2. Validation - Fix common SMILES/structure issues
 * 3. Polymer Rules - Polymerizability, mechanical, sustainability analysis
 * 4. Auto-repair - Fix invalid SMILES automatically
 */

import { PlacedMolecule, Position } from '../types';
import { 
    PolymerValidationResult, 
    ValidationRuleCode,
    analyzePolymerStructure,
    PolymerAnalysis,
    repairSmiles,
    SmilesRepairResult,
    generateSmiles
} from './index';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Ideal bond length between connected molecules (in grid units) */
const IDEAL_BOND_LENGTH = 3.0;

/** Minimum acceptable bond length */
const MIN_BOND_LENGTH = 2.0;

/** Maximum acceptable bond length */
const MAX_BOND_LENGTH = 5.0;

/** Minimum distance between unconnected molecules to avoid overlap */
const MIN_MOLECULE_SPACING = 2.5;

/** Grid boundary (molecules must stay within this limit) */
const GRID_HALF_SIZE = 10;

/** Spring constant for bond length correction */
const BOND_SPRING_K = 0.3;

/** Repulsion constant for overlapping molecules */
const REPULSION_K = 0.5;

/** Maximum iterations for force-based optimization */
const MAX_ITERATIONS = 50;

/** Convergence threshold (stop if total movement is below this) */
const CONVERGENCE_THRESHOLD = 0.01;

// =============================================================================
// TYPES
// =============================================================================

export interface OptimizationResult {
  molecules: PlacedMolecule[];
  changes: OptimizationChange[];
  summary: string;
}

export interface OptimizationChange {
  type: 'position' | 'connection' | 'removal' | 'addition';
  moleculeId?: number;
  moleculeName?: string;
  description: string;
  oldValue?: any;
  newValue?: any;
}

export interface OptimizationSuggestion {
  title: string;
  description: string;
  action: 'auto' | 'manual';
  priority: 'high' | 'medium' | 'low';
  category?: 'polymerizability' | 'mechanical' | 'sustainability' | 'validation' | 'spatial';
  ruleCode?: ValidationRuleCode;
}

// =============================================================================
// POLYMER RULE THRESHOLDS
// =============================================================================

/** Thresholds for polymerizability rules */
const POLYMERIZABILITY_THRESHOLDS = {
  MIN_REACTIVE_SITES: 2,        // Rule P1: Minimum connection points
  MAX_REACTIVE_SITES: 4,        // Rule P2: Crosslink risk threshold
};

/** Thresholds for mechanical/physical behavior rules */
const MECHANICAL_THRESHOLDS = {
  HIGH_FLEXIBILITY_ROTATABLE: 5,   // Rule M1: Many rotatable bonds
  HIGH_AROMATIC_RATIO: 0.4,        // Rule M2: Aromatic atoms / total atoms
  LOW_POLAR_GROUPS: 2,             // Rule M3: Minimum polar groups for strength
};

/** Thresholds for sustainability rules */
const SUSTAINABILITY_THRESHOLDS = {
  HALOGEN_WARNING: 2,           // Rule S2: Halogen count warning
  HIGH_MW_THRESHOLD: 500,       // Rule S3: High molecular weight threshold
  MAX_ATOMS_FOR_DEGRADABILITY: 50,  // Rule S3: Atom count threshold
};

/**
 * Extended suggestion with repair information
 */
export interface PolymerRuleSuggestion extends OptimizationSuggestion {
  analysis?: Partial<PolymerAnalysis>;
  repairResult?: SmilesRepairResult;
}

// =============================================================================
// SPATIAL OPTIMIZATION
// =============================================================================

/**
 * Calculate Euclidean distance between two positions
 */
const distance = (p1: Position, p2: Position): number => {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dz = p2.z - p1.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

/**
 * Clamp a position to grid boundaries
 */
const clampToGrid = (pos: Position): Position => ({
  x: Math.max(-GRID_HALF_SIZE, Math.min(GRID_HALF_SIZE, pos.x)),
  y: pos.y,
  z: Math.max(-GRID_HALF_SIZE, Math.min(GRID_HALF_SIZE, pos.z))
});

/**
 * Calculate center of mass of all molecules
 */
const centerOfMass = (molecules: PlacedMolecule[]): Position => {
  if (molecules.length === 0) return { x: 0, y: 0, z: 0 };
  
  const sum = molecules.reduce(
    (acc, mol) => ({
      x: acc.x + mol.position.x,
      y: acc.y + mol.position.y,
      z: acc.z + mol.position.z
    }),
    { x: 0, y: 0, z: 0 }
  );
  
  return {
    x: sum.x / molecules.length,
    y: sum.y / molecules.length,
    z: sum.z / molecules.length
  };
};

/**
 * Optimize bond lengths using spring-like forces
 * Connected molecules are pulled/pushed to ideal bond length
 */
export const optimizeBondLengths = (molecules: PlacedMolecule[]): OptimizationResult => {
  if (molecules.length < 2) {
    return {
      molecules,
      changes: [],
      summary: 'Not enough molecules to optimize bonds'
    };
  }

  // Deep copy molecules
  let optimized = molecules.map(m => ({
    ...m,
    position: { ...m.position }
  }));

  const changes: OptimizationChange[] = [];
  let totalIterations = 0;

  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    totalIterations = iter + 1;
    let totalMovement = 0;

    // Calculate forces on each molecule
    const forces: Map<number, Position> = new Map();
    optimized.forEach(mol => {
      forces.set(mol.id, { x: 0, y: 0, z: 0 });
    });

    // Bond spring forces (connected molecules)
    optimized.forEach(mol => {
      mol.connections.forEach(connId => {
        // Only process each bond once (using lower ID)
        if (mol.id >= connId) return;

        const connMol = optimized.find(m => m.id === connId);
        if (!connMol) return;

        const dist = distance(mol.position, connMol.position);
        if (dist < 0.01) return; // Avoid division by zero

        // Spring force: F = k * (distance - idealLength)
        const displacement = dist - IDEAL_BOND_LENGTH;
        const forceMagnitude = BOND_SPRING_K * displacement;

        // Direction from mol to connMol
        const dx = (connMol.position.x - mol.position.x) / dist;
        const dz = (connMol.position.z - mol.position.z) / dist;

        // Apply force to both molecules (opposite directions)
        const force1 = forces.get(mol.id)!;
        const force2 = forces.get(connMol.id)!;

        force1.x += dx * forceMagnitude;
        force1.z += dz * forceMagnitude;
        force2.x -= dx * forceMagnitude;
        force2.z -= dz * forceMagnitude;
      });
    });

    // Apply forces and clamp to grid
    optimized = optimized.map(mol => {
      const force = forces.get(mol.id)!;
      const newPos = clampToGrid({
        x: mol.position.x + force.x,
        y: mol.position.y,
        z: mol.position.z + force.z
      });

      totalMovement += Math.abs(force.x) + Math.abs(force.z);

      return { ...mol, position: newPos };
    });

    // Check convergence
    if (totalMovement < CONVERGENCE_THRESHOLD) {
      break;
    }
  }

  // Track which molecules moved significantly
  molecules.forEach((original, idx) => {
    const newMol = optimized.find(m => m.id === original.id);
    if (!newMol) return;

    const dist = distance(original.position, newMol.position);
    if (dist > 0.1) {
      changes.push({
        type: 'position',
        moleculeId: original.id,
        moleculeName: original.name,
        description: `Repositioned ${original.name} to optimize bond length`,
        oldValue: original.position,
        newValue: newMol.position
      });
    }
  });

  return {
    molecules: optimized,
    changes,
    summary: `Bond optimization completed in ${totalIterations} iterations. ${changes.length} molecule(s) repositioned.`
  };
};

/**
 * Spread out overlapping/close molecules that are not connected
 */
export const spreadMolecules = (molecules: PlacedMolecule[]): OptimizationResult => {
  if (molecules.length < 2) {
    return {
      molecules,
      changes: [],
      summary: 'Not enough molecules to spread'
    };
  }

  let optimized = molecules.map(m => ({
    ...m,
    position: { ...m.position }
  }));

  const changes: OptimizationChange[] = [];

  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    let totalMovement = 0;

    // Calculate repulsion forces between non-connected molecules
    const forces: Map<number, Position> = new Map();
    optimized.forEach(mol => {
      forces.set(mol.id, { x: 0, y: 0, z: 0 });
    });

    for (let i = 0; i < optimized.length; i++) {
      for (let j = i + 1; j < optimized.length; j++) {
        const mol1 = optimized[i];
        const mol2 = optimized[j];

        // Skip connected molecules (they're handled by bond optimization)
        if (mol1.connections.includes(mol2.id)) continue;

        const dist = distance(mol1.position, mol2.position);
        if (dist >= MIN_MOLECULE_SPACING || dist < 0.01) continue;

        // Repulsion force for overlapping molecules
        const overlap = MIN_MOLECULE_SPACING - dist;
        const forceMagnitude = REPULSION_K * overlap;

        const dx = (mol2.position.x - mol1.position.x) / dist;
        const dz = (mol2.position.z - mol1.position.z) / dist;

        const force1 = forces.get(mol1.id)!;
        const force2 = forces.get(mol2.id)!;

        force1.x -= dx * forceMagnitude;
        force1.z -= dz * forceMagnitude;
        force2.x += dx * forceMagnitude;
        force2.z += dz * forceMagnitude;
      }
    }

    // Apply forces
    optimized = optimized.map(mol => {
      const force = forces.get(mol.id)!;
      const newPos = clampToGrid({
        x: mol.position.x + force.x,
        y: mol.position.y,
        z: mol.position.z + force.z
      });

      totalMovement += Math.abs(force.x) + Math.abs(force.z);
      return { ...mol, position: newPos };
    });

    if (totalMovement < CONVERGENCE_THRESHOLD) break;
  }

  // Track changes
  molecules.forEach(original => {
    const newMol = optimized.find(m => m.id === original.id);
    if (!newMol) return;

    const dist = distance(original.position, newMol.position);
    if (dist > 0.1) {
      changes.push({
        type: 'position',
        moleculeId: original.id,
        moleculeName: original.name,
        description: `Moved ${original.name} to reduce overlap`,
        oldValue: original.position,
        newValue: newMol.position
      });
    }
  });

  return {
    molecules: optimized,
    changes,
    summary: `Spread optimization: ${changes.length} molecule(s) moved to reduce overlap.`
  };
};

/**
 * Center the structure on the grid origin
 */
export const centerStructure = (molecules: PlacedMolecule[]): OptimizationResult => {
  if (molecules.length === 0) {
    return { molecules, changes: [], summary: 'No molecules to center' };
  }

  const center = centerOfMass(molecules);
  
  // Skip if already centered
  if (Math.abs(center.x) < 0.5 && Math.abs(center.z) < 0.5) {
    return { molecules, changes: [], summary: 'Structure already centered' };
  }

  const optimized = molecules.map(mol => ({
    ...mol,
    position: clampToGrid({
      x: mol.position.x - center.x,
      y: mol.position.y,
      z: mol.position.z - center.z
    })
  }));

  return {
    molecules: optimized,
    changes: [{
      type: 'position',
      description: `Centered structure (shifted by ${center.x.toFixed(1)}, ${center.z.toFixed(1)})`,
    }],
    summary: `Structure centered on grid origin.`
  };
};

// =============================================================================
// VALIDATION-BASED OPTIMIZATION
// =============================================================================

/**
 * Analyze validation errors and provide optimization suggestions
 */
export const analyzeValidationErrors = (
  validationResult: PolymerValidationResult
): OptimizationSuggestion[] => {
  const suggestions: OptimizationSuggestion[] = [];

  if (validationResult.isValid) {
    return suggestions;
  }

  validationResult.errors.forEach(error => {
    switch (error.code) {
      case ValidationRuleCode.RULE_2_DISCONNECTED:
        suggestions.push({
          title: 'Connect Fragments',
          description: 'Structure has disconnected parts. Connect all molecules with bonds to form a single polymer chain.',
          action: 'manual',
          priority: 'high'
        });
        break;

      case ValidationRuleCode.RULE_5_MIN_REACTIVE_SITES:
        suggestions.push({
          title: 'Add Reactive Groups',
          description: 'Add functional groups with reactive sites (e.g., -OH, -COOH, -NH2) to enable polymer chain growth.',
          action: 'manual',
          priority: 'high'
        });
        break;

      case ValidationRuleCode.RULE_6_FULLY_CAPPED:
        suggestions.push({
          title: 'Remove End Caps',
          description: 'Structure is fully saturated. Remove protecting groups or use monomers with open reactive sites.',
          action: 'manual',
          priority: 'high'
        });
        break;

      case ValidationRuleCode.CANVAS_SPATIAL_ERROR:
        suggestions.push({
          title: 'Optimize Positions',
          description: 'Bond distances are invalid. Use position optimization to fix molecular spacing.',
          action: 'auto',
          priority: 'medium'
        });
        break;

      case ValidationRuleCode.RULE_1_VALENCE:
        suggestions.push({
          title: 'Fix Valence Errors',
          description: 'Some atoms exceed their bond capacity. Remove excess bonds or use different molecules.',
          action: 'manual',
          priority: 'high'
        });
        break;

      case ValidationRuleCode.RULE_4_INVALID_SYNTAX:
        suggestions.push({
          title: 'Check SMILES Syntax',
          description: 'Invalid SMILES structure. Re-check molecule connections and try removing recently added bonds.',
          action: 'manual',
          priority: 'high'
        });
        break;

      case ValidationRuleCode.RULE_10_FORBIDDEN_ELEMENTS:
        suggestions.push({
          title: 'Remove Unsupported Elements',
          description: 'Structure contains elements not supported for ML prediction. Use only C, H, O, N, S, halogens, Si.',
          action: 'manual',
          priority: 'medium'
        });
        break;

      case ValidationRuleCode.RULE_11_RADICALS_CHARGES:
        suggestions.push({
          title: 'Neutralize Structure',
          description: 'Structure contains radicals or charges. These are unstable for polymer formation.',
          action: 'manual',
          priority: 'medium'
        });
        break;

      default:
        // Generic suggestion for other errors
        if (!suggestions.some(s => s.title === 'Review Structure')) {
          suggestions.push({
            title: 'Review Structure',
            description: `Validation error: ${error.message}. Review and adjust your polymer structure.`,
            action: 'manual',
            priority: 'medium'
          });
        }
    }
  });

  // Add warnings as low priority suggestions
  validationResult.warnings.forEach(warning => {
    suggestions.push({
      title: 'Warning',
      description: warning.message,
      action: 'manual',
      priority: 'low'
    });
  });

  return suggestions;
};

// =============================================================================
// COMPREHENSIVE POLYMER RULE ANALYSIS
// =============================================================================

/**
 * Analyze polymer structure against all rule categories:
 * - Polymerizability (P1, P2)  
 * - Mechanical behavior (M1, M2, M3)
 * - Sustainability (S1, S2, S3)
 * 
 * @param smiles - SMILES string to analyze
 * @returns Array of suggestions with detailed recommendations
 */
export const analyzePolymerRules = async (
  smiles: string
): Promise<PolymerRuleSuggestion[]> => {
  const suggestions: PolymerRuleSuggestion[] = [];
  
  if (!smiles || smiles.trim() === '') {
    return suggestions;
  }
  
  // Get comprehensive polymer analysis
  const analysis = await analyzePolymerStructure(smiles);
  
  // === POLYMERIZABILITY RULES ===
  
  // Rule P1: Minimum reactive sites (already in validation, but add suggestion text)
  if (analysis.reactiveSites < POLYMERIZABILITY_THRESHOLDS.MIN_REACTIVE_SITES) {
    suggestions.push({
      title: 'Cannot Polymerize - Too Few Reactive Sites',
      description: `Found ${analysis.reactiveSites} reactive site(s). Add at least two reactive functional groups (-OH, -COOH, -NH2, or [*]) to enable chain formation.`,
      action: 'manual',
      priority: 'high',
      category: 'polymerizability',
      ruleCode: ValidationRuleCode.RULE_5_MIN_REACTIVE_SITES,
      analysis: { reactiveSites: analysis.reactiveSites }
    });
  }
  
  // Rule P2: Over-functionalized (crosslink risk)
  if (analysis.reactiveSites > POLYMERIZABILITY_THRESHOLDS.MAX_REACTIVE_SITES) {
    suggestions.push({
      title: 'Crosslink Risk - Over-Functionalized',
      description: `Found ${analysis.reactiveSites} reactive sites. Too many reactive sites may cause crosslinking and form brittle gel networks. Reduce functionality for linear polymers.`,
      action: 'manual',
      priority: 'medium',
      category: 'polymerizability',
      ruleCode: ValidationRuleCode.RULE_P2_CROSSLINK_RISK,
      analysis: { reactiveSites: analysis.reactiveSites }
    });
  }
  
  // === MECHANICAL BEHAVIOR RULES ===
  
  // Rule M1: Too flexible (low Tg prediction)
  const aromaticRatio = analysis.atomCount > 0 
    ? analysis.aromaticAtoms / analysis.atomCount 
    : 0;
  
  if (analysis.rotatableBonds > MECHANICAL_THRESHOLDS.HIGH_FLEXIBILITY_ROTATABLE 
      && analysis.aromaticRings === 0) {
    suggestions.push({
      title: 'Low Rigidity - High Flexibility',
      description: `Structure has ${analysis.rotatableBonds} rotatable bonds and no rigid rings. This may result in a low glass transition temperature (Tg). Adding aromatic or cyclic groups may increase stiffness.`,
      action: 'manual',
      priority: 'low',
      category: 'mechanical',
      ruleCode: ValidationRuleCode.RULE_M1_TOO_FLEXIBLE,
      analysis: { rotatableBonds: analysis.rotatableBonds, aromaticRings: analysis.aromaticRings }
    });
  }
  
  // Rule M2: Too brittle (high aromatic content)
  if (aromaticRatio > MECHANICAL_THRESHOLDS.HIGH_AROMATIC_RATIO) {
    suggestions.push({
      title: 'Brittleness Risk - High Rigidity',
      description: `High aromatic content (${(aromaticRatio * 100).toFixed(0)}% aromatic atoms) increases rigidity but may cause brittleness. Consider adding flexible spacers (-CH2- or ether linkages) to improve toughness.`,
      action: 'manual',
      priority: 'low',
      category: 'mechanical',
      ruleCode: ValidationRuleCode.RULE_M2_TOO_BRITTLE,
      analysis: { aromaticAtoms: analysis.aromaticAtoms, atomCount: analysis.atomCount }
    });
  }
  
  // Rule M3: Weak intermolecular forces
  const polarGroups = analysis.hBondDonors + analysis.hBondAcceptors;
  if (polarGroups < MECHANICAL_THRESHOLDS.LOW_POLAR_GROUPS && analysis.atomCount > 5) {
    suggestions.push({
      title: 'Weak Intermolecular Forces',
      description: `Only ${polarGroups} polar group(s) detected. Lack of polar groups reduces intermolecular interactions. Add -OH, -NH, or ester groups to improve material strength through hydrogen bonding.`,
      action: 'manual',
      priority: 'low',
      category: 'mechanical',
      ruleCode: ValidationRuleCode.RULE_M3_WEAK_FORCES,
      analysis: { hBondDonors: analysis.hBondDonors, hBondAcceptors: analysis.hBondAcceptors }
    });
  }
  
  // === SUSTAINABILITY RULES ===
  
  // Rule S1: Non-degradable backbone
  if (analysis.totalDegradableLinkages === 0 && analysis.atomCount > 10) {
    suggestions.push({
      title: 'Non-Biodegradable Backbone',
      description: 'Backbone contains only C-C bonds without hydrolyzable linkages (no esters, amides, or carbonates). Consider adding ester or amide linkages to enable biodegradability.',
      action: 'manual',
      priority: 'medium',
      category: 'sustainability',
      ruleCode: ValidationRuleCode.RULE_S1_NON_DEGRADABLE,
      analysis: { 
        esterLinkages: analysis.esterLinkages, 
        amideLinkages: analysis.amideLinkages,
        totalDegradableLinkages: analysis.totalDegradableLinkages 
      }
    });
  }
  
  // Rule S2: Halogen warning
  if (analysis.totalHalogens >= SUSTAINABILITY_THRESHOLDS.HALOGEN_WARNING) {
    const halogenDetails = [];
    if (analysis.fluorineCount > 0) halogenDetails.push(`F:${analysis.fluorineCount}`);
    if (analysis.chlorineCount > 0) halogenDetails.push(`Cl:${analysis.chlorineCount}`);
    if (analysis.bromineCount > 0) halogenDetails.push(`Br:${analysis.bromineCount}`);
    if (analysis.iodineCount > 0) halogenDetails.push(`I:${analysis.iodineCount}`);
    
    suggestions.push({
      title: 'Environmental Concern - Halogens',
      description: `Found ${analysis.totalHalogens} halogen atom(s) (${halogenDetails.join(', ')}). Halogenated polymers are less environmentally friendly and harder to recycle. Consider non-halogen alternatives for greener design.`,
      action: 'manual',
      priority: 'medium',
      category: 'sustainability',
      ruleCode: ValidationRuleCode.RULE_S2_HALOGEN_WARNING,
      analysis: { 
        totalHalogens: analysis.totalHalogens,
        fluorineCount: analysis.fluorineCount,
        chlorineCount: analysis.chlorineCount,
        bromineCount: analysis.bromineCount,
        iodineCount: analysis.iodineCount
      }
    });
  }
  
  // Rule S3: High MW repeat unit
  if (analysis.atomCount > SUSTAINABILITY_THRESHOLDS.MAX_ATOMS_FOR_DEGRADABILITY) {
    suggestions.push({
      title: 'Large Repeat Unit',
      description: `Structure has ${analysis.atomCount} atoms (estimated MW: ~${analysis.estimatedMW}). Large repeat units may hinder degradability and processing. Consider simplifying the monomer for better processability.`,
      action: 'manual',
      priority: 'low',
      category: 'sustainability',
      ruleCode: ValidationRuleCode.RULE_S3_HIGH_MW_UNIT,
      analysis: { atomCount: analysis.atomCount, estimatedMW: analysis.estimatedMW }
    });
  }
  
  return suggestions;
};

/**
 * Attempt to auto-repair invalid SMILES and provide feedback
 * 
 * @param smiles - Potentially invalid SMILES string
 * @returns Repair result with suggestions
 */
export const attemptSmilesRepair = async (
  smiles: string
): Promise<{ repairResult: SmilesRepairResult; suggestions: PolymerRuleSuggestion[] }> => {
  const suggestions: PolymerRuleSuggestion[] = [];
  
  const repairResult = await repairSmiles(smiles);
  
  if (repairResult.success) {
    if (repairResult.wasModified) {
      suggestions.push({
        title: 'SMILES Auto-Repaired',
        description: `Original SMILES was modified. Changes: ${repairResult.repairSteps.join(', ')}`,
        action: 'auto',
        priority: 'medium',
        category: 'validation',
        repairResult
      });
    }
  } else {
    suggestions.push({
      title: 'SMILES Repair Failed',
      description: `Could not repair SMILES. Attempted: ${repairResult.repairSteps.join(', ')}. ${repairResult.error || ''}`,
      action: 'manual',
      priority: 'high',
      category: 'validation',
      ruleCode: ValidationRuleCode.RULE_4_INVALID_SYNTAX,
      repairResult
    });
  }
  
  return { repairResult, suggestions };
};

/**
 * Get comprehensive optimization suggestions combining all analyses:
 * - Spatial issues (bond lengths, overlaps)
 * - Validation errors
 * - Polymer rules (polymerizability, mechanical, sustainability)
 * - Auto-repair suggestions
 * 
 * @param molecules - Array of placed molecules
 * @param validationResult - Optional validation result
 * @param attemptRepair - Whether to attempt SMILES repair
 * @returns Comprehensive array of suggestions sorted by priority
 */
export const getComprehensiveSuggestions = async (
  molecules: PlacedMolecule[],
  validationResult?: PolymerValidationResult,
  attemptRepair: boolean = true
): Promise<PolymerRuleSuggestion[]> => {
  const allSuggestions: PolymerRuleSuggestion[] = [];
  
  if (molecules.length === 0) {
    allSuggestions.push({
      title: 'Add Molecules',
      description: 'Add molecules to the canvas to begin building your polymer.',
      action: 'manual',
      priority: 'high',
      category: 'spatial'
    });
    return allSuggestions;
  }
  
  // Generate SMILES for analysis
  let smiles = '';
  try {
    smiles = await generateSmiles(molecules);
  } catch {
    allSuggestions.push({
      title: 'Cannot Generate SMILES',
      description: 'Unable to generate SMILES from current structure. Check molecule connections.',
      action: 'manual',
      priority: 'high',
      category: 'validation'
    });
  }
  
  // Attempt repair if SMILES is available and requested
  if (smiles && attemptRepair) {
    const { suggestions: repairSuggestions, repairResult } = await attemptSmilesRepair(smiles);
    
    // If repair was successful and modified, use repaired SMILES for analysis
    if (repairResult.success && repairResult.wasModified) {
      allSuggestions.push(...repairSuggestions);
      smiles = repairResult.canonical || repairResult.repaired;
    } else if (!repairResult.success) {
      allSuggestions.push(...repairSuggestions);
    }
  }
  
  // Run polymer rule analysis if we have valid SMILES
  if (smiles) {
    const ruleSuggestions = await analyzePolymerRules(smiles);
    allSuggestions.push(...ruleSuggestions);
  }
  
  // Add validation-based suggestions
  if (validationResult) {
    const validationSuggestions = analyzeValidationErrors(validationResult);
    // Add category to validation suggestions
    validationSuggestions.forEach(s => {
      allSuggestions.push({
        ...s,
        category: 'validation'
      });
    });
  }
  
  // Add spatial suggestions
  const spatialSuggestions = checkSpatialIssues(molecules);
  allSuggestions.push(...spatialSuggestions);
  
  // Sort by priority: high > medium > low
  const priorityOrder = { high: 0, medium: 1, low: 2 };
  allSuggestions.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);
  
  // De-duplicate suggestions with same title
  const seen = new Set<string>();
  return allSuggestions.filter(s => {
    if (seen.has(s.title)) return false;
    seen.add(s.title);
    return true;
  });
};

/**
 * Check for spatial issues in molecule placement
 * @param molecules - Array of placed molecules
 * @returns Suggestions for spatial improvements
 */
const checkSpatialIssues = (molecules: PlacedMolecule[]): PolymerRuleSuggestion[] => {
  const suggestions: PolymerRuleSuggestion[] = [];
  
  // Check for disconnected fragments
  const visited = new Set<number>();
  const queue = [molecules[0].id];
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (visited.has(current)) continue;
    visited.add(current);
    const mol = molecules.find(m => m.id === current);
    mol?.connections.forEach(c => {
      if (!visited.has(c)) queue.push(c);
    });
  }

  if (visited.size < molecules.length) {
    suggestions.push({
      title: 'Disconnected Fragments',
      description: `Structure has ${molecules.length - visited.size + 1} disconnected fragments. Use auto-connect or manually bond molecules.`,
      action: 'auto',
      priority: 'high',
      category: 'spatial',
      ruleCode: ValidationRuleCode.RULE_2_DISCONNECTED
    });
  }

  // Check for isolated molecules
  const isolatedCount = molecules.filter(m => m.connections.length === 0).length;
  if (isolatedCount > 0 && molecules.length > 1) {
    suggestions.push({
      title: 'Isolated Molecules',
      description: `${isolatedCount} molecule(s) have no bonds. Connect them to form a polymer chain.`,
      action: 'manual',
      priority: 'high',
      category: 'spatial'
    });
  }

  // Check for bond length issues
  let shortBonds = 0;
  let longBonds = 0;
  molecules.forEach(mol => {
    mol.connections.forEach(connId => {
      if (mol.id >= connId) return;
      const conn = molecules.find(m => m.id === connId);
      if (conn) {
        const dist = distance(mol.position, conn.position);
        if (dist < MIN_BOND_LENGTH) shortBonds++;
        else if (dist > MAX_BOND_LENGTH) longBonds++;
      }
    });
  });

  if (shortBonds > 0 || longBonds > 0) {
    suggestions.push({
      title: 'Bond Length Issues',
      description: `${shortBonds} bond(s) too short, ${longBonds} bond(s) too long. Run position optimization to fix.`,
      action: 'auto',
      priority: 'medium',
      category: 'spatial',
      ruleCode: ValidationRuleCode.CANVAS_SPATIAL_ERROR
    });
  }

  return suggestions;
};

/**
 * Attempt to auto-fix disconnected fragments by connecting nearest unconnected molecules
 */
export const autoConnectFragments = (molecules: PlacedMolecule[]): OptimizationResult => {
  if (molecules.length < 2) {
    return { molecules, changes: [], summary: 'Not enough molecules to connect' };
  }

  // Find connected components using BFS
  const findComponents = (): Set<number>[] => {
    const visited = new Set<number>();
    const components: Set<number>[] = [];

    molecules.forEach(mol => {
      if (visited.has(mol.id)) return;

      const component = new Set<number>();
      const queue = [mol.id];

      while (queue.length > 0) {
        const currentId = queue.shift()!;
        if (visited.has(currentId)) continue;

        visited.add(currentId);
        component.add(currentId);

        const current = molecules.find(m => m.id === currentId);
        current?.connections.forEach(connId => {
          if (!visited.has(connId)) queue.push(connId);
        });
      }

      components.push(component);
    });

    return components;
  };

  const components = findComponents();
  
  if (components.length <= 1) {
    return { molecules, changes: [], summary: 'Structure is already fully connected' };
  }

  // Deep copy for modifications
  let optimized = molecules.map(m => ({
    ...m,
    connections: [...m.connections]
  }));

  const changes: OptimizationChange[] = [];

  // Connect components by finding nearest molecules between each pair
  while (components.length > 1) {
    let minDist = Infinity;
    let bestPair: [number, number] | null = null;

    // Find nearest pair between first component and others
    const comp1 = components[0];
    
    for (let i = 1; i < components.length; i++) {
      const comp2 = components[i];

      comp1.forEach(id1 => {
        const mol1 = optimized.find(m => m.id === id1)!;
        comp2.forEach(id2 => {
          const mol2 = optimized.find(m => m.id === id2)!;
          const dist = distance(mol1.position, mol2.position);

          if (dist < minDist) {
            minDist = dist;
            bestPair = [id1, id2];
          }
        });
      });
    }

    if (bestPair) {
      const id1 = bestPair[0];
      const id2 = bestPair[1];
      const mol1 = optimized.find(m => m.id === id1)!;
      const mol2 = optimized.find(m => m.id === id2)!;

      // Create connection
      mol1.connections.push(id2);
      mol2.connections.push(id1);

      changes.push({
        type: 'connection',
        description: `Auto-connected ${mol1.name} to ${mol2.name} (distance: ${minDist.toFixed(1)})`,
        moleculeId: id1,
        moleculeName: mol1.name
      });

      // Merge components
      const comp2Idx = components.findIndex(c => c.has(id2));
      components[comp2Idx].forEach(id => components[0].add(id));
      components.splice(comp2Idx, 1);
    } else {
      break;
    }
  }

  return {
    molecules: optimized,
    changes,
    summary: `Auto-connected ${changes.length} fragment(s) to form a single polymer.`
  };
};

// =============================================================================
// COMBINED OPTIMIZATION
// =============================================================================

/**
 * Run full optimization pipeline:
 * 1. Center structure
 * 2. Optimize bond lengths
 * 3. Spread overlapping molecules
 * 4. Auto-connect if disconnected (optional)
 */
export const optimizeStructure = (
  molecules: PlacedMolecule[],
  options: {
    autoConnect?: boolean;
    validationResult?: PolymerValidationResult;
  } = {}
): OptimizationResult => {
  if (molecules.length === 0) {
    return { molecules: [], changes: [], summary: 'No molecules to optimize' };
  }

  const allChanges: OptimizationChange[] = [];
  const summaries: string[] = [];
  let currentMolecules = molecules;

  // Step 1: Auto-connect fragments if requested and disconnected
  if (options.autoConnect) {
    const connectResult = autoConnectFragments(currentMolecules);
    if (connectResult.changes.length > 0) {
      currentMolecules = connectResult.molecules;
      allChanges.push(...connectResult.changes);
      summaries.push(connectResult.summary);
    }
  }

  // Step 2: Center structure
  const centerResult = centerStructure(currentMolecules);
  if (centerResult.changes.length > 0) {
    currentMolecules = centerResult.molecules;
    allChanges.push(...centerResult.changes);
    summaries.push(centerResult.summary);
  }

  // Step 3: Optimize bond lengths
  const bondResult = optimizeBondLengths(currentMolecules);
  if (bondResult.changes.length > 0) {
    currentMolecules = bondResult.molecules;
    allChanges.push(...bondResult.changes);
    summaries.push(bondResult.summary);
  }

  // Step 4: Spread overlapping molecules
  const spreadResult = spreadMolecules(currentMolecules);
  if (spreadResult.changes.length > 0) {
    currentMolecules = spreadResult.molecules;
    allChanges.push(...spreadResult.changes);
    summaries.push(spreadResult.summary);
  }

  // Generate validation-based suggestions if validation result provided
  if (options.validationResult && !options.validationResult.isValid) {
    const suggestions = analyzeValidationErrors(options.validationResult);
    suggestions.forEach(s => {
      allChanges.push({
        type: 'addition',
        description: `Suggestion: ${s.title} - ${s.description}`
      });
    });
  }

  const finalSummary = allChanges.length > 0
    ? `Optimization complete: ${allChanges.filter(c => c.type !== 'addition').length} changes applied. ${summaries.join(' ')}`
    : 'Structure is already optimized.';

  return {
    molecules: currentMolecules,
    changes: allChanges,
    summary: finalSummary
  };
};

/**
 * Get optimization suggestions without applying changes
 * 
 * NOTE: For comprehensive analysis including polymer rules (mechanical,
 * sustainability, polymerizability), use getComprehensiveSuggestions() instead.
 * This function provides basic spatial and validation suggestions only.
 * 
 * @deprecated Use getComprehensiveSuggestions for full analysis
 */
export const getOptimizationSuggestions = (
  molecules: PlacedMolecule[],
  validationResult?: PolymerValidationResult
): OptimizationSuggestion[] => {
  const suggestions: OptimizationSuggestion[] = [];

  if (molecules.length === 0) {
    suggestions.push({
      title: 'Add Molecules',
      description: 'Add molecules to the canvas to begin building your polymer.',
      action: 'manual',
      priority: 'high'
    });
    return suggestions;
  }

  // Check for disconnected fragments
  const visited = new Set<number>();
  const queue = [molecules[0].id];
  while (queue.length > 0) {
    const current = queue.shift()!;
    if (visited.has(current)) continue;
    visited.add(current);
    const mol = molecules.find(m => m.id === current);
    mol?.connections.forEach(c => {
      if (!visited.has(c)) queue.push(c);
    });
  }

  if (visited.size < molecules.length) {
    suggestions.push({
      title: 'Connect Fragments',
      description: `Structure has ${molecules.length - visited.size + 1} disconnected fragments. Use auto-connect or manually bond molecules.`,
      action: 'auto',
      priority: 'high'
    });
  }

  // Check for molecules with no connections
  const isolatedCount = molecules.filter(m => m.connections.length === 0).length;
  if (isolatedCount > 0 && molecules.length > 1) {
    suggestions.push({
      title: 'Bond Isolated Molecules',
      description: `${isolatedCount} molecule(s) have no bonds. Connect them to form a polymer chain.`,
      action: 'manual',
      priority: 'high'
    });
  }

  // Check for bond length issues
  let shortBonds = 0;
  let longBonds = 0;
  molecules.forEach(mol => {
    mol.connections.forEach(connId => {
      if (mol.id >= connId) return;
      const conn = molecules.find(m => m.id === connId);
      if (conn) {
        const dist = distance(mol.position, conn.position);
        if (dist < MIN_BOND_LENGTH) shortBonds++;
        else if (dist > MAX_BOND_LENGTH) longBonds++;
      }
    });
  });

  if (shortBonds > 0 || longBonds > 0) {
    suggestions.push({
      title: 'Adjust Bond Lengths',
      description: `${shortBonds} bond(s) too short, ${longBonds} bond(s) too long. Run position optimization.`,
      action: 'auto',
      priority: 'medium'
    });
  }

  // Add validation-based suggestions
  if (validationResult) {
    suggestions.push(...analyzeValidationErrors(validationResult));
  }

  return suggestions;
};

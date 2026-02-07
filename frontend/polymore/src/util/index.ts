/**
 * Module: util/index.ts
 * Purpose: Polymer SMILES validation and generation using RDKit.js (WebAssembly)
 * Inputs: PlacedMolecule arrays from canvas
 * Outputs: Validated SMILES, canonical forms, polymer properties
 * 
 * Uses @rdkit/rdkit for chemically accurate validation in the browser.
 * RDKit.js is loaded via CDN script tag in public/index.html
 * 
 * Validation Rules:
 * - Layer 1: Basic chemistry validity (valence, syntax, ring closures)
 * - Layer 2: Polymer-specific (reactive sites, connectivity)
 * - Layer 3: ML-friendly constraints (allowed atoms, size limits)
 */

import { PlacedMolecule, Position } from "../types";
import { predictTier1, predictTier2 } from "../services/services";

// =============================================================================
// VALIDATION RULE DEFINITIONS
// =============================================================================

/**
 * Validation rule codes for specific error identification
 */
export enum ValidationRuleCode {
    // Layer 1 - Basic Chemistry Validity
    RULE_1_VALENCE = "RULE_1_VALENCE",
    RULE_2_DISCONNECTED = "RULE_2_DISCONNECTED",
    RULE_3_RING_CLOSURE = "RULE_3_RING_CLOSURE",
    RULE_4_INVALID_SYNTAX = "RULE_4_INVALID_SYNTAX",

    // Layer 2 - Polymer-Specific Rules
    RULE_5_MIN_REACTIVE_SITES = "RULE_5_MIN_REACTIVE_SITES",
    RULE_6_FULLY_CAPPED = "RULE_6_FULLY_CAPPED",
    RULE_7_STERIC_HINDRANCE = "RULE_7_STERIC_HINDRANCE",
    RULE_8_REPEAT_SIZE = "RULE_8_REPEAT_SIZE",

    // Layer 3 - ML-Friendly Rules
    RULE_9_NOT_CANONICAL = "RULE_9_NOT_CANONICAL",
    RULE_10_FORBIDDEN_ELEMENTS = "RULE_10_FORBIDDEN_ELEMENTS",
    RULE_11_RADICALS_CHARGES = "RULE_11_RADICALS_CHARGES",

    // Polymerizability Rules
    RULE_P2_CROSSLINK_RISK = "RULE_P2_CROSSLINK_RISK",

    // Mechanical/Physical Rules
    RULE_M1_TOO_FLEXIBLE = "RULE_M1_TOO_FLEXIBLE",
    RULE_M2_TOO_BRITTLE = "RULE_M2_TOO_BRITTLE",
    RULE_M3_WEAK_FORCES = "RULE_M3_WEAK_FORCES",

    // Sustainability Rules
    RULE_S1_NON_DEGRADABLE = "RULE_S1_NON_DEGRADABLE",
    RULE_S2_HALOGEN_WARNING = "RULE_S2_HALOGEN_WARNING",
    RULE_S3_HIGH_MW_UNIT = "RULE_S3_HIGH_MW_UNIT",

    // Canvas/Placement Rules
    CANVAS_NO_MOLECULES = "CANVAS_NO_MOLECULES",
    CANVAS_INVALID_STRUCTURE = "CANVAS_INVALID_STRUCTURE",
    CANVAS_SPATIAL_ERROR = "CANVAS_SPATIAL_ERROR",

    // General
    RDKIT_NOT_LOADED = "RDKIT_NOT_LOADED",
    GENERATION_FAILED = "GENERATION_FAILED"
}

/**
 * Human-readable error messages for each validation rule
 */
export const ValidationRuleMessages: Record<ValidationRuleCode, { title: string; description: string; suggestion: string }> = {
    // Layer 1 - Basic Chemistry
    [ValidationRuleCode.RULE_1_VALENCE]: {
        title: "Valence Error",
        description: "Atom has exceeded its maximum bond capacity. Each element has a maximum number of bonds: C=4, O=2, N=3, H=1, halogens=1.",
        suggestion: "Check your molecular structure for atoms with too many bonds."
    },
    [ValidationRuleCode.RULE_2_DISCONNECTED]: {
        title: "Disconnected Fragments",
        description: "The structure contains multiple disconnected molecular fragments. A valid polymer should be a single connected component.",
        suggestion: "Connect all molecular groups together or remove isolated fragments."
    },
    [ValidationRuleCode.RULE_3_RING_CLOSURE]: {
        title: "Unclosed Ring",
        description: "Ring notation is incomplete. Ring digits must come in pairs to form closed rings.",
        suggestion: "Ensure all ring numbers have matching opening and closing positions."
    },
    [ValidationRuleCode.RULE_4_INVALID_SYNTAX]: {
        title: "Invalid SMILES Syntax",
        description: "The SMILES string contains syntax errors and cannot be parsed by RDKit.",
        suggestion: "Check for unbalanced parentheses, invalid characters, or malformed notation."
    },

    // Layer 2 - Polymer-Specific
    [ValidationRuleCode.RULE_5_MIN_REACTIVE_SITES]: {
        title: "Insufficient Reactive Sites",
        description: "A polymer repeat unit must have at least 2 reactive sites (connection points) to form a chain.",
        suggestion: "Add functional groups with reactive sites (e.g., -OH, -COOH, -NH2) or connection points [*]."
    },
    [ValidationRuleCode.RULE_6_FULLY_CAPPED]: {
        title: "Fully Saturated Structure",
        description: "All bonding positions are occupied. This structure cannot extend to form a polymer chain.",
        suggestion: "Remove end-caps or add reactive functional groups that can form new bonds."
    },
    [ValidationRuleCode.RULE_7_STERIC_HINDRANCE]: {
        title: "Steric Hindrance Warning",
        description: "The structure may have significant steric strain that could prevent polymerization.",
        suggestion: "Consider simplifying bulky side groups or adjusting ring configurations."
    },
    [ValidationRuleCode.RULE_8_REPEAT_SIZE]: {
        title: "Repeat Unit Too Large",
        description: "The polymer repeat unit exceeds the maximum allowed size (100 atoms). Large units cause computational issues.",
        suggestion: "Break down into smaller repeating units or simplify the structure."
    },

    // Layer 3 - ML-Friendly
    [ValidationRuleCode.RULE_9_NOT_CANONICAL]: {
        title: "Non-Canonical Form",
        description: "The SMILES is not in canonical form. This can cause duplicate entries in ML training.",
        suggestion: "Use the canonical SMILES representation provided."
    },
    [ValidationRuleCode.RULE_10_FORBIDDEN_ELEMENTS]: {
        title: "Unsupported Elements",
        description: "The structure contains elements not supported for polymer ML predictions. Allowed: C, H, O, N, S, F, Cl, Br, I, Si.",
        suggestion: "Remove or replace metal atoms and other unsupported elements."
    },
    [ValidationRuleCode.RULE_11_RADICALS_CHARGES]: {
        title: "Radicals or Charges Detected",
        description: "The structure contains radical species or formal charges that are typically unstable.",
        suggestion: "Neutralize charges and complete radical valences for stable polymer structures."
    },

    // Polymerizability
    [ValidationRuleCode.RULE_P2_CROSSLINK_RISK]: {
        title: "Crosslink Risk - Over-Functionalized",
        description: "Too many reactive sites (>4) may cause crosslinking and form brittle networks instead of linear polymers.",
        suggestion: "Reduce functionality for linear polymers. Remove some reactive groups to prevent gel formation."
    },

    // Mechanical/Physical Behavior
    [ValidationRuleCode.RULE_M1_TOO_FLEXIBLE]: {
        title: "Low Rigidity - High Flexibility",
        description: "Structure contains many rotatable bonds and no rigid rings. This may result in a low glass transition temperature (Tg).",
        suggestion: "Adding aromatic or cyclic groups may increase stiffness and Tg."
    },
    [ValidationRuleCode.RULE_M2_TOO_BRITTLE]: {
        title: "High Rigidity - Brittleness Risk",
        description: "High aromatic content or many double bonds increases rigidity but may cause brittleness.",
        suggestion: "Consider adding flexible spacers (-CH2- or ether linkages) to improve toughness."
    },
    [ValidationRuleCode.RULE_M3_WEAK_FORCES]: {
        title: "Weak Intermolecular Forces",
        description: "Lack of polar groups reduces intermolecular interactions, potentially weakening the material.",
        suggestion: "Add -OH, -NH, or ester groups to improve strength through hydrogen bonding."
    },

    // Sustainability
    [ValidationRuleCode.RULE_S1_NON_DEGRADABLE]: {
        title: "Non-Biodegradable Backbone",
        description: "Backbone contains only C-C bonds without hydrolyzable linkages. This may limit biodegradability.",
        suggestion: "Consider ester, amide, or carbonate linkages to enable biodegradability."
    },
    [ValidationRuleCode.RULE_S2_HALOGEN_WARNING]: {
        title: "Environmental Concern - Halogens",
        description: "Halogenated polymers are less environmentally friendly and harder to recycle.",
        suggestion: "Consider non-halogen alternatives for greener polymer design."
    },
    [ValidationRuleCode.RULE_S3_HIGH_MW_UNIT]: {
        title: "Large Repeat Unit",
        description: "Large repeat units may hinder degradability and processing efficiency.",
        suggestion: "Consider simplifying the monomer for better processability and degradation."
    },

    // Canvas/Placement
    [ValidationRuleCode.CANVAS_NO_MOLECULES]: {
        title: "No Molecules Placed",
        description: "The canvas is empty. Add molecules to create a polymer structure.",
        suggestion: "Drag molecules from the sidebar onto the canvas."
    },
    [ValidationRuleCode.CANVAS_INVALID_STRUCTURE]: {
        title: "Invalid Canvas Structure",
        description: "The placed molecules have invalid data (missing positions, IDs, or SMILES).",
        suggestion: "Try removing and re-placing the molecules."
    },
    [ValidationRuleCode.CANVAS_SPATIAL_ERROR]: {
        title: "Spatial Arrangement Error",
        description: "Bond distances are invalid - molecules are either too far apart or overlapping.",
        suggestion: "Adjust molecule positions to appropriate bonding distances."
    },

    // General
    [ValidationRuleCode.RDKIT_NOT_LOADED]: {
        title: "Chemistry Engine Not Ready",
        description: "RDKit.js WebAssembly module is still loading.",
        suggestion: "Please wait a moment and try again."
    },
    [ValidationRuleCode.GENERATION_FAILED]: {
        title: "SMILES Generation Failed",
        description: "Failed to generate a SMILES string from the current molecular arrangement.",
        suggestion: "Check that molecules are properly connected."
    }
};

/**
 * A single validation error with rule code and context
 */
export interface ValidationError {
    code: ValidationRuleCode;
    message: string;
    details?: string;
    moleculeName?: string;
    smiles?: string;
}

/**
 * Comprehensive validation result with detailed errors
 */
export interface PolymerValidationResult {
    isValid: boolean;
    smiles: string;
    canonicalSmiles: string;
    errors: ValidationError[];
    warnings: ValidationError[];
    polymerType: 'linear' | 'branched' | 'cyclic' | 'unknown';
    molecularWeight?: number;
    atomCount?: number;
    // Rule check status
    rulesChecked: {
        layer1: boolean;
        layer2: boolean;
        layer3: boolean;
    };
}

// RDKit.js type definitions (module loaded via CDN script tag)
interface JSMol {
    delete(): void;
    get_smiles(): string;
    get_cxsmiles(): string;
    get_molblock(): string;
    get_svg(width?: number, height?: number): string;
    get_descriptors(): string;
    get_substruct_match(q: JSMol): string;
    get_substruct_matches(q: JSMol): string;
}

interface RDKitModule {
    get_mol(input: string, details_json?: string): JSMol | null;
    get_qmol(input: string): JSMol | null;
    version(): string;
}

// Extend Window interface for global initRDKitModule
declare global {
    interface Window {
        initRDKitModule: () => Promise<RDKitModule>;
    }
}

// RDKit.js singleton instance
let rdkitInstance: RDKitModule | null = null;
let rdkitInitPromise: Promise<RDKitModule> | null = null;

/**
 * Initialize RDKit.js WebAssembly module (singleton pattern)
 * Call this early in the app lifecycle to preload the WASM
 * @returns Promise resolving to the RDKit module instance
 */
export const initRDKit = async (): Promise<RDKitModule> => {
    if (rdkitInstance) {
        return rdkitInstance;
    }

    if (rdkitInitPromise) {
        return rdkitInitPromise;
    }

    rdkitInitPromise = (async () => {
        // Use global initRDKitModule loaded via CDN script tag
        if (typeof window.initRDKitModule !== 'function') {
            throw new Error("RDKit.js not loaded - ensure the CDN script is included in index.html");
        }

        const rdkit = await window.initRDKitModule();
        rdkitInstance = rdkit;
        console.log("RDKit.js initialized successfully, version:", rdkitInstance.version());
        return rdkitInstance;
    })();

    return rdkitInitPromise;
};

/**
 * Get the RDKit instance, initializing if needed
 * @returns Promise resolving to RDKit module
 */
export const getRDKit = async (): Promise<RDKitModule> => {
    if (rdkitInstance) {
        return rdkitInstance;
    }
    return initRDKit();
};

/**
 * Configuration for polymer SMILES validation
 */
interface ValidationConfig {
    maxBondDistance: number;
    minBondDistance: number;
    maxConnectionsPerAtom: number;
}

const DEFAULT_CONFIG: ValidationConfig = {
    maxBondDistance: 5.0,
    minBondDistance: 0.5,
    maxConnectionsPerAtom: 4
};

/**
 * Result of polymer SMILES validation
 */
export interface ValidationResult {
    isValid: boolean;
    smiles: string;
    canonicalSmiles: string;
    errors: string[];
    warnings: string[];
    polymerType: 'linear' | 'branched' | 'cyclic' | 'unknown';
    molecularWeight?: number;
}

/**
 * Result of single SMILES validation
 */
export interface SmilesValidationResult {
    isValid: boolean;
    canonicalSmiles: string | null;
    error: string | null;
    molecularWeight?: number;
}

/**
 * Calculate Euclidean distance between two 3D positions
 */
const calculateDistance = (p1: Position, p2: Position): number => {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = p1.z - p2.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

/**
 * Build adjacency map from placed molecules
 */
const buildAdjacencyMap = (molecules: PlacedMolecule[]): Map<number, number[]> => {
    const adjacencyMap = new Map<number, number[]>();
    for (const mol of molecules) {
        adjacencyMap.set(mol.id, [...mol.connections]);
    }
    return adjacencyMap;
};

/**
 * Detect cycles in the molecular graph using DFS
 */
const detectCycles = (molecules: PlacedMolecule[]): boolean => {
    if (molecules.length === 0) return false;

    const visited = new Set<number>();
    const adjacencyMap = buildAdjacencyMap(molecules);

    const hasCycle = (nodeId: number, parentId: number | null): boolean => {
        visited.add(nodeId);
        const neighbors = adjacencyMap.get(nodeId) || [];

        for (const neighbor of neighbors) {
            if (!visited.has(neighbor)) {
                if (hasCycle(neighbor, nodeId)) return true;
            } else if (neighbor !== parentId) {
                return true;
            }
        }
        return false;
    };

    for (const mol of molecules) {
        if (!visited.has(mol.id)) {
            if (hasCycle(mol.id, null)) return true;
        }
    }

    return false;
};

/**
 * Classify polymer type based on graph structure
 */
const classifyPolymerType = (molecules: PlacedMolecule[]): ValidationResult['polymerType'] => {
    if (molecules.length === 0) return 'unknown';

    if (detectCycles(molecules)) return 'cyclic';

    const hasBranching = molecules.some(mol => mol.connections.length > 2);
    if (hasBranching) return 'branched';

    const endPoints = molecules.filter(mol => mol.connections.length <= 1);
    if (endPoints.length === 2 || (endPoints.length === 0 && molecules.length === 1)) {
        return 'linear';
    }

    return 'unknown';
};

/**
 * Validate a single SMILES string using RDKit.js
 * @param smiles - SMILES string to validate
 * @returns Validation result with canonical form and properties
 */
export const validateSmiles = async (smiles: string): Promise<SmilesValidationResult> => {
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);

        if (!mol) {
            return {
                isValid: false,
                canonicalSmiles: null,
                error: "Invalid SMILES: could not parse molecule"
            };
        }

        try {
            const canonical = mol.get_smiles();
            const descriptors = JSON.parse(mol.get_descriptors());
            const molWeight = descriptors.exactmw;

            return {
                isValid: true,
                canonicalSmiles: canonical,
                error: null,
                molecularWeight: parseFloat(molWeight) || undefined
            };
        } finally {
            mol.delete();
        }
    } catch (error) {
        return {
            isValid: false,
            canonicalSmiles: null,
            error: `Validation error: ${error}`
        };
    }
};

/**
 * Get canonical SMILES using RDKit.js
 * @param smiles - Input SMILES
 * @returns Canonical SMILES or original if invalid
 */
export const getCanonicalSmiles = async (smiles: string): Promise<string> => {
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);

        if (!mol) return smiles;

        try {
            return mol.get_smiles();
        } finally {
            mol.delete();
        }
    } catch {
        return smiles;
    }
};

/**
 * Validate basic structural properties of placed molecules
 */
export const validatePlacedMolecule = (molecules: PlacedMolecule[]): boolean => {
    if (!molecules || molecules.length === 0) return false;

    const idSet = new Set<number>();

    for (const mol of molecules) {
        if (typeof mol.id !== 'number' || isNaN(mol.id)) return false;
        if (idSet.has(mol.id)) return false;
        idSet.add(mol.id);

        if (!mol.position ||
            typeof mol.position.x !== 'number' || isNaN(mol.position.x) ||
            typeof mol.position.y !== 'number' || isNaN(mol.position.y) ||
            typeof mol.position.z !== 'number' || isNaN(mol.position.z)) {
            return false;
        }

        if (!mol.smiles || typeof mol.smiles !== 'string') return false;

        if (mol.connections) {
            for (const connId of mol.connections) {
                if (!idSet.has(connId) && !molecules.some(m => m.id === connId)) {
                    return false;
                }
            }
        }
    }

    return true;
};

/**
 * Validate spatial arrangement of molecules
 */
export const validateSpatialArrangement = (
    molecules: PlacedMolecule[],
    config: ValidationConfig = DEFAULT_CONFIG
): { isValid: boolean; errors: string[] } => {
    const errors: string[] = [];
    const molMap = new Map(molecules.map(m => [m.id, m]));

    for (const mol of molecules) {
        if (mol.connections.length > config.maxConnectionsPerAtom) {
            errors.push(
                `${mol.name} (ID: ${mol.id}) has ${mol.connections.length} connections, ` +
                `exceeds maximum of ${config.maxConnectionsPerAtom}`
            );
        }

        for (const connId of mol.connections) {
            const connMol = molMap.get(connId);
            if (!connMol) {
                errors.push(`${mol.name} references non-existent molecule ID: ${connId}`);
                continue;
            }

            const distance = calculateDistance(mol.position, connMol.position);

            if (distance > config.maxBondDistance) {
                errors.push(
                    `Bond between ${mol.name} and ${connMol.name} is too long ` +
                    `(${distance.toFixed(2)} > ${config.maxBondDistance})`
                );
            }

            if (distance < config.minBondDistance) {
                errors.push(
                    `${mol.name} and ${connMol.name} are too close ` +
                    `(${distance.toFixed(2)} < ${config.minBondDistance})`
                );
            }
        }
    }

    return { isValid: errors.length === 0, errors };
};

/**
 * Generate SMILES from placed molecules using graph traversal
 * Then validate and canonicalize with RDKit.js
 */
export const generateSmiles = async (molecules: PlacedMolecule[]): Promise<string> => {
    if (!validatePlacedMolecule(molecules)) {
        throw new Error("Invalid molecule data");
    }

    if (molecules.length === 0) return "";
    if (molecules.length === 1) return molecules[0].smiles;

    const molMap = new Map(molecules.map(m => [m.id, m]));
    const visited = new Set<number>();
    const smilesFragments: string[] = [];

    const traverseAndBuild = (currentId: number): string => {
        if (visited.has(currentId)) return "";

        visited.add(currentId);
        const current = molMap.get(currentId);
        if (!current) return "";

        let smiles = current.smiles;
        const unvisitedNeighbors = current.connections.filter(id => !visited.has(id));

        if (unvisitedNeighbors.length === 0) return smiles;

        const mainChainSmiles = traverseAndBuild(unvisitedNeighbors[0]);
        if (mainChainSmiles) smiles += mainChainSmiles;

        for (let i = 1; i < unvisitedNeighbors.length; i++) {
            const branchSmiles = traverseAndBuild(unvisitedNeighbors[i]);
            if (branchSmiles) smiles += `(${branchSmiles})`;
        }

        return smiles;
    };

    const endPoints = molecules.filter(m => m.connections.length === 1);
    const startMol = endPoints.length > 0 ? endPoints[0] : molecules[0];

    const mainSmiles = traverseAndBuild(startMol.id);
    smilesFragments.push(mainSmiles);

    for (const mol of molecules) {
        if (!visited.has(mol.id)) {
            const componentSmiles = traverseAndBuild(mol.id);
            if (componentSmiles) smilesFragments.push(componentSmiles);
        }
    }

    return smilesFragments.join('.');
};

/**
 * Comprehensive polymer configuration validation using RDKit.js
 * Validates structure, spatial arrangement, and chemical validity
 */
export const validatePolymerConfiguration = async (
    molecules: PlacedMolecule[],
    config: ValidationConfig = DEFAULT_CONFIG
): Promise<ValidationResult> => {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!molecules || molecules.length === 0) {
        return {
            isValid: false,
            smiles: "",
            canonicalSmiles: "",
            errors: ["No molecules placed"],
            warnings: [],
            polymerType: 'unknown'
        };
    }

    // Structural validation
    if (!validatePlacedMolecule(molecules)) {
        errors.push("Invalid molecule data structure");
    }

    // Spatial validation
    const spatialResult = validateSpatialArrangement(molecules, config);
    errors.push(...spatialResult.errors);

    // Disconnected molecules warning
    const disconnected = molecules.filter(m => m.connections.length === 0);
    if (disconnected.length > 0 && molecules.length > 1) {
        warnings.push(
            `${disconnected.length} molecules not connected: ${disconnected.map(m => m.name).join(', ')}`
        );
    }

    // Validate individual SMILES with RDKit.js
    for (const mol of molecules) {
        const validation = await validateSmiles(mol.smiles);
        if (!validation.isValid) {
            errors.push(`Invalid SMILES for ${mol.name}: ${validation.error}`);
        }
    }

    const polymerType = classifyPolymerType(molecules);

    // Generate and validate combined SMILES
    let smiles = "";
    let canonicalSmiles = "";
    let molecularWeight: number | undefined;

    try {
        smiles = await generateSmiles(molecules);

        // Validate combined SMILES with RDKit.js
        const combinedValidation = await validateSmiles(smiles);
        if (combinedValidation.isValid) {
            canonicalSmiles = combinedValidation.canonicalSmiles || smiles;
            molecularWeight = combinedValidation.molecularWeight;
        } else {
            warnings.push("Combined SMILES may not represent a single molecule");
            canonicalSmiles = smiles;
        }

        if (polymerType === 'linear' && molecules.length > 2) {
            warnings.push("Consider using pSMILES notation [*]...[*] for repeating units");
        }
    } catch (error) {
        errors.push(`Failed to generate SMILES: ${error}`);
    }

    return {
        isValid: errors.length === 0,
        smiles,
        canonicalSmiles,
        errors,
        warnings,
        polymerType,
        molecularWeight
    };
};

/**
 * Convert SMILES to polymer SMILES (pSMILES) with star atoms
 */
export const convertToPSMILES = (smiles: string, repeatUnit: boolean = true): string => {
    if (!smiles) return "";
    return repeatUnit ? `[*]${smiles}[*]` : smiles;
};

/**
 * Generate SVG molecule drawing using RDKit.js
 * @param smiles - SMILES string
 * @param width - Image width (default 300)
 * @param height - Image height (default 200)
 * @returns SVG string or null if invalid
 */
export const getMoleculeSVG = async (
    smiles: string,
    width: number = 300,
    height: number = 200
): Promise<string | null> => {
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);

        if (!mol) return null;

        try {
            return mol.get_svg(width, height);
        } finally {
            mol.delete();
        }
    } catch {
        return null;
    }
};

/**
 * Calculate molecular descriptors using RDKit.js
 * @param smiles - SMILES string
 * @returns Object with calculated descriptors
 */
export const getMolecularDescriptors = async (smiles: string): Promise<Record<string, number> | null> => {
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);

        if (!mol) return null;

        try {
            const descriptorsJson = mol.get_descriptors();
            return JSON.parse(descriptorsJson);
        } finally {
            mol.delete();
        }
    } catch {
        return null;
    }
};

/**
 * Check if a substructure pattern exists in a molecule
 * @param smiles - Molecule SMILES
 * @param pattern - SMARTS pattern to search for
 * @returns true if pattern found
 */
export const hasSubstructure = async (smiles: string, pattern: string): Promise<boolean> => {
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);
        const query = rdkit.get_qmol(pattern);

        if (!mol || !query) return false;

        try {
            const match = mol.get_substruct_match(query);
            return match !== "{}";
        } finally {
            mol.delete();
            query.delete();
        }
    } catch {
        return false;
    }
};

/**
 * Detect functional groups in a molecule using SMARTS patterns
 */
export const detectFunctionalGroups = async (smiles: string): Promise<Array<{ name: string; count: number }>> => {
    const patterns: Record<string, string> = {
        ester: "[#6][CX3](=O)[OX2H0][#6]",
        carboxylic_acid: "[CX3](=O)[OX2H1]",
        amide: "[NX3][CX3](=[OX1])[#6]",
        amine_primary: "[NX3;H2;!$(NC=O)]",
        amine_secondary: "[NX3;H1;!$(NC=O)]",
        alcohol: "[OX2H][CX4]",
        ether: "[OD2]([#6])[#6]",
        aldehyde: "[CX3H1](=O)[#6]",
        ketone: "[#6][CX3](=O)[#6]",
        alkene: "[CX3]=[CX3]",
        aromatic: "c1ccccc1",
        halide: "[#6][F,Cl,Br,I]"
    };

    const results: Array<{ name: string; count: number }> = [];

    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);

        if (!mol) return results;

        try {
            for (const [name, pattern] of Object.entries(patterns)) {
                const query = rdkit.get_qmol(pattern);
                if (query) {
                    const matches = mol.get_substruct_matches(query);
                    // Parse matches string "[[0,1],[2,3]]" to count
                    const count = (matches.match(/\[/g) || []).length / 2; // Rough count approximation
                    if (count > 0) {
                        results.push({ name, count });
                    }
                    query.delete();
                }
            }
        } finally {
            mol.delete();
        }
    } catch (e) {
        console.error("Functional group detection failed:", e);
    }

    return results;
};








// =============================================================================
// POLYMER PROPERTY ANALYSIS (For Optimization Rules)
// =============================================================================

/**
 * SMARTS patterns for polymer property analysis
 * Used for mechanical, sustainability, and polymerizability assessments
 */
const POLYMER_ANALYSIS_PATTERNS = {
    // Rotatable bonds - single bonds that can freely rotate (flexibility indicator)
    rotatable_bond: "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]",

    // Aromatic systems (rigidity indicator)
    aromatic_ring: "c1ccccc1",
    aromatic_atom: "[c,n,o,s]",

    // Polar groups (intermolecular force indicator)
    h_bond_donor: "[#7,#8,#9;H]",  // N-H, O-H, F-H
    h_bond_acceptor: "[#7,#8,#9;!H0]",  // N, O, F with lone pairs
    hydroxyl: "[OX2H]",
    amine: "[NX3;H2,H1]",

    // Degradable linkages (sustainability indicator)
    ester_linkage: "[#6][CX3](=O)[OX2][#6]",
    amide_linkage: "[NX3][CX3](=[OX1])[#6]",
    carbonate_linkage: "[OX2][CX3](=[OX1])[OX2]",
    urethane_linkage: "[NX3][CX3](=[OX1])[OX2]",

    // Halogens (environmental concern)
    fluorine: "[F]",
    chlorine: "[Cl]",
    bromine: "[Br]",
    iodine: "[I]",
    any_halogen: "[F,Cl,Br,I]",

    // Unsaturation (rigidity/reactivity)
    double_bond: "[CX3]=[CX3]",
    triple_bond: "[CX2]#[CX2]",

    // Connection points for pSMILES
    star_atom: "[#0]"  // [*] wildcard
};

/**
 * Polymer analysis result containing all structural metrics
 */
export interface PolymerAnalysis {
    // Structural counts
    rotatableBonds: number;
    aromaticAtoms: number;
    aromaticRings: number;
    doubleBonds: number;
    tripleBonds: number;

    // Polar groups
    hBondDonors: number;
    hBondAcceptors: number;
    hydroxylGroups: number;
    amineGroups: number;

    // Degradable linkages
    esterLinkages: number;
    amideLinkages: number;
    carbonateLinkages: number;
    urethaneLinkages: number;
    totalDegradableLinkages: number;

    // Halogens
    fluorineCount: number;
    chlorineCount: number;
    bromineCount: number;
    iodineCount: number;
    totalHalogens: number;

    // Reactive sites
    reactiveSites: number;
    connectionPoints: number;

    // Derived metrics (computed from counts)
    flexibilityScore: number;      // 0-100: higher = more flexible
    rigidityScore: number;         // 0-100: higher = more rigid
    polarityScore: number;         // 0-100: higher = more polar
    sustainabilityScore: number;   // 0-100: higher = more sustainable

    // Molecular info
    atomCount: number;
    estimatedMW: number;
}

/**
 * Count pattern matches in a SMILES string using RDKit
 */
const countPatternMatches = async (smiles: string, pattern: string): Promise<number> => {
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);

        if (!mol) return 0;

        try {
            const query = rdkit.get_qmol(pattern);
            if (!query) return 0;

            try {
                const matches = mol.get_substruct_matches(query);
                const matchArray = JSON.parse(matches);
                return matchArray.length;
            } finally {
                query.delete();
            }
        } finally {
            mol.delete();
        }
    } catch {
        return 0;
    }
};

/**
 * Comprehensive polymer structure analysis
 * Analyzes SMILES for mechanical, sustainability, and polymerizability properties
 * 
 * @param smiles - SMILES string to analyze
 * @returns PolymerAnalysis with all structural metrics
 */
export const analyzePolymerStructure = async (smiles: string): Promise<PolymerAnalysis> => {
    // Count all patterns in parallel for efficiency
    const [
        rotatableBonds,
        aromaticAtoms,
        aromaticRings,
        doubleBonds,
        tripleBonds,
        hBondDonors,
        hBondAcceptors,
        hydroxylGroups,
        amineGroups,
        esterLinkages,
        amideLinkages,
        carbonateLinkages,
        urethaneLinkages,
        fluorineCount,
        chlorineCount,
        bromineCount,
        iodineCount,
        connectionPoints,
        reactiveSites
    ] = await Promise.all([
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.rotatable_bond),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.aromatic_atom),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.aromatic_ring),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.double_bond),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.triple_bond),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.h_bond_donor),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.h_bond_acceptor),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.hydroxyl),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.amine),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.ester_linkage),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.amide_linkage),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.carbonate_linkage),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.urethane_linkage),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.fluorine),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.chlorine),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.bromine),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.iodine),
        countPatternMatches(smiles, POLYMER_ANALYSIS_PATTERNS.star_atom),
        countReactiveSites(smiles)
    ]);

    const totalDegradableLinkages = esterLinkages + amideLinkages + carbonateLinkages + urethaneLinkages;
    const totalHalogens = fluorineCount + chlorineCount + bromineCount + iodineCount;
    const atomCount = countAtomsInSmiles(smiles);

    // Calculate derived scores (0-100 scale)

    // Flexibility: high rotatable bonds + low aromatics = flexible
    // Normalize by atom count to make comparable across molecules
    const normalizedRotatable = atomCount > 0 ? (rotatableBonds / atomCount) * 100 : 0;
    const normalizedAromatic = atomCount > 0 ? (aromaticAtoms / atomCount) * 100 : 0;
    const flexibilityScore = Math.min(100, Math.max(0,
        normalizedRotatable * 2 - normalizedAromatic
    ));

    // Rigidity: high aromatics + double bonds = rigid
    const rigidityScore = Math.min(100, Math.max(0,
        normalizedAromatic + (doubleBonds / Math.max(1, atomCount)) * 50
    ));

    // Polarity: polar groups relative to size
    const polarGroups = hBondDonors + hBondAcceptors;
    const polarityScore = Math.min(100, Math.max(0,
        atomCount > 0 ? (polarGroups / atomCount) * 200 : 0
    ));

    // Sustainability: degradable linkages, no halogens
    // Higher score = more sustainable
    const sustainabilityScore = Math.min(100, Math.max(0,
        (totalDegradableLinkages > 0 ? 50 : 0) +
        (totalHalogens === 0 ? 30 : -totalHalogens * 10) +
        (atomCount < 50 ? 20 : 0)
    ));

    // Estimate MW (rough approximation)
    const estimatedMW = atomCount * 12; // Rough average atomic mass

    return {
        rotatableBonds,
        aromaticAtoms,
        aromaticRings,
        doubleBonds,
        tripleBonds,
        hBondDonors,
        hBondAcceptors,
        hydroxylGroups,
        amineGroups,
        esterLinkages,
        amideLinkages,
        carbonateLinkages,
        urethaneLinkages,
        totalDegradableLinkages,
        fluorineCount,
        chlorineCount,
        bromineCount,
        iodineCount,
        totalHalogens,
        reactiveSites,
        connectionPoints,
        flexibilityScore,
        rigidityScore,
        polarityScore,
        sustainabilityScore,
        atomCount,
        estimatedMW
    };
};

// =============================================================================
// SMILES AUTO-REPAIR PIPELINE
// =============================================================================

/**
 * Result of SMILES repair attempt
 */
export interface SmilesRepairResult {
    success: boolean;
    original: string;
    repaired: string;
    canonical: string;
    wasModified: boolean;
    repairSteps: string[];
    error?: string;
}

/**
 * Extract the largest fragment from a multi-fragment SMILES
 * Fragments are separated by '.' in SMILES notation
 * 
 * @param smiles - SMILES string potentially containing fragments
 * @returns Largest fragment by atom count
 */
const extractLargestFragment = (smiles: string): string => {
    if (!smiles.includes('.')) return smiles;

    const fragments = smiles.split('.');
    let largest = fragments[0];
    let maxAtoms = countAtomsInSmiles(largest);

    for (let i = 1; i < fragments.length; i++) {
        const atomCount = countAtomsInSmiles(fragments[i]);
        if (atomCount > maxAtoms) {
            maxAtoms = atomCount;
            largest = fragments[i];
        }
    }

    return largest;
};

/**
 * Attempt to repair ring notation issues
 * Finds unpaired ring digits and removes them
 * 
 * @param smiles - SMILES string with potential ring issues  
 * @returns Repaired SMILES or original if repair not possible
 */
const repairRingNotation = (smiles: string): string => {
    // Find all ring digits used
    const ringDigits: Map<string, number> = new Map();
    let cleanSmiles = smiles;

    // Count occurrences of each ring digit (1-9, %10-%99)
    // eslint-disable-next-line no-useless-escape
    const digitPattern = /(%\d{2}|\d)(?![^\[]*\])/g;
    const matches = smiles.match(digitPattern) || [];

    for (const digit of matches) {
        ringDigits.set(digit, (ringDigits.get(digit) || 0) + 1);
    }

    // Ring digits should appear in pairs - remove unpaired ones
    ringDigits.forEach((count, digit) => {
        if (count % 2 !== 0) {
            // Unpaired ring digit - remove last occurrence
            const lastIndex = cleanSmiles.lastIndexOf(digit);
            if (lastIndex !== -1) {
                cleanSmiles = cleanSmiles.slice(0, lastIndex) + cleanSmiles.slice(lastIndex + digit.length);
            }
        }
    });

    return cleanSmiles;;
};

/**
 * Balance parentheses in SMILES string
 * Adds missing closing parentheses or removes excess opening ones
 * 
 * @param smiles - SMILES string with potential parenthesis issues
 * @returns Repaired SMILES
 */
const balanceParentheses = (smiles: string): string => {
    let openCount = 0;
    let result = '';

    for (const char of smiles) {
        if (char === '(') {
            openCount++;
            result += char;
        } else if (char === ')') {
            if (openCount > 0) {
                openCount--;
                result += char;
            }
            // Skip excess closing parentheses
        } else {
            result += char;
        }
    }

    // Add missing closing parentheses
    while (openCount > 0) {
        result += ')';
        openCount--;
    }

    return result;
};

/**
 * Convert SMILES to pSMILES format by adding connection points
 * Only if the structure doesn't already have connection points
 * 
 * @param smiles - SMILES string
 * @returns pSMILES with [*] connection points
 */
const convertTopSMILES = (smiles: string): string => {
    // Check if already has connection points
    if (smiles.includes('[*]') || smiles.includes('[#0]')) {
        return smiles;
    }

    // Add connection points at both ends
    return `[*]${smiles}[*]`;
};

/**
 * Comprehensive SMILES repair pipeline
 * Attempts multiple repair strategies in order of likelihood of success
 * 
 * Repair order:
 * 1. Try direct canonicalization (fixes most issues)
 * 2. Extract largest fragment (removes salts/counterions)
 * 3. Balance parentheses
 * 4. Repair ring notation
 * 5. Try with lenient parsing
 * 6. Convert to pSMILES format
 * 
 * @param smiles - Original SMILES string (potentially invalid)
 * @returns SmilesRepairResult with original, repaired, and canonical forms
 */
export const repairSmiles = async (smiles: string): Promise<SmilesRepairResult> => {
    const repairSteps: string[] = [];
    let currentSmiles = smiles.trim();

    // Handle empty input
    if (!currentSmiles) {
        return {
            success: false,
            original: smiles,
            repaired: '',
            canonical: '',
            wasModified: false,
            repairSteps: ['Input was empty'],
            error: 'Empty SMILES string'
        };
    }

    // Step 1: Try direct validation and canonicalization
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(currentSmiles);

        if (mol) {
            try {
                const canonical = mol.get_smiles();
                return {
                    success: true,
                    original: smiles,
                    repaired: smiles,
                    canonical,
                    wasModified: false,
                    repairSteps: ['Valid SMILES - canonicalized']
                };
            } finally {
                mol.delete();
            }
        }
    } catch {
        // Continue to repair steps
    }

    repairSteps.push('Original SMILES invalid - attempting repair');

    // Step 2: Extract largest fragment (removes salts, counterions)
    if (currentSmiles.includes('.')) {
        const largestFragment = extractLargestFragment(currentSmiles);
        if (largestFragment !== currentSmiles) {
            currentSmiles = largestFragment;
            repairSteps.push('Removed disconnected fragments (salts/counterions)');

            // Try validation after fragment extraction
            const validation = await validateSmiles(currentSmiles);
            if (validation.isValid && validation.canonicalSmiles) {
                return {
                    success: true,
                    original: smiles,
                    repaired: currentSmiles,
                    canonical: validation.canonicalSmiles,
                    wasModified: true,
                    repairSteps
                };
            }
        }
    }

    // Step 3: Balance parentheses
    const balanced = balanceParentheses(currentSmiles);
    if (balanced !== currentSmiles) {
        currentSmiles = balanced;
        repairSteps.push('Balanced parentheses');

        const validation = await validateSmiles(currentSmiles);
        if (validation.isValid && validation.canonicalSmiles) {
            return {
                success: true,
                original: smiles,
                repaired: currentSmiles,
                canonical: validation.canonicalSmiles,
                wasModified: true,
                repairSteps
            };
        }
    }

    // Step 4: Repair ring notation
    const ringRepaired = repairRingNotation(currentSmiles);
    if (ringRepaired !== currentSmiles) {
        currentSmiles = ringRepaired;
        repairSteps.push('Repaired ring notation');

        const validation = await validateSmiles(currentSmiles);
        if (validation.isValid && validation.canonicalSmiles) {
            return {
                success: true,
                original: smiles,
                repaired: currentSmiles,
                canonical: validation.canonicalSmiles,
                wasModified: true,
                repairSteps
            };
        }
    }

    // Step 5: Try lenient parsing (RDKit with relaxed rules)
    try {
        const rdkit = await getRDKit();
        // Try parsing with sanitize=false equivalent - use details_json
        const mol = rdkit.get_mol(currentSmiles, JSON.stringify({ sanitize: false }));

        if (mol) {
            try {
                // Try to get a valid SMILES even from partially parsed molecule
                const canonical = mol.get_smiles();
                if (canonical) {
                    repairSteps.push('Parsed with lenient sanitization');
                    return {
                        success: true,
                        original: smiles,
                        repaired: canonical,
                        canonical,
                        wasModified: true,
                        repairSteps
                    };
                }
            } finally {
                mol.delete();
            }
        }
    } catch {
        // Continue to next step
    }

    // Step 6: Strip to just atoms (last resort)
    // Remove all non-essential characters and try to build basic structure
    const atomsOnly = currentSmiles.replace(/[^A-Za-z]/g, '');
    if (atomsOnly.length > 0) {
        const validation = await validateSmiles(atomsOnly);
        if (validation.isValid && validation.canonicalSmiles) {
            repairSteps.push('Stripped to atoms only (lost bond information)');
            return {
                success: true,
                original: smiles,
                repaired: atomsOnly,
                canonical: validation.canonicalSmiles,
                wasModified: true,
                repairSteps
            };
        }
    }

    // All repair attempts failed
    repairSteps.push('All repair attempts failed');
    return {
        success: false,
        original: smiles,
        repaired: currentSmiles,
        canonical: '',
        wasModified: currentSmiles !== smiles,
        repairSteps,
        error: 'Could not repair SMILES to valid form'
    };
};

/**
 * Smart SMILES conversion that repairs and optionally converts to pSMILES
 * Best used before validation/prediction to maximize success rate
 * 
 * @param smiles - Input SMILES (potentially invalid)
 * @param options - Conversion options
 * @returns Repaired and optionally converted SMILES
 */
export const smartSmilesConvert = async (
    smiles: string,
    options: {
        convertTopSMILES?: boolean;
        removeFragments?: boolean;
    } = {}
): Promise<SmilesRepairResult> => {
    let result = await repairSmiles(smiles);

    if (!result.success) {
        return result;
    }

    let finalSmiles = result.canonical || result.repaired;
    const additionalSteps: string[] = [];

    // Remove fragments if requested
    if (options.removeFragments && finalSmiles.includes('.')) {
        finalSmiles = extractLargestFragment(finalSmiles);
        additionalSteps.push('Removed additional fragments');
    }

    // Convert to pSMILES if requested
    if (options.convertTopSMILES && !finalSmiles.includes('[*]')) {
        finalSmiles = convertTopSMILES(finalSmiles);
        additionalSteps.push('Converted to pSMILES format');
    }

    return {
        ...result,
        repaired: finalSmiles,
        canonical: finalSmiles,
        wasModified: result.wasModified || additionalSteps.length > 0,
        repairSteps: [...result.repairSteps, ...additionalSteps]
    };
};

// =============================================================================
// COMPREHENSIVE POLYMER VALIDATION (Layer 1-3)
// =============================================================================

// Allowed elements for polymer ML predictions
const ALLOWED_ELEMENTS = new Set(['C', 'H', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'I', 'Si', 'c', 'n', 'o', 's']);

// Maximum atom count for repeat unit
const MAX_ATOM_COUNT = 100;

// SMARTS patterns for reactive sites (polymerizable groups)
const REACTIVE_SITE_PATTERNS: Record<string, string> = {
    alcohol: "[OX2H]",
    carboxylic_acid: "[CX3](=O)[OX2H1]",
    amine: "[NX3;H2,H1;!$(NC=O)]",
    isocyanate: "[NX2]=[CX2]=[OX1]",
    epoxide: "C1OC1",
    vinyl: "[CX3]=[CX3]",
    connection_point: "[#0]", // wildcard atom [*]
    halide_reactive: "[CX4][Cl,Br,I]"
};

/**
 * Check if SMILES contains only allowed elements (Rule 10)
 */
const checkAllowedElements = (smiles: string): { valid: boolean; forbidden: string[] } => {
    const forbidden: string[] = [];

    // Extract element symbols from SMILES (simplified parsing)
    const elementPattern = /\[([A-Z][a-z]?)/g;
    let match;
    while ((match = elementPattern.exec(smiles)) !== null) {
        const element = match[1];
        if (!ALLOWED_ELEMENTS.has(element)) {
            forbidden.push(element);
        }
    }

    // Check uppercase letters outside brackets (organic subset)
    const organicPattern = /(?<!\[)([A-Z][a-z]?)(?![a-z\]])/g;
    while ((match = organicPattern.exec(smiles)) !== null) {
        const element = match[1];
        // Filter out ring numbers and common organic symbols
        if (element.length === 1 && !'BCNOPSFIK'.includes(element)) {
            continue;
        }
        if (element.length === 2 && !['Cl', 'Br', 'Si'].includes(element)) {
            if (!ALLOWED_ELEMENTS.has(element)) {
                forbidden.push(element);
            }
        }
    }

    return { valid: forbidden.length === 0, forbidden: Array.from(new Set(forbidden)) };
};

/**
 * Check for radicals and formal charges (Rule 11)
 */
const checkRadicalsAndCharges = (smiles: string): { hasRadicals: boolean; hasCharges: boolean } => {
    // Check for explicit charges in brackets [X+], [X-], [X+2], etc.
    const chargePattern = /\[[^\]]*[+-]\d*\]/;
    const hasCharges = chargePattern.test(smiles);

    // Check for radical notation (rare in SMILES but possible)
    const radicalPattern = /\[[^\]]*\.\]/;
    const hasRadicals = radicalPattern.test(smiles);

    return { hasRadicals, hasCharges };
};

/**
 * Count atoms in a SMILES string (approximate)
 */
const countAtomsInSmiles = (smiles: string): number => {
    // Count uppercase letters (atom symbols) and bracket expressions
    let count = 0;
    const bracketPattern = /\[[^\]]+\]/g;
    const bracketAtoms = smiles.match(bracketPattern) || [];
    count += bracketAtoms.length;

    // Remove bracket expressions and count remaining uppercase letters
    const withoutBrackets = smiles.replace(bracketPattern, '');
    const organicAtoms = withoutBrackets.match(/[A-Z]/g) || [];
    count += organicAtoms.length;

    // Add implicit hydrogens estimate (simplified)
    return count;
};

/**
 * Check for disconnected fragments (Rule 2)
 */
const checkDisconnectedFragments = (smiles: string): boolean => {
    // Fragments are separated by '.' in SMILES
    return smiles.includes('.');
};

/**
 * Count reactive sites using SMARTS patterns
 */
const countReactiveSites = async (smiles: string): Promise<number> => {
    let totalSites = 0;

    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);

        if (!mol) return 0;

        try {
            for (const smartsPattern of Object.values(REACTIVE_SITE_PATTERNS)) {
                const query = rdkit.get_qmol(smartsPattern);
                if (query) {
                    const matches = mol.get_substruct_matches(query);
                    const matchArray = JSON.parse(matches);
                    totalSites += matchArray.length;
                    query.delete();
                }
            }
        } finally {
            mol.delete();
        }
    } catch {
        return 0;
    }

    return totalSites;
};

/**
 * Check if molecule is fully capped (no reactive sites)
 */
const isFullyCapped = async (smiles: string): Promise<boolean> => {
    const reactiveSites = await countReactiveSites(smiles);
    return reactiveSites === 0;
};

/**
 * Comprehensive polymer validation implementing all layers
 * This is the main validation function to use for validate/predict operations
 */
export const validatePolymerComprehensive = async (
    molecules: PlacedMolecule[]
): Promise<PolymerValidationResult> => {
    const errors: ValidationError[] = [];
    const warnings: ValidationError[] = [];

    // Track which layers were checked
    const rulesChecked = {
        layer1: false,
        layer2: false,
        layer3: false
    };

    // === CANVAS VALIDATION ===
    if (!molecules || molecules.length === 0) {
        errors.push({
            code: ValidationRuleCode.CANVAS_NO_MOLECULES,
            message: ValidationRuleMessages[ValidationRuleCode.CANVAS_NO_MOLECULES].title
        });
        return {
            isValid: false,
            smiles: "",
            canonicalSmiles: "",
            errors,
            warnings,
            polymerType: 'unknown',
            rulesChecked
        };
    }

    // Check canvas structure
    if (!validatePlacedMolecule(molecules)) {
        errors.push({
            code: ValidationRuleCode.CANVAS_INVALID_STRUCTURE,
            message: ValidationRuleMessages[ValidationRuleCode.CANVAS_INVALID_STRUCTURE].title
        });
    }

    // Check spatial arrangement
    const spatialResult = validateSpatialArrangement(molecules);
    if (!spatialResult.isValid) {
        for (const spatialError of spatialResult.errors) {
            errors.push({
                code: ValidationRuleCode.CANVAS_SPATIAL_ERROR,
                message: ValidationRuleMessages[ValidationRuleCode.CANVAS_SPATIAL_ERROR].title,
                details: spatialError
            });
        }
    }

    // Check for RDKit availability
    let rdkitAvailable = false;
    try {
        await getRDKit();
        rdkitAvailable = true;
    } catch {
        errors.push({
            code: ValidationRuleCode.RDKIT_NOT_LOADED,
            message: ValidationRuleMessages[ValidationRuleCode.RDKIT_NOT_LOADED].title
        });
    }

    // === LAYER 1: Basic Chemistry Validity ===
    rulesChecked.layer1 = true;

    // Validate each molecule's SMILES
    for (const mol of molecules) {
        if (rdkitAvailable) {
            const validation = await validateSmiles(mol.smiles);
            if (!validation.isValid) {
                // Rule 4 - Invalid syntax (RDKit couldn't parse)
                errors.push({
                    code: ValidationRuleCode.RULE_4_INVALID_SYNTAX,
                    message: `${mol.name}: ${ValidationRuleMessages[ValidationRuleCode.RULE_4_INVALID_SYNTAX].title}`,
                    details: validation.error || undefined,
                    moleculeName: mol.name,
                    smiles: mol.smiles
                });
            }
        }
    }

    // Generate combined SMILES
    let smiles = "";
    let canonicalSmiles = "";
    let molecularWeight: number | undefined;
    let atomCount: number | undefined;

    try {
        smiles = await generateSmiles(molecules);
    } catch (error) {
        errors.push({
            code: ValidationRuleCode.GENERATION_FAILED,
            message: ValidationRuleMessages[ValidationRuleCode.GENERATION_FAILED].title,
            details: String(error)
        });
        return {
            isValid: false,
            smiles: "",
            canonicalSmiles: "",
            errors,
            warnings,
            polymerType: classifyPolymerType(molecules),
            rulesChecked
        };
    }

    // Rule 2 - Check for disconnected fragments
    if (checkDisconnectedFragments(smiles)) {
        // Check if disconnection is intentional (multiple separate molecules)
        const disconnectedCount = smiles.split('.').length;
        if (disconnectedCount > 1 && molecules.length > 1) {
            const disconnectedMols = molecules.filter(m => m.connections.length === 0);
            if (disconnectedMols.length > 0) {
                errors.push({
                    code: ValidationRuleCode.RULE_2_DISCONNECTED,
                    message: ValidationRuleMessages[ValidationRuleCode.RULE_2_DISCONNECTED].title,
                    details: `Found ${disconnectedCount} disconnected fragments. Unconnected molecules: ${disconnectedMols.map(m => m.name).join(', ')}`
                });
            }
        }
    }

    // Validate combined SMILES with RDKit
    if (rdkitAvailable && smiles) {
        const combinedValidation = await validateSmiles(smiles);
        if (combinedValidation.isValid) {
            canonicalSmiles = combinedValidation.canonicalSmiles || smiles;
            molecularWeight = combinedValidation.molecularWeight;
        } else {
            // Could be Rule 1 (valence), Rule 3 (ring), or Rule 4 (syntax)
            const errorMsg = combinedValidation.error?.toLowerCase() || '';
            if (errorMsg.includes('valence') || errorMsg.includes('bond')) {
                errors.push({
                    code: ValidationRuleCode.RULE_1_VALENCE,
                    message: ValidationRuleMessages[ValidationRuleCode.RULE_1_VALENCE].title,
                    details: combinedValidation.error || undefined,
                    smiles
                });
            } else if (errorMsg.includes('ring')) {
                errors.push({
                    code: ValidationRuleCode.RULE_3_RING_CLOSURE,
                    message: ValidationRuleMessages[ValidationRuleCode.RULE_3_RING_CLOSURE].title,
                    details: combinedValidation.error || undefined,
                    smiles
                });
            } else {
                errors.push({
                    code: ValidationRuleCode.RULE_4_INVALID_SYNTAX,
                    message: ValidationRuleMessages[ValidationRuleCode.RULE_4_INVALID_SYNTAX].title,
                    details: combinedValidation.error || undefined,
                    smiles
                });
            }
            canonicalSmiles = smiles;
        }
    }

    // === LAYER 2: Polymer-Specific Rules ===
    rulesChecked.layer2 = true;

    // Rule 5 - Check for minimum reactive sites
    if (rdkitAvailable && smiles) {
        const reactiveSites = await countReactiveSites(smiles);
        if (reactiveSites < 2 && molecules.length > 1) {
            // Only warn if trying to build a polymer (multiple molecules)
            warnings.push({
                code: ValidationRuleCode.RULE_5_MIN_REACTIVE_SITES,
                message: ValidationRuleMessages[ValidationRuleCode.RULE_5_MIN_REACTIVE_SITES].title,
                details: `Found ${reactiveSites} reactive site(s). Need at least 2 for polymerization.`
            });
        }
    }

    // Rule 6 - Check if fully capped
    if (rdkitAvailable && smiles && molecules.length === 1) {
        const capped = await isFullyCapped(smiles);
        if (capped) {
            warnings.push({
                code: ValidationRuleCode.RULE_6_FULLY_CAPPED,
                message: ValidationRuleMessages[ValidationRuleCode.RULE_6_FULLY_CAPPED].title,
                details: "This monomer has no reactive sites for chain extension."
            });
        }
    }

    // Rule 8 - Check repeat unit size
    atomCount = countAtomsInSmiles(smiles);
    if (atomCount > MAX_ATOM_COUNT) {
        errors.push({
            code: ValidationRuleCode.RULE_8_REPEAT_SIZE,
            message: ValidationRuleMessages[ValidationRuleCode.RULE_8_REPEAT_SIZE].title,
            details: `Structure has approximately ${atomCount} atoms (max: ${MAX_ATOM_COUNT}).`
        });
    }

    // === LAYER 3: ML-Friendly Rules ===
    rulesChecked.layer3 = true;

    // Rule 9 - Canonical form check
    if (canonicalSmiles && smiles !== canonicalSmiles) {
        warnings.push({
            code: ValidationRuleCode.RULE_9_NOT_CANONICAL,
            message: ValidationRuleMessages[ValidationRuleCode.RULE_9_NOT_CANONICAL].title,
            details: `Original: ${smiles}\nCanonical: ${canonicalSmiles}`
        });
    }

    // Rule 10 - Allowed elements check
    const elementsCheck = checkAllowedElements(smiles);
    if (!elementsCheck.valid) {
        errors.push({
            code: ValidationRuleCode.RULE_10_FORBIDDEN_ELEMENTS,
            message: ValidationRuleMessages[ValidationRuleCode.RULE_10_FORBIDDEN_ELEMENTS].title,
            details: `Forbidden elements found: ${elementsCheck.forbidden.join(', ')}`
        });
    }

    // Rule 11 - Radicals and charges check
    const radicalChargeCheck = checkRadicalsAndCharges(smiles);
    if (radicalChargeCheck.hasCharges) {
        warnings.push({
            code: ValidationRuleCode.RULE_11_RADICALS_CHARGES,
            message: "Formal Charges Detected",
            details: "Structure contains formal charges. Consider using neutral forms for stable polymers."
        });
    }
    if (radicalChargeCheck.hasRadicals) {
        errors.push({
            code: ValidationRuleCode.RULE_11_RADICALS_CHARGES,
            message: ValidationRuleMessages[ValidationRuleCode.RULE_11_RADICALS_CHARGES].title,
            details: "Radical species detected. These are typically unstable."
        });
    }

    const polymerType = classifyPolymerType(molecules);

    return {
        isValid: errors.length === 0,
        smiles,
        canonicalSmiles: canonicalSmiles || smiles,
        errors,
        warnings,
        polymerType,
        molecularWeight,
        atomCount,
        rulesChecked
    };
};

/**
 * API endpoint for property prediction combining Tier 1 (heuristics) and Tier 2 (physics-based)
 * Calls both /predict/tier-1 and /predict/tier-2 endpoints and averages results 50/50
 * Both tiers return values in 0-10 scale
 */
export const predictPropertiesFromBackend = async (
    smiles: string
): Promise<{ success: boolean; properties?: Record<string, number>; error?: string; tier1Only?: boolean }> => {
    try {
        // Call both Tier 1 and Tier 2 in parallel
        const [tier1Response, tier2Response] = await Promise.allSettled([
            predictTier1(smiles),
            predictTier2(smiles)
        ]);

        console.log('Tier 1 response:', tier1Response);
        console.log('Tier 2 response:', tier2Response);

        // Extract Tier 1 data
        let tier1Data: { strength: number; flexibility: number; degradability: number; sustainability: number; sas_score?: number } | null = null;
        if (tier1Response.status === 'fulfilled' && tier1Response.value.status === 200 && tier1Response.value.data) {
            tier1Data = tier1Response.value.data;
        }

        // Extract Tier 2 data
        let tier2Data: { strength: number; flexibility: number; degradability: number; sustainability: number } | null = null;
        if (tier2Response.status === 'fulfilled' && tier2Response.value.status === 200 && tier2Response.value.data) {
            tier2Data = tier2Response.value.data as any;
        }

        // If neither succeeded, return error
        if (!tier1Data && !tier2Data) {
            const errorMsg = tier1Response.status === 'rejected'
                ? tier1Response.reason?.message
                : (tier1Response.value as any)?.error || 'Both prediction tiers failed';
            return {
                success: false,
                error: errorMsg
            };
        }

        // Calculate composite scores (50/50 average if both available, otherwise use available tier)
        let compositeProperties: Record<string, number>;
        let tier1Only = false;

        if (tier1Data && tier2Data) {
            // Both available - average 50/50
            compositeProperties = {
                strength: (tier1Data.strength + tier2Data.strength) / 2,
                flexibility: (tier1Data.flexibility + tier2Data.flexibility) / 2,
                degradability: (tier1Data.degradability + tier2Data.degradability) / 2,
                sustainability: (tier1Data.sustainability + tier2Data.sustainability) / 2,
                sas_score: tier1Data.sas_score ?? 5 // SAS score only from Tier 1
            };
            console.log('Composite (50/50 Tier1+Tier2):', compositeProperties);
        } else if (tier1Data) {
            // Only Tier 1 available
            compositeProperties = {
                strength: tier1Data.strength,
                flexibility: tier1Data.flexibility,
                degradability: tier1Data.degradability,
                sustainability: tier1Data.sustainability,
                sas_score: tier1Data.sas_score ?? 5
            };
            tier1Only = true;
            console.log('Using Tier 1 only:', compositeProperties);
        } else {
            // Only Tier 2 available
            compositeProperties = {
                strength: tier2Data!.strength,
                flexibility: tier2Data!.flexibility,
                degradability: tier2Data!.degradability,
                sustainability: tier2Data!.sustainability,
                sas_score: 5 // Default when only Tier 2
            };
            console.log('Using Tier 2 only:', compositeProperties);
        }

        return {
            success: true,
            properties: compositeProperties,
            tier1Only
        };
    } catch (error: any) {
        return {
            success: false,
            error: error.message || 'Failed to connect to prediction server'
        };
    }
};

/**
 * Helper function to format validation errors for display
 */
export const formatValidationErrors = (errors: ValidationError[]): string[] => {
    return errors.map(err => {
        const ruleInfo = ValidationRuleMessages[err.code];
        let formatted = `${ruleInfo.title}`;
        if (err.moleculeName) {
            formatted = `[${err.moleculeName}] ${formatted}`;
        }
        if (err.details) {
            formatted += `: ${err.details}`;
        }
        return formatted;
    });
};

/**
 * Get suggestion for a validation error
 */
export const getErrorSuggestion = (code: ValidationRuleCode): string => {
    return ValidationRuleMessages[code]?.suggestion || "Check your molecular structure.";
};

// =============================================================================
// SMILES TO MOLECULES PARSER
// =============================================================================

// Element color mapping for visualization
const ELEMENT_COLORS: Record<string, string> = {
    C: '#404040',   // Carbon - dark gray
    H: '#A0D8FF',   // Hydrogen - light cyan
    O: '#FF0000',   // Oxygen - red
    N: '#0000FF',   // Nitrogen - blue
    S: '#FFFF00',   // Sulfur - yellow
    P: '#FFA500',   // Phosphorus - orange
    F: '#90E050',   // Fluorine - light green
    Cl: '#1FF01F',  // Chlorine - green
    Br: '#A52A2A',  // Bromine - brown
    I: '#940094',   // Iodine - purple
    Si: '#F0C8A0',  // Silicon - tan
    default: '#FF69B4' // Unknown - pink
};

// Element atomic weights
const ELEMENT_WEIGHTS: Record<string, number> = {
    C: 12.011, H: 1.008, O: 15.999, N: 14.007, S: 32.065,
    P: 30.974, F: 18.998, Cl: 35.453, Br: 79.904, I: 126.904,
    Si: 28.086
};

/**
 * Result of parsing SMILES into molecule structures
 */
export interface ParsedSmilesResult {
    success: boolean;
    molecules: PlacedMolecule[];
    error?: string;
    atomCount?: number;
    bondCount?: number;
}

/**
 * Parse a SMILES string into PlacedMolecule structures
 * Uses RDKit to extract atom and bond information
 * Generates 2D coordinates for visualization
 * 
 * @param smiles - SMILES string to parse
 * @param startId - Starting ID for molecules (default: 1)
 * @param centerOffset - Position offset for the structure center
 * @returns ParsedSmilesResult with array of PlacedMolecule objects
 */
export const parseSmilestoMolecules = async (
    smiles: string,
    startId: number = 1,
    centerOffset: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 }
): Promise<ParsedSmilesResult> => {
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);

        if (!mol) {
            return {
                success: false,
                molecules: [],
                error: 'Invalid SMILES: Could not parse molecule'
            };
        }

        try {
            // Get Molblock which contains 2D coordinates
            const molblock = mol.get_molblock();
            const lines = molblock.split('\n');

            // Parse counts line (4th line in Molfile V2000)
            // Format: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
            // aaa = number of atoms, bbb = number of bonds
            const countsLine = lines[3];
            const numAtoms = parseInt(countsLine.substring(0, 3).trim());
            const numBonds = parseInt(countsLine.substring(3, 6).trim());

            if (numAtoms === 0) {
                return {
                    success: false,
                    molecules: [],
                    error: 'No atoms found in molecule'
                };
            }

            // Scale factor for coordinates
            const scale = 2.5;

            // Parse atom block (starts at line 5, 0-indexed line 4)
            const atoms: Array<{ x: number; y: number; z: number; symbol: string }> = [];
            for (let i = 0; i < numAtoms; i++) {
                const atomLine = lines[4 + i];
                // V2000 format: xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaa
                const x = parseFloat(atomLine.substring(0, 10).trim());
                const y = parseFloat(atomLine.substring(10, 20).trim());
                const z = parseFloat(atomLine.substring(20, 30).trim());
                const symbol = atomLine.substring(31, 34).trim();
                atoms.push({ x, y, z, symbol });
            }

            // Calculate center of mass for centering
            let sumX = 0, sumY = 0;
            for (const atom of atoms) {
                sumX += atom.x;
                sumY += atom.y;
            }
            const centerX = sumX / atoms.length;
            const centerY = sumY / atoms.length;

            // Create PlacedMolecule for each atom
            const molecules: PlacedMolecule[] = [];
            const atomIdMap = new Map<number, number>();

            for (let i = 0; i < atoms.length; i++) {
                const atom = atoms[i];
                const element = atom.symbol || 'C';
                const moleculeId = startId + i;
                atomIdMap.set(i, moleculeId);

                // Center and scale coordinates, project onto XZ plane (Y=0 for ground)
                const posX = (atom.x - centerX) * scale + centerOffset.x;
                const posZ = (atom.y - centerY) * scale + centerOffset.z;
                const posY = centerOffset.y;

                molecules.push({
                    id: moleculeId,
                    name: element,
                    formula: element,
                    smiles: `[${element}]`,
                    icon: element.charAt(0),
                    color: ELEMENT_COLORS[element] || ELEMENT_COLORS.default,
                    weight: ELEMENT_WEIGHTS[element] || 12,
                    position: { x: posX, y: posY, z: posZ },
                    connections: []
                });
            }

            // Parse bond block
            let bondCount = 0;
            for (let i = 0; i < numBonds; i++) {
                const bondLine = lines[4 + numAtoms + i];
                // V2000 format: 111222tttsssxxxrrrccc
                // 111 = first atom, 222 = second atom, ttt = bond type
                const atom1Idx = parseInt(bondLine.substring(0, 3).trim()) - 1; // 1-indexed to 0-indexed
                const atom2Idx = parseInt(bondLine.substring(3, 6).trim()) - 1;

                const mol1Id = atomIdMap.get(atom1Idx);
                const mol2Id = atomIdMap.get(atom2Idx);

                if (mol1Id !== undefined && mol2Id !== undefined) {
                    const mol1 = molecules.find(m => m.id === mol1Id);
                    const mol2 = molecules.find(m => m.id === mol2Id);

                    if (mol1 && mol2) {
                        if (!mol1.connections.includes(mol2Id)) {
                            mol1.connections.push(mol2Id);
                        }
                        if (!mol2.connections.includes(mol1Id)) {
                            mol2.connections.push(mol1Id);
                        }
                        bondCount++;
                    }
                }
            }

            return {
                success: true,
                molecules,
                atomCount: numAtoms,
                bondCount
            };

        } finally {
            mol.delete();
        }
    } catch (error) {
        return {
            success: false,
            molecules: [],
            error: `Failed to parse SMILES: ${error}`
        };
    }
};
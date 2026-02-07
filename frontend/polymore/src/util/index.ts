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
export const detectFunctionalGroups = async (smiles: string): Promise<Array<{name: string; count: number}>> => {
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
    
    const results: Array<{name: string; count: number}> = [];
    
    try {
        const rdkit = await getRDKit();
        const mol = rdkit.get_mol(smiles);
        
        if (!mol) return results;
        
        try {
            for (const [name, smartsPattern] of Object.entries(patterns)) {
                const query = rdkit.get_qmol(smartsPattern);
                if (query) {
                    const matches = mol.get_substruct_matches(query);
                    const matchArray = JSON.parse(matches);
                    if (matchArray.length > 0) {
                        results.push({ name, count: matchArray.length });
                    }
                    query.delete();
                }
            }
        } finally {
            mol.delete();
        }
    } catch {
        // Return empty results on error
    }
    
    return results;
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
 * API endpoint for property prediction (to be called after validation)
 */
export const predictPropertiesFromBackend = async (
    smiles: string
): Promise<{ success: boolean; properties?: Record<string, number>; error?: string }> => {
    try {
        const response = await fetch('http://localhost:8000/api/predict-properties', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return {
                success: false,
                error: errorData.error || `Server error: ${response.status}`
            };
        }
        
        const data = await response.json();
        return {
            success: true,
            properties: data.properties
        };
    } catch (error) {
        return {
            success: false,
            error: `Failed to connect to prediction server: ${error}`
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
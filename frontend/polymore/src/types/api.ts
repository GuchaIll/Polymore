// Generic API Response Wrapper
export interface ResponseModel<T> {
    status: number;
    message: string;
    data?: T;
    error?: string;
}

// Request Models
export interface AnalysisRequest {
    smiles: string;
}

// Data Models
export interface TierOneAnalysis {
    strength: number;
    flexibility: number;
    degradability: number;
    sustainability: number;
    sas_score: number;
    meta: Record<string, any>;
}

export interface TaskSubmission {
    task_id: string;
    status: string;
    message: string;
}

export interface Task {
    task_id: string;
    status: 'PENDING' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | string;
    result?: Record<string, any>;
    error?: string;
    progress?: number;
    message?: string;
}

// This might be the shape of 'result' in TaskStatusResponse when success
export interface TierTwoAnalysis {
    strength: number;
    flexibility: number;
    degradability: number;
    sustainability: number;
    meta: Record<string, any>;
}

export interface SmilesValidationResponse {
    isValid: boolean;
    canonicalSmiles?: string;
    error?: string;
    molecularWeight?: number;
    formula?: string;
}

export interface Position {
    x: number;
    y: number;
    z: number;
}

export interface MoleculeData {
    id: number;
    smiles: string;
    name: string;
    position: Position;
    connections: number[];
}

export interface ValidatePolymerRequest {
    molecules: MoleculeData[];
    generatedSmiles: string;
}

export interface ValidationResponse {
    isValid: boolean;
    canonicalSmiles: string;
    errors: string[];
    warnings: string[];
    polymerType: string;
    molecularWeight?: number;
    aromaticRings?: number;
}

export interface Tier3AnalysisResult {
    id: number;
    smiles: string;
    result: any;
    created_at: string;
    updated_at: string;
}

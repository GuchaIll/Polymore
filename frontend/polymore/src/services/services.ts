import client from '../api/client';
import {
    ResponseModel,
    TierOneAnalysis,
    TaskSubmission,
    Task,
    SmilesValidationResponse,
    ValidatePolymerRequest,
    ValidationResponse,
} from '../types/api';

/**
 * Predict polymer properties from SMILES using heuristics (Tier 1).
 */
export const predictTier1 = async (smiles: string): Promise<ResponseModel<TierOneAnalysis>> => {
    try {
        const response = await client.post<ResponseModel<TierOneAnalysis>>('/predict/tier-1', { smiles });
        return response.data;
    } catch (error: any) {
        throw handleApiError(error);
    }
};

/**
 * Submit a high-compute analysis task (Tier 2).
 */
export const predictTier2 = async (smiles: string): Promise<ResponseModel<TaskSubmission>> => {
    try {
        const response = await client.post<ResponseModel<TaskSubmission>>('/predict/tier-2', { smiles });
        return response.data;
    } catch (error: any) {
        throw handleApiError(error);
    }
};

/**
 * Retrieve the status and result of a background task.
 */
export const getTaskStatus = async (taskId: string): Promise<ResponseModel<Task>> => {
    try {
        const response = await client.get<ResponseModel<Task>>(`/tasks/${taskId}`);
        return response.data;
    } catch (error: any) {
        throw handleApiError(error);
    }
};

/**
 * Validate a single SMILES string.
 */
export const validateSmiles = async (smiles: string): Promise<ResponseModel<SmilesValidationResponse>> => {
    try {
        const response = await client.post<ResponseModel<SmilesValidationResponse>>('/validate-smiles', { smiles });
        return response.data;
    } catch (error: any) {
        throw handleApiError(error);
    }
}

/**
 * Validate a polymer configuration.
 */
export const validatePolymer = async (request: ValidatePolymerRequest): Promise<ResponseModel<ValidationResponse>> => {
    try {
        const response = await client.post<ResponseModel<ValidationResponse>>('/validate-polymer', request);
        return response.data;
    } catch (error: any) {
        throw handleApiError(error);
    }
}

/**
 * Helper to extract error message from API response or Axios error.
 */
const handleApiError = (error: any): Error => {
    if (error.response && error.response.data) {
        const apiError = error.response.data as ResponseModel<null>;
        return new Error(apiError.error || apiError.message || 'Unknown API error');
    }
    return new Error(error.message || 'Network error');
};

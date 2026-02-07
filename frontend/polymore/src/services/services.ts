import client from '../api/client';
import {
    ResponseModel,
    TierOneAnalysis,
    TierTwoAnalysis,
    Task,
    TaskDetail,
    TaskListResponse,
    TaskUpdateRequest,
    TaskDeleteResponse,
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
 * Predict polymer properties using GNN model (Tier 2).
 * Now synchronous - returns analysis directly.
 */
export const predictTier2 = async (smiles: string): Promise<ResponseModel<TierTwoAnalysis>> => {
    try {
        const response = await client.post<ResponseModel<TierTwoAnalysis>>('/predict/tier-2', { smiles });
        return response.data;
    } catch (error: any) {
        throw handleApiError(error);
    }
};

/**
 * Submit a tier-3 prediction task to the Celery queue (GPU-based analysis).
 */
export const predictTier3 = async (smiles: string): Promise<ResponseModel<TaskSubmission>> => {
    try {
        const response = await client.post<ResponseModel<TaskSubmission>>('/predict/tier-3', { smiles });
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
 * List all tasks with optional filtering and pagination.
 */
export const listTasks = async (params?: {
    type?: string;
    status?: string;
    limit?: number;
    offset?: number;
}): Promise<ResponseModel<TaskListResponse>> => {
    try {
        const response = await client.get<ResponseModel<TaskListResponse>>('/tasks', { params });
        return response.data;
    } catch (error: any) {
        throw handleApiError(error);
    }
};

/**
 * Update a task's details (status, progress, result, etc.).
 */
export const updateTask = async (
    taskId: string,
    updateData: TaskUpdateRequest
): Promise<ResponseModel<TaskDetail>> => {
    try {
        const response = await client.patch<ResponseModel<TaskDetail>>(`/tasks/${taskId}`, updateData);
        return response.data;
    } catch (error: any) {
        throw handleApiError(error);
    }
};

/**
 * Delete a task from the database.
 */
export const deleteTask = async (taskId: string): Promise<ResponseModel<TaskDeleteResponse>> => {
    try {
        const response = await client.delete<ResponseModel<TaskDeleteResponse>>(`/tasks/${taskId}`);
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

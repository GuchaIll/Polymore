import { useState, useCallback } from 'react';
import { validatePolymer } from '../services/services';
import { ValidationResponse, ValidatePolymerRequest } from '../types/api';

interface UseServiceState<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
}

export const usePolymerValidation = () => {
    const [state, setState] = useState<UseServiceState<ValidationResponse>>({
        data: null,
        loading: false,
        error: null,
    });

    const validate = useCallback(async (request: ValidatePolymerRequest) => {
        setState(prev => ({ ...prev, loading: true, error: null }));
        try {
            const response = await validatePolymer(request);
            setState({ data: response.data || null, loading: false, error: null });
            return response.data;
        } catch (err: any) {
            setState({ data: null, loading: false, error: err.message });
            throw err;
        }
    }, []);

    return { ...state, validate };
};

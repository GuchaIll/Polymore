import { useState, useCallback } from 'react';
import { validateSmiles } from '../services/services';
import { SmilesValidationResponse } from '../types/api';

interface UseServiceState<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
}

export const useSmilesValidation = () => {
    const [state, setState] = useState<UseServiceState<SmilesValidationResponse>>({
        data: null,
        loading: false,
        error: null,
    });

    const validate = useCallback(async (smiles: string) => {
        setState(prev => ({ ...prev, loading: true, error: null }));
        try {
            const response = await validateSmiles(smiles);
            setState({ data: response.data || null, loading: false, error: null });
            return response.data;
        } catch (err: any) {
            setState({ data: null, loading: false, error: err.message });
            throw err;
        }
    }, []);

    return { ...state, validate };
};

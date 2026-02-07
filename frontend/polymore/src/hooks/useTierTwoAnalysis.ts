import { useState, useCallback } from 'react';
import { predictTier2 } from '../services/services';
import { TierTwoAnalysis } from '../types/api';

interface UseServiceState<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
}

export const useTierTwoAnalysis = () => {
    const [state, setState] = useState<UseServiceState<TierTwoAnalysis>>({
        data: null,
        loading: false,
        error: null,
    });

    const submit = useCallback(async (smiles: string) => {
        setState(prev => ({ ...prev, loading: true, error: null }));
        try {
            const response = await predictTier2(smiles);
            setState({ data: response.data || null, loading: false, error: null });
            return response.data;
        } catch (err: any) {
            setState({ data: null, loading: false, error: err.message });
            throw err;
        }
    }, []);

    return { ...state, submit };
};

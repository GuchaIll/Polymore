import { useState, useCallback } from 'react';
import { predictTier1 } from '../services/services';
import { TierOneAnalysis } from '../types/api';

interface UseServiceState<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
}

export const useTierOneAnalysis = () => {
    const [state, setState] = useState<UseServiceState<TierOneAnalysis>>({
        data: null,
        loading: false,
        error: null,
    });

    const analyze = useCallback(async (smiles: string) => {
        setState(prev => ({ ...prev, loading: true, error: null }));
        try {
            const response = await predictTier1(smiles);
            setState({ data: response.data || null, loading: false, error: null });
            return response.data;
        } catch (err: any) {
            setState({ data: null, loading: false, error: err.message });
            throw err;
        }
    }, []);

    return { ...state, analyze };
};

import { useState, useCallback } from 'react';
import { getTaskStatus } from '../services/services';
import { Task } from '../types/api';

interface UseServiceState<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
}

export const useTaskPolling = () => {
    const [state, setState] = useState<UseServiceState<Task>>({
        data: null,
        loading: false,
        error: null,
    });

    const checkStatus = useCallback(async (taskId: string) => {
        setState(prev => ({ ...prev, loading: true, error: null }));
        try {
            const response = await getTaskStatus(taskId);
            setState({ data: response.data || null, loading: false, error: null });
            return response.data;
        } catch (err: any) {
            setState({ data: null, loading: false, error: err.message });
            throw err;
        }
    }, []);

    return { ...state, checkStatus };
};

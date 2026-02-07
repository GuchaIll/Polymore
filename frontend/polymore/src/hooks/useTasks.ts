import { useState, useCallback } from 'react';
import { listTasks, updateTask, deleteTask } from '../services/services';
import { TaskDetail, TaskListResponse, TaskUpdateRequest, TaskDeleteResponse } from '../types/api';

interface UseTasksState {
    tasks: TaskDetail[];
    total: number;
    loading: boolean;
    error: string | null;
}

interface ListTasksParams {
    type?: string;
    status?: string;
    limit?: number;
    offset?: number;
}

/**
 * Hook for managing task CRUD operations.
 * Provides methods to list, update, and delete tasks.
 */
export const useTasks = () => {
    const [state, setState] = useState<UseTasksState>({
        tasks: [],
        total: 0,
        loading: false,
        error: null,
    });

    /**
     * Fetch tasks with optional filtering and pagination.
     */
    const fetchTasks = useCallback(async (params?: ListTasksParams) => {
        setState(prev => ({ ...prev, loading: true, error: null }));
        try {
            const response = await listTasks(params);
            const data = response.data as TaskListResponse;
            setState({
                tasks: data.tasks,
                total: data.total,
                loading: false,
                error: null,
            });
            return data;
        } catch (err: any) {
            setState(prev => ({
                ...prev,
                loading: false,
                error: err.message,
            }));
            throw err;
        }
    }, []);

    /**
     * Update a specific task.
     */
    const update = useCallback(async (taskId: string, updateData: TaskUpdateRequest) => {
        setState(prev => ({ ...prev, loading: true, error: null }));
        try {
            const response = await updateTask(taskId, updateData);
            const updatedTask = response.data as TaskDetail;

            // Update the task in the local state
            setState(prev => ({
                ...prev,
                tasks: prev.tasks.map(task =>
                    task.task_id === taskId ? updatedTask : task
                ),
                loading: false,
                error: null,
            }));

            return updatedTask;
        } catch (err: any) {
            setState(prev => ({
                ...prev,
                loading: false,
                error: err.message,
            }));
            throw err;
        }
    }, []);

    /**
     * Delete a specific task.
     */
    const remove = useCallback(async (taskId: string) => {
        setState(prev => ({ ...prev, loading: true, error: null }));
        try {
            const response = await deleteTask(taskId);
            const result = response.data as TaskDeleteResponse;

            // Remove the task from local state
            setState(prev => ({
                ...prev,
                tasks: prev.tasks.filter(task => task.task_id !== taskId),
                total: prev.total - 1,
                loading: false,
                error: null,
            }));

            return result;
        } catch (err: any) {
            setState(prev => ({
                ...prev,
                loading: false,
                error: err.message,
            }));
            throw err;
        }
    }, []);

    /**
     * Clear all tasks from local state.
     */
    const clearTasks = useCallback(() => {
        setState({
            tasks: [],
            total: 0,
            loading: false,
            error: null,
        });
    }, []);

    return {
        tasks: state.tasks,
        total: state.total,
        loading: state.loading,
        error: state.error,
        fetchTasks,
        update,
        remove,
        clearTasks,
    };
};

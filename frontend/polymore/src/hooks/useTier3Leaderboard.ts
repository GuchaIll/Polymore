import { useState, useEffect, useCallback } from 'react';
import { getTier3Results } from '../services/services';
import { Tier3AnalysisResult } from '../types/api';

export const useTier3Leaderboard = (initialLimit = 10) => {
    const [results, setResults] = useState<Tier3AnalysisResult[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [page, setPage] = useState(0);
    const [limit, setLimit] = useState(initialLimit);
    const [hasMore, setHasMore] = useState(true);

    const fetchResults = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const skip = page * limit;
            const data = await getTier3Results(skip, limit);

            if (data.data) {
                setResults(data.data);
                if (data.data.length < limit) {
                    setHasMore(false);
                } else {
                    setHasMore(true);
                }
            } else {
                setResults([]);
                setHasMore(false);
            }
        } catch (err: any) {
            setError(err.message || 'Failed to fetch leaderboard data');
        } finally {
            setLoading(false);
        }
    }, [page, limit]);

    useEffect(() => {
        fetchResults();
    }, [fetchResults]);

    const nextPage = () => {
        if (hasMore) {
            setPage(prev => prev + 1);
        }
    };

    const prevPage = () => {
        if (page > 0) {
            setPage(prev => prev - 1);
        }
    };

    const resetPage = () => {
        setPage(0);
    };

    return {
        results,
        loading,
        error,
        page,
        limit,
        hasMore,
        setLimit,
        nextPage,
        prevPage,
        resetPage,
        refresh: fetchResults
    };
};

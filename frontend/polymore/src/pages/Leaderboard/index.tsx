import React from 'react';
import { useTier3Leaderboard } from '../../hooks';
import { ArrowLeft, ArrowRight, Trophy, Link, Clock } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Leaderboard: React.FC = () => {
    const {
        results,
        loading,
        error,
        page,
        hasMore,
        nextPage,
        prevPage
    } = useTier3Leaderboard(10);

    const navigate = useNavigate();

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    return (
        <div className="min-h-screen bg-poly-light-bg dark:bg-poly-bg text-poly-light-text dark:text-poly-text p-8">
            <div className="max-w-7xl mx-auto">
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={() => navigate('/')}
                            className="p-2 rounded-full hover:bg-poly-light-border dark:hover:bg-poly-border transition-colors"
                        >
                            <ArrowLeft className="w-6 h-6" />
                        </button>
                        <h1 className="text-3xl font-bold flex items-center gap-3">
                            <Trophy className="w-8 h-8 text-yellow-500" />
                            Polymer Leaderboard
                        </h1>
                    </div>
                </div>

                {error && (
                    <div className="bg-red-500/10 border border-red-500/50 text-red-500 p-4 rounded-lg mb-6">
                        {error}
                    </div>
                )}

                <div className="bg-white dark:bg-poly-sidebar rounded-xl shadow-lg overflow-hidden border border-poly-light-border dark:border-poly-border">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="bg-poly-light-border dark:bg-poly-border text-left">
                                    <th className="p-4 font-semibold text-sm uppercase tracking-wider">Rank</th>
                                    <th className="p-4 font-semibold text-sm uppercase tracking-wider">SMILES</th>
                                    <th className="p-4 font-semibold text-sm uppercase tracking-wider">Energy (Ha)</th>
                                    <th className="p-4 font-semibold text-sm uppercase tracking-wider">Gap (eV)</th>
                                    <th className="p-4 font-semibold text-sm uppercase tracking-wider">Dipole (Debye)</th>
                                    <th className="p-4 font-semibold text-sm uppercase tracking-wider text-right">Date</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-poly-light-border dark:divide-poly-border">
                                {loading && results.length === 0 ? (
                                    <tr>
                                        <td colSpan={6} className="p-8 text-center text-poly-light-muted dark:text-poly-muted">
                                            Loading leaderboard data...
                                        </td>
                                    </tr>
                                ) : (
                                    results.map((item, index) => (
                                        <tr
                                            key={item.id}
                                            className="hover:bg-poly-light-accent/5 dark:hover:bg-poly-accent/10 transition-colors"
                                        >
                                            <td className="p-4 font-medium text-poly-light-muted dark:text-poly-muted">
                                                #{page * 10 + index + 1}
                                            </td>
                                            <td className="p-4 font-mono text-sm max-w-xs truncate" title={item.smiles}>
                                                <div className="flex items-center gap-2">
                                                    <Link className="w-4 h-4 text-poly-light-muted dark:text-poly-muted opacity-50" />
                                                    {item.smiles}
                                                </div>
                                            </td>
                                            <td className="p-4 font-mono">
                                                {item.result?.energy?.toFixed(4) ?? '-'}
                                            </td>
                                            <td className="p-4 font-mono text-emerald-600 dark:text-emerald-400">
                                                {item.result?.gap?.toFixed(4) ?? '-'}
                                            </td>
                                            <td className="p-4 font-mono text-blue-600 dark:text-blue-400">
                                                {item.result?.dipole?.toFixed(4) ?? '-'}
                                            </td>
                                            <td className="p-4 text-right text-sm text-poly-light-muted dark:text-poly-muted">
                                                <div className="flex items-center justify-end gap-2">
                                                    {formatDate(item.created_at)}
                                                    <Clock className="w-3 h-3 opacity-50" />
                                                </div>
                                            </td>
                                        </tr>
                                    ))
                                )}
                                {!loading && results.length === 0 && !error && (
                                    <tr>
                                        <td colSpan={6} className="p-8 text-center text-poly-light-muted dark:text-poly-muted">
                                            No analysis results found yet. Start predicting!
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>

                    <div className="p-4 border-t border-poly-light-border dark:border-poly-border flex items-center justify-between">
                        <div className="text-sm text-poly-light-muted dark:text-poly-muted">
                            Page {page + 1}
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={prevPage}
                                disabled={page === 0 || loading}
                                className="px-4 py-2 rounded-lg bg-poly-light-border dark:bg-poly-border hover:bg-poly-light-accent hover:text-white dark:hover:bg-poly-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                                <ArrowLeft className="w-4 h-4" /> Previous
                            </button>
                            <button
                                onClick={nextPage}
                                disabled={!hasMore || loading}
                                className="px-4 py-2 rounded-lg bg-poly-light-border dark:bg-poly-border hover:bg-poly-light-accent hover:text-white dark:hover:bg-poly-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                                Next <ArrowRight className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Leaderboard;

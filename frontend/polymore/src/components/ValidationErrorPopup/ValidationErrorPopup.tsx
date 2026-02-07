/**
 * Module: ValidationErrorPopup.tsx
 * Purpose: Modal popup component for displaying detailed validation errors
 * Inputs: Array of ValidationError objects, visibility state, close callback
 * Outputs: Rendered modal with error details, descriptions, suggestions, and auto-repair
 */

import React from 'react';
import { ValidationError, ValidationRuleCode, ValidationRuleMessages, SmilesRepairResult } from '../../util';

interface ValidationErrorPopupProps {
    /** Whether the popup is visible */
    isOpen: boolean;
    /** Callback to close the popup */
    onClose: () => void;
    /** Array of validation errors to display */
    errors: ValidationError[];
    /** Array of validation warnings to display */
    warnings?: ValidationError[];
    /** Title for the popup */
    title?: string;
    /** Auto-repair result if available */
    repairResult?: SmilesRepairResult | null;
    /** Callback to trigger auto-repair */
    onAutoRepair?: () => void;
    /** Whether repair is in progress */
    isRepairing?: boolean;
}

/**
 * Get severity color class based on rule code
 */
const getSeverityColor = (code: ValidationRuleCode): string => {
    // Layer 1 errors are critical
    if ([
        ValidationRuleCode.RULE_1_VALENCE,
        ValidationRuleCode.RULE_2_DISCONNECTED,
        ValidationRuleCode.RULE_3_RING_CLOSURE,
        ValidationRuleCode.RULE_4_INVALID_SYNTAX
    ].includes(code)) {
        return 'border-red-500 bg-red-50 dark:bg-red-900/20';
    }
    // Layer 2 errors are important
    if ([
        ValidationRuleCode.RULE_5_MIN_REACTIVE_SITES,
        ValidationRuleCode.RULE_6_FULLY_CAPPED,
        ValidationRuleCode.RULE_8_REPEAT_SIZE
    ].includes(code)) {
        return 'border-orange-500 bg-orange-50 dark:bg-orange-900/20';
    }
    // Layer 3 errors are warnings
    return 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20';
};

/**
 * Get icon based on error severity
 */
const getIcon = (code: ValidationRuleCode, isWarning: boolean): string => {
    if (isWarning) return '!';
    if ([
        ValidationRuleCode.RULE_1_VALENCE,
        ValidationRuleCode.RULE_2_DISCONNECTED,
        ValidationRuleCode.RULE_3_RING_CLOSURE,
        ValidationRuleCode.RULE_4_INVALID_SYNTAX
    ].includes(code)) {
        return 'X';
    }
    return '!';
};

/**
 * ValidationErrorPopup - Modal for displaying detailed validation errors
 * 
 * @example
 * <ValidationErrorPopup
 *   isOpen={showPopup}
 *   onClose={() => setShowPopup(false)}
 *   errors={validationResult.errors}
 *   warnings={validationResult.warnings}
 * />
 */
const ValidationErrorPopup: React.FC<ValidationErrorPopupProps> = ({
    isOpen,
    onClose,
    errors,
    warnings = [],
    title = 'Validation Failed',
    repairResult,
    onAutoRepair,
    isRepairing = false
}) => {
    // Press Escape to close - must be before any conditional returns
    React.useEffect(() => {
        if (!isOpen) return;
        
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    // Click outside to close
    const handleBackdropClick = (e: React.MouseEvent) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    return (
        <div 
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
            onClick={handleBackdropClick}
        >
            <div className="bg-white dark:bg-poly-card max-w-2xl w-full mx-4 rounded-xl shadow-2xl max-h-[80vh] overflow-hidden flex flex-col animate-fadeIn">
                {/* Header */}
                <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between bg-red-500 text-white">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center text-lg font-bold">
                            !
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold">{title}</h2>
                            <p className="text-sm text-white/80">
                                {errors.length} error{errors.length !== 1 ? 's' : ''}
                                {warnings.length > 0 && `, ${warnings.length} warning${warnings.length !== 1 ? 's' : ''}`}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                        aria-label="Close"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {/* Errors */}
                    {errors.length > 0 && (
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-red-600 dark:text-red-400 uppercase tracking-wide">
                                Errors (Must Fix)
                            </h3>
                            {errors.map((error, index) => (
                                <ErrorCard key={`error-${index}`} error={error} isWarning={false} />
                            ))}
                        </div>
                    )}

                    {/* Warnings */}
                    {warnings.length > 0 && (
                        <div className="space-y-3 mt-6">
                            <h3 className="text-sm font-semibold text-yellow-600 dark:text-yellow-400 uppercase tracking-wide">
                                Warnings (Recommended to Fix)
                            </h3>
                            {warnings.map((warning, index) => (
                                <ErrorCard key={`warning-${index}`} error={warning} isWarning={true} />
                            ))}
                        </div>
                    )}

                    {/* Auto-Repair Section */}
                    {(repairResult || onAutoRepair) && (
                        <div className="mt-6 border-t border-gray-200 dark:border-gray-700 pt-4">
                            <h3 className="text-sm font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wide mb-3">
                                Auto-Repair
                            </h3>
                            
                            {/* Show repair result if available */}
                            {repairResult && (
                                <div className={`border-l-4 rounded-r-lg p-4 mb-3 ${
                                    repairResult.success 
                                        ? 'border-green-500 bg-green-50 dark:bg-green-900/20' 
                                        : 'border-gray-500 bg-gray-50 dark:bg-gray-900/20'
                                }`}>
                                    <div className="flex items-start gap-3">
                                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5 ${
                                            repairResult.success 
                                                ? 'bg-green-500 text-white' 
                                                : 'bg-gray-500 text-white'
                                        }`}>
                                            {repairResult.success ? '\u2713' : '?'}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <h4 className="font-semibold text-gray-900 dark:text-white">
                                                {repairResult.success 
                                                    ? (repairResult.wasModified ? 'Repair Successful' : 'SMILES Already Valid')
                                                    : 'Repair Unsuccessful'}
                                            </h4>
                                            
                                            {/* Repair steps */}
                                            {repairResult.repairSteps.length > 0 && (
                                                <ul className="text-sm text-gray-600 dark:text-gray-300 mt-1 list-disc list-inside">
                                                    {repairResult.repairSteps.map((step, idx) => (
                                                        <li key={idx}>{step}</li>
                                                    ))}
                                                </ul>
                                            )}
                                            
                                            {/* Show repaired SMILES if different */}
                                            {repairResult.success && repairResult.wasModified && (
                                                <div className="mt-2 space-y-1">
                                                    <div className="text-xs">
                                                        <span className="text-gray-500 dark:text-gray-400">Original: </span>
                                                        <code className="font-mono text-red-600 dark:text-red-400 break-all">
                                                            {repairResult.original}
                                                        </code>
                                                    </div>
                                                    <div className="text-xs">
                                                        <span className="text-gray-500 dark:text-gray-400">Repaired: </span>
                                                        <code className="font-mono text-green-600 dark:text-green-400 break-all">
                                                            {repairResult.canonical || repairResult.repaired}
                                                        </code>
                                                    </div>
                                                </div>
                                            )}
                                            
                                            {/* Error message if failed */}
                                            {!repairResult.success && repairResult.error && (
                                                <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                                                    {repairResult.error}
                                                </p>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )}
                            
                            {/* Auto-repair button */}
                            {onAutoRepair && (
                                <button
                                    onClick={onAutoRepair}
                                    disabled={isRepairing}
                                    className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                >
                                    {isRepairing ? (
                                        <>
                                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                            Repairing...
                                        </>
                                    ) : (
                                        <>
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                            </svg>
                                            {repairResult ? 'Try Repair Again' : 'Attempt Auto-Repair'}
                                        </>
                                    )}
                                </button>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 flex justify-between items-center">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                        Fix {errors.length > 0 ? 'errors' : 'warnings'} to continue
                    </p>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-poly-accent text-white rounded-lg hover:bg-poly-accent/90 transition-colors font-medium"
                    >
                        Got it
                    </button>
                </div>
            </div>
        </div>
    );
};

/**
 * ErrorCard - Individual error display component
 */
const ErrorCard: React.FC<{ error: ValidationError; isWarning: boolean }> = ({ error, isWarning }) => {
    const ruleInfo = ValidationRuleMessages[error.code];
    const severityColor = isWarning 
        ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20' 
        : getSeverityColor(error.code);
    const icon = getIcon(error.code, isWarning);
    const iconBg = isWarning 
        ? 'bg-yellow-500' 
        : error.code.startsWith('RULE_1') || error.code.startsWith('RULE_2') || 
          error.code.startsWith('RULE_3') || error.code.startsWith('RULE_4') 
          ? 'bg-red-500' 
          : 'bg-orange-500';

    return (
        <div className={`border-l-4 rounded-r-lg p-4 ${severityColor}`}>
            <div className="flex items-start gap-3">
                {/* Icon */}
                <div className={`w-6 h-6 ${iconBg} text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5`}>
                    {icon}
                </div>
                
                <div className="flex-1 min-w-0">
                    {/* Title */}
                    <div className="flex items-center gap-2 flex-wrap">
                        <h4 className="font-semibold text-gray-900 dark:text-white">
                            {error.message || ruleInfo?.title || 'Unknown Error'}
                        </h4>
                        {error.moleculeName && (
                            <span className="text-xs px-2 py-0.5 bg-gray-200 dark:bg-gray-600 rounded text-gray-600 dark:text-gray-300">
                                {error.moleculeName}
                            </span>
                        )}
                    </div>

                    {/* Description */}
                    <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                        {ruleInfo?.description || 'Unknown validation error occurred.'}
                    </p>

                    {/* Details */}
                    {error.details && (
                        <div className="mt-2 text-xs font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded text-gray-700 dark:text-gray-300 overflow-x-auto">
                            {error.details}
                        </div>
                    )}

                    {/* SMILES if present */}
                    {error.smiles && (
                        <div className="mt-2 text-xs">
                            <span className="text-gray-500 dark:text-gray-400">SMILES: </span>
                            <code className="font-mono text-gray-700 dark:text-gray-300">{error.smiles}</code>
                        </div>
                    )}

                    {/* Suggestion */}
                    <div className="mt-3 flex items-start gap-2 text-sm">
                        <span className="text-poly-accent dark:text-green-400 font-medium">Suggestion:</span>
                        <span className="text-gray-600 dark:text-gray-300">
                            {ruleInfo?.suggestion || 'Review and correct your molecular structure.'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ValidationErrorPopup;

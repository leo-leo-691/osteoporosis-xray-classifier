import React, { useEffect, useState } from 'react';
import {
  Table,
  Grid3X3,
  RefreshCw,
  AlertCircle,
  FileCheck2,
  ArrowRight
} from 'lucide-react';
import { API_BASE_URL } from '../config';
import type { MetricsResponse } from '../types';

export const EvaluationView: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/metrics`);
      if (!response.ok) {
        throw new Error(`Failed to load evaluation metrics: HTTP ${response.status}`);
      }
      const data: MetricsResponse = await response.json();
      setMetrics(data);
    } catch (err: any) {
      console.error('Metrics fetch error:', err);
      setError(err.message || 'Unable to retrieve evaluation metrics from server.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 flex flex-col items-center justify-center space-y-4">
        <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin" />
        <span className="text-xs font-mono text-slate-400">Loading held-out test set metrics...</span>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="p-6 rounded-2xl bg-rose-950/30 border border-rose-800 text-rose-300 space-y-3">
          <div className="flex items-center gap-2 font-semibold">
            <AlertCircle className="w-5 h-5 text-rose-400" />
            <span>Evaluation Metrics Unavailable</span>
          </div>
          <p className="text-xs text-rose-300 leading-relaxed">{error}</p>
          <button
            onClick={fetchMetrics}
            className="px-3 py-1.5 rounded-lg bg-rose-800 hover:bg-rose-700 text-white text-xs font-medium transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Row totals for confusion matrix percentage share
  const rowTotals = metrics.confusion_matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const totalSamples = rowTotals.reduce((a, b) => a + b, 0);
  const totalCorrect = metrics.confusion_matrix.reduce((acc, row, i) => acc + row[i], 0);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
      {/* Title & Scope */}
      <div>
        <h1 className="text-xl sm:text-2xl font-semibold text-slate-100 tracking-tight flex items-center gap-2.5">
          <FileCheck2 className="w-5 h-5 text-cyan-400" />
          Model Architecture & Held-Out Test Set
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
          Benchmark evaluation and performance characteristics measured on the fixed held-out test split.
        </p>
      </div>

      {/* Hero Stat Row: All numbers in the same neutral phosphor/slate colour */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Test Accuracy */}
        <div className="rounded-2xl border border-slate-800 bg-[#0d1219] p-5 space-y-2 shadow-sm">
          <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
            Test Accuracy
          </span>
          <div className="text-3xl font-bold font-mono tabular-nums text-slate-100">
            {(metrics.accuracy * 100).toFixed(2)}%
          </div>
          <div className="text-xs font-mono text-slate-400 pt-1 border-t border-slate-800/80">
            n = {totalSamples} · {totalCorrect} correct · 95% CI {(metrics.bootstrap_95_ci[0] * 100).toFixed(1)}–{(metrics.bootstrap_95_ci[1] * 100).toFixed(1)}%
          </div>
        </div>

        {/* Macro F1 */}
        <div className="rounded-2xl border border-slate-800 bg-[#0d1219] p-5 space-y-2 shadow-sm">
          <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
            Macro F1-Score
          </span>
          <div className="text-3xl font-bold font-mono tabular-nums text-slate-100">
            {(metrics.macro_f1 * 100).toFixed(2)}%
          </div>
          <div className="text-xs font-mono text-slate-400 pt-1 border-t border-slate-800/80">
            Weighted F1: {(metrics.weighted_f1 * 100).toFixed(2)}%
          </div>
        </div>

        {/* Macro ROC-AUC */}
        <div className="rounded-2xl border border-slate-800 bg-[#0d1219] p-5 space-y-2 shadow-sm">
          <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">
            Macro ROC-AUC
          </span>
          <div className="text-3xl font-bold font-mono tabular-nums text-slate-100">
            {metrics.roc_auc_macro.toFixed(4)}
          </div>
          <div className="text-xs font-mono text-slate-400 pt-1 border-t border-slate-800/80">
            One-vs-Rest multi-class average
          </div>
        </div>
      </div>

      {/* Compact Horizontal Pipeline Architecture Diagram */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d1219] p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-400">
            Network Pipeline Architecture
          </h2>
          <span className="text-[11px] font-mono text-slate-500">AdamW · 25 epochs · Seed 42</span>
        </div>

        {/* Horizontal Pipeline flow */}
        <div className="overflow-x-auto pb-2">
          <div className="min-w-[680px] flex items-center justify-between gap-3 text-xs">
            {/* Step 1: Input */}
            <div className="flex-1 p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider block">Input</span>
              <div className="font-semibold text-slate-200">224 × 224 RGB</div>
              <p className="text-[11px] text-slate-500">ImageNet normalized</p>
            </div>

            <ArrowRight className="w-4 h-4 text-slate-600 shrink-0" />

            {/* Step 2: Backbone */}
            <div className="flex-[1.6] p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider block">Backbone</span>
              <div className="font-semibold text-slate-200">ResNet-50 (Pretrained)</div>
              <p className="text-[11px] text-slate-500">layer3 & layer4 fine-tuned</p>
            </div>

            <ArrowRight className="w-4 h-4 text-slate-600 shrink-0" />

            {/* Step 3: Regularization / Head */}
            <div className="flex-1 p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider block">Classifier Head</span>
              <div className="font-semibold text-slate-200">Dropout(0.35)</div>
              <p className="text-[11px] text-slate-500">Linear(2048 → 3)</p>
            </div>

            <ArrowRight className="w-4 h-4 text-slate-600 shrink-0" />

            {/* Step 4: Output */}
            <div className="flex-1 p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
              <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider block">Output</span>
              <div className="font-semibold text-slate-200">3 Classes</div>
              <p className="text-[11px] text-slate-500">Weighted Cross-Entropy</p>
            </div>
          </div>
        </div>
      </div>

      {/* Per-Class Table & Confusion Matrix Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left: Per-Class Table */}
        <div className="lg:col-span-7 rounded-2xl border border-slate-800 bg-[#0d1219] p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Table className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-semibold text-slate-100">Per-Class Metrics</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900/90 text-slate-400 text-[11px] uppercase border-y border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 font-sans font-medium">Class</th>
                  <th className="py-2.5 px-3 text-right">Precision</th>
                  <th className="py-2.5 px-3 text-right">Recall</th>
                  <th className="py-2.5 px-3 text-right">F1-Score</th>
                  <th className="py-2.5 px-3 text-right">ROC-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80 text-xs tabular-nums">
                {metrics.class_names.map((name, i) => {
                  const rocMap = [
                    metrics.roc_auc_normal,
                    metrics.roc_auc_osteopenia,
                    metrics.roc_auc_osteoporosis,
                  ];
                  return (
                    <tr key={name} className="hover:bg-slate-900/40">
                      <td className="py-3 px-3 font-sans font-medium capitalize text-slate-200">
                        {name}
                      </td>
                      <td className="py-3 px-3 text-right text-slate-300">
                        {(metrics.per_class_precision[i] * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 text-right text-slate-300">
                        {(metrics.per_class_recall[i] * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 text-right font-semibold text-slate-100">
                        {(metrics.per_class_f1[i] * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 text-right text-cyan-400">
                        {rocMap[i].toFixed(4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Confusion Matrix with Single Sequential Scale */}
        <div className="lg:col-span-5 rounded-2xl border border-slate-800 bg-[#0d1219] p-6 space-y-3">
          <div className="flex items-center gap-2">
            <Grid3X3 className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-semibold text-slate-100">Confusion Matrix</h2>
          </div>

          <div className="text-[11px] font-mono text-slate-400 space-y-0.5">
            <div>rows = true class, columns = predicted class</div>
            <div className="text-slate-500">% = share of the true class</div>
          </div>

          <div className="pt-2">
            {/* Column Headers (Predicted) */}
            <div className="grid grid-cols-4 gap-1.5 text-center text-[10px] font-mono uppercase text-slate-400 mb-1.5">
              <div className="text-left lowercase text-slate-600 self-end">t \ p</div>
              <div>Norm</div>
              <div>Pen</div>
              <div>Por</div>
            </div>

            {/* Matrix Rows */}
            <div className="space-y-1.5">
              {metrics.confusion_matrix.map((row, rIdx) => {
                const rowName = metrics.class_names[rIdx];
                return (
                  <div key={rowName} className="grid grid-cols-4 gap-1.5 items-center">
                    {/* Row Label (True Class) */}
                    <div className="text-xs font-mono capitalize text-slate-300 truncate">
                      {rowName.slice(0, 5)}.
                    </div>

                    {/* Cells: Sequential single-hue color scale (darker/more saturated = more images) */}
                    {row.map((val, cIdx) => {
                      const isDiagonal = rIdx === cIdx;
                      const maxVal = 42; // Max value in confusion matrix for scaling
                      const intensity = Math.min(1, val / maxVal);
                      const pct = rowTotals[rIdx] > 0 ? ((val / rowTotals[rIdx]) * 100).toFixed(0) : '0';

                      // Single sequential cyan/slate hue: intensity-based background
                      let bgStyle = '';
                      let borderStyle = 'border-slate-800';

                      if (isDiagonal) {
                        // Diagonal visually distinct with cyan tint
                        borderStyle = 'border-cyan-500/40';
                        bgStyle = `rgba(0, 229, 255, ${0.12 + intensity * 0.25})`;
                      } else if (val > 0) {
                        // Off-diagonal errors in subtle neutral tint
                        borderStyle = 'border-slate-700/60';
                        bgStyle = `rgba(148, 163, 184, ${0.05 + intensity * 0.15})`;
                      } else {
                        bgStyle = 'rgba(15, 23, 42, 0.4)';
                      }

                      return (
                        <div
                          key={`${rIdx}-${cIdx}`}
                          className={`h-12 rounded-lg border ${borderStyle} flex flex-col items-center justify-center font-mono tabular-nums transition-colors`}
                          style={{ backgroundColor: bgStyle }}
                          title={`True: ${rowName}, Predicted: ${metrics.class_names[cIdx]} (${val} cases, ${pct}% of true class)`}
                        >
                          <span className="text-sm font-semibold text-slate-100">{val}</span>
                          <span className="text-[10px] text-slate-400 font-sans">{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Limitations: Calm, clearly visible "Read before interpreting" section (exact 4 sentences) */}
      <div className="rounded-2xl border border-slate-800 bg-[#0a0e14] p-6 space-y-4">
        <div className="flex items-center gap-2 text-slate-200">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <h2 className="text-sm font-semibold tracking-tight">Read before interpreting</h2>
        </div>

        <ul className="space-y-3 text-xs sm:text-sm text-slate-300 list-disc list-inside leading-relaxed">
          <li>
            The test result is an image-level evaluation on a combined dataset. Patient-level separation could not be verified for the secondary dataset, so the result may be inflated by patient leakage.
          </li>
          <li>
            Patient-level 5-fold cross-validation on the primary dataset gave 51.00% ± 7.44% accuracy. The cause of the gap is not yet known.
          </li>
          <li>
            No significance testing was done between models.
          </li>
          <li>
            Not validated for clinical use.
          </li>
        </ul>
      </div>
    </div>
  );
};

'use client';

import { useEffect, useState } from 'react';
import {
  RefreshCw, CheckCircle, XCircle, TrendingUp, TrendingDown,
  Database, Cpu, Trophy, AlertTriangle, Clock, ChevronDown, ChevronUp
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Cell
} from 'recharts';
import { triggerRetrain, getModelRegistry, checkNewData, RetrainResult, ModelRegistryEntry } from '@/lib/api';

function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="text-white font-semibold">{(value * 100).toFixed(2)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${value * 100}%`, background: color }} />
      </div>
    </div>
  );
}

export default function RetrainPage() {
  const [retrainResult, setRetrainResult] = useState<RetrainResult | null>(null);
  const [registry, setRegistry] = useState<ModelRegistryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [dataInfo, setDataInfo] = useState<any>(null);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchRegistry = async () => {
    try {
      const [reg, data] = await Promise.all([getModelRegistry(), checkNewData()]);
      setRegistry(reg.models);
      setDataInfo(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => { fetchRegistry(); }, []);

  const handleRetrain = async () => {
    setLoading(true);
    setError(null);
    setRetrainResult(null);
    try {
      const result = await triggerRetrain();
      setRetrainResult(result);
      await fetchRegistry();
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Retraining failed. Check backend logs.');
    } finally {
      setLoading(false);
    }
  };

  // Model comparison bar data
  const comparisonData = retrainResult?.all_model_metrics
    ? Object.entries(retrainResult.all_model_metrics).map(([name, metrics]: [string, any]) => ({
        name: name.replace('RandomForest', 'RF').replace('LogisticRegression', 'LR').replace('XGBoost', 'XGB'),
        'ROC-AUC': parseFloat((metrics.roc_auc * 100).toFixed(2)),
        'F1 Score': parseFloat((metrics.f1_score * 100).toFixed(2)),
        'Recall': parseFloat((metrics.recall * 100).toFixed(2)),
        'Precision': parseFloat((metrics.precision * 100).toFixed(2)),
      }))
    : [];

  // Feature importance for chart
  const featureData = retrainResult?.feature_importance
    ? Object.entries(retrainResult.feature_importance)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .slice(0, 12)
        .map(([feature, importance]) => ({ feature, importance: parseFloat(((importance as number) * 100).toFixed(3)) }))
    : [];

  const COLORS = ['#818cf8', '#22d3ee', '#34d399', '#f59e0b'];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-1">
          Auto <span className="gradient-text">Retraining</span> Pipeline
        </h1>
        <p className="text-slate-400">
          Performance-gated model update — production model upgraded ONLY if new model is better
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Retrain Panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Data Status */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <Database className="w-4 h-4 text-cyan-400" />
              New Data Status
            </h3>
            {dataInfo ? (
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  {dataInfo.new_data_available ? (
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-400" />
                  )}
                  <span className={dataInfo.new_data_available ? 'text-emerald-400' : 'text-red-400'}>
                    {dataInfo.new_data_available ? 'New data available' : 'No new data found'}
                  </span>
                </div>
                {dataInfo.new_data_available && (
                  <div className="text-slate-400 text-xs pl-6">
                    {dataInfo.record_count.toLocaleString()} new transactions ready
                  </div>
                )}
              </div>
            ) : (
              <div className="text-slate-500 text-sm">Checking...</div>
            )}
          </div>

          {/* Retrain Button */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <RefreshCw className="w-4 h-4 text-violet-400" />
              Trigger Retraining
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              This will train new models on combined data and compare with the current production model.
              The production model is replaced <strong className="text-white">ONLY</strong> if performance improves.
            </p>

            <button
              onClick={handleRetrain}
              disabled={loading}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-violet-500 to-cyan-500 hover:from-violet-600 hover:to-cyan-600 text-white font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Training models... (may take ~30s)
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  Start Auto-Retraining
                </>
              )}
            </button>

            {error && (
              <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs">
                {error}
              </div>
            )}
          </div>

          {/* Pipeline steps */}
          <div className="glass-card rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-3">Retraining Pipeline Steps</h3>
            <div className="space-y-2">
              {[
                { step: '1', label: 'Load new transactions', color: 'bg-violet-500' },
                { step: '2', label: 'Merge with existing data', color: 'bg-cyan-500' },
                { step: '3', label: 'Preprocess + SMOTE', color: 'bg-emerald-500' },
                { step: '4', label: 'Train LR + RF + XGBoost', color: 'bg-amber-500' },
                { step: '5', label: 'Evaluate all models', color: 'bg-orange-500' },
                { step: '6', label: 'Compare vs production', color: 'bg-red-500' },
                { step: '7', label: 'Gate: upgrade if better', color: 'bg-pink-500' },
              ].map(({ step, label, color }) => (
                <div key={step} className="flex items-center gap-3 text-xs">
                  <div className={`w-5 h-5 rounded-full ${color} flex items-center justify-center text-white font-bold text-[10px] flex-shrink-0`}>
                    {step}
                  </div>
                  <span className="text-slate-300">{label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Retrain Results */}
        <div className="lg:col-span-2 space-y-4">
          {retrainResult ? (
            <>
              {/* Comparison Card */}
              <div className={`rounded-xl border p-6 animate-slide-up ${
                retrainResult.improved
                  ? 'bg-emerald-500/10 border-emerald-500/30'
                  : 'bg-amber-500/10 border-amber-500/30'
              }`}>
                <div className="flex items-center gap-3 mb-4">
                  {retrainResult.improved ? (
                    <Trophy className="w-7 h-7 text-emerald-400" />
                  ) : (
                    <AlertTriangle className="w-7 h-7 text-amber-400" />
                  )}
                  <div>
                    <div className={`text-lg font-bold ${retrainResult.improved ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {retrainResult.improved ? '🚀 Model Upgraded!' : '⚠ Current Model Retained'}
                    </div>
                    <div className="text-sm text-slate-400 mt-0.5">{retrainResult.message}</div>
                  </div>
                </div>

                {/* Old vs New comparison */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 rounded-xl p-4">
                    <div className="text-xs text-slate-500 mb-2">Previous Model</div>
                    <div className="text-lg font-bold text-slate-300">{retrainResult.old_model.version}</div>
                    <div className="text-sm text-slate-400">{retrainResult.old_model.model_type}</div>
                    <div className="text-2xl font-bold text-white mt-2">
                      {(retrainResult.old_model.roc_auc * 100).toFixed(2)}%
                    </div>
                    <div className="text-xs text-slate-500">ROC-AUC</div>
                  </div>
                  <div className={`rounded-xl p-4 ${retrainResult.improved ? 'bg-emerald-500/20' : 'bg-slate-800/50'}`}>
                    <div className="text-xs text-slate-500 mb-2">
                      New Model {retrainResult.improved && '(PRODUCTION)'}
                    </div>
                    <div className="text-lg font-bold text-white">{retrainResult.new_model.version}</div>
                    <div className="text-sm text-slate-400">{retrainResult.new_model.model_type}</div>
                    <div className={`text-2xl font-bold mt-2 ${retrainResult.improved ? 'text-emerald-400' : 'text-white'}`}>
                      {(retrainResult.new_model.roc_auc * 100).toFixed(2)}%
                    </div>
                    <div className="text-xs text-slate-500">ROC-AUC</div>
                  </div>
                </div>

                {/* Improvement indicator */}
                <div className={`mt-3 text-center text-sm font-semibold ${retrainResult.improved ? 'text-emerald-400' : 'text-slate-400'}`}>
                  {retrainResult.improvement > 0 ? '↑' : '↓'} Improvement:{' '}
                  {retrainResult.improvement > 0 ? '+' : ''}{(retrainResult.improvement * 100).toFixed(4)}% ROC-AUC
                </div>
              </div>

              {/* New model full metrics */}
              {retrainResult.new_model.metrics && (
                <div className="glass-card rounded-xl p-5 animate-slide-up">
                  <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-violet-400" />
                    New Model Full Metrics
                  </h3>
                  <div className="space-y-3">
                    {[
                      { label: 'ROC-AUC', key: 'roc_auc', color: '#818cf8' },
                      { label: 'Recall', key: 'recall', color: '#34d399' },
                      { label: 'F1 Score', key: 'f1_score', color: '#f59e0b' },
                      { label: 'Precision', key: 'precision', color: '#22d3ee' },
                      { label: 'Accuracy', key: 'accuracy', color: '#f472b6' },
                    ].map(({ label, key, color }) => (
                      <MetricBar key={key} label={label} value={retrainResult.new_model.metrics[key] || 0} color={color} />
                    ))}
                  </div>
                  <div className="mt-3 text-xs text-slate-500">
                    Dataset: {retrainResult.dataset_size?.toLocaleString()} records ·{' '}
                    Fraud: {retrainResult.fraud_count?.toLocaleString()} cases
                  </div>
                </div>
              )}

              {/* Model competition bar chart */}
              {comparisonData.length > 0 && (
                <div className="glass-card rounded-xl p-5 animate-slide-up">
                  <h3 className="text-sm font-semibold text-white mb-4">
                    Model Competition Results
                  </h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={comparisonData} margin={{ left: -10, right: 10, top: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} tickLine={false} />
                      <YAxis domain={[50, 100]} tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} unit="%" />
                      <Tooltip
                        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                        formatter={(v: any) => [`${v.toFixed(2)}%`]}
                      />
                      <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
                      <Bar dataKey="ROC-AUC" fill="#818cf8" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="F1 Score" fill="#22d3ee" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Recall" fill="#34d399" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Feature importance */}
              {featureData.length > 0 && (
                <div className="glass-card rounded-xl p-5 animate-slide-up">
                  <h3 className="text-sm font-semibold text-white mb-4">Top Feature Importance (New Model)</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={featureData} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                      <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false} unit="%" />
                      <YAxis dataKey="feature" type="category" tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={false} width={40} />
                      <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                        formatter={(v: any) => [`${v.toFixed(3)}%`, 'Importance']} />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {featureData.map((_, i) => <Cell key={i} fill={`hsl(${270 - i * 15}, 70%, 65%)`} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          ) : (
            <div className="glass-card rounded-xl p-12 text-center flex flex-col items-center justify-center min-h-80">
              <div className="w-20 h-20 rounded-full bg-violet-500/10 border border-violet-500/20 flex items-center justify-center mb-4">
                <RefreshCw className="w-10 h-10 text-violet-400/50" />
              </div>
              <h3 className="text-lg font-semibold text-slate-300 mb-2">Ready to Retrain</h3>
              <p className="text-slate-500 text-sm max-w-sm">
                Click <strong className="text-slate-300">Start Auto-Retraining</strong> to train new models on the latest data.
                Results and model comparison will appear here.
              </p>
              <div className="mt-6 text-xs text-slate-600 space-y-1">
                <div>→ Trains: Logistic Regression, Random Forest, XGBoost</div>
                <div>→ Compares new vs current production model</div>
                <div>→ Only upgrades if ROC-AUC improves</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Model Registry Table */}
      <div className="glass-card rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-cyan-400" />
          Model Version Registry
          <span className="ml-auto text-xs text-slate-500">{registry.length} versions</span>
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-500 text-left border-b border-slate-700/50">
                <th className="pb-3 pr-4 font-medium">Version</th>
                <th className="pb-3 pr-4 font-medium">Algorithm</th>
                <th className="pb-3 pr-4 font-medium">ROC-AUC</th>
                <th className="pb-3 pr-4 font-medium">F1</th>
                <th className="pb-3 pr-4 font-medium">Recall</th>
                <th className="pb-3 pr-4 font-medium">Dataset</th>
                <th className="pb-3 pr-4 font-medium">Status</th>
                <th className="pb-3 font-medium">Trained</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {registry.length === 0 ? (
                <tr><td colSpan={8} className="text-center py-8 text-slate-500">No models trained yet</td></tr>
              ) : (
                registry.map((model) => (
                  <>
                    <tr
                      key={model.id}
                      className={`hover:bg-slate-800/20 transition-colors cursor-pointer ${model.is_production ? 'bg-violet-500/5' : ''}`}
                      onClick={() => setExpandedRow(expandedRow === model.version ? null : model.version)}
                    >
                      <td className="py-3 pr-4">
                        <span className="font-mono font-bold text-white">{model.version}</span>
                      </td>
                      <td className="py-3 pr-4 text-slate-300">{model.model_type}</td>
                      <td className="py-3 pr-4 font-mono font-semibold text-violet-400">
                        {(model.roc_auc * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 pr-4 font-mono text-cyan-400">
                        {(model.f1_score * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 pr-4 font-mono text-emerald-400">
                        {(model.recall * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 pr-4 text-slate-400 text-xs">
                        {model.dataset_size?.toLocaleString()} rows
                      </td>
                      <td className="py-3 pr-4">
                        {model.is_production ? (
                          <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-violet-500/20 text-violet-300 border border-violet-500/30">
                            PRODUCTION
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded-full text-xs text-slate-500 border border-slate-700">
                            archived
                          </span>
                        )}
                      </td>
                      <td className="py-3 text-slate-500 text-xs">
                        {model.created_at ? new Date(model.created_at).toLocaleDateString() : '—'}
                        {expandedRow === model.version
                          ? <ChevronUp className="w-3 h-3 inline ml-1" />
                          : <ChevronDown className="w-3 h-3 inline ml-1" />}
                      </td>
                    </tr>
                    {expandedRow === model.version && (
                      <tr key={`${model.id}-expanded`} className="bg-slate-800/30">
                        <td colSpan={8} className="px-4 py-4">
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                            {[
                              { label: 'Accuracy', value: `${(model.accuracy * 100).toFixed(2)}%` },
                              { label: 'Precision', value: `${(model.precision * 100).toFixed(2)}%` },
                              { label: 'True Negatives', value: model.confusion_matrix?.tn },
                              { label: 'False Positives', value: model.confusion_matrix?.fp },
                              { label: 'False Negatives', value: model.confusion_matrix?.fn },
                              { label: 'True Positives', value: model.confusion_matrix?.tp },
                              { label: 'Fraud Count', value: model.fraud_count },
                              { label: 'Fraud Rate', value: `${((model.fraud_count / model.dataset_size) * 100).toFixed(2)}%` },
                            ].map(({ label, value }) => (
                              <div key={label}>
                                <div className="text-slate-500">{label}</div>
                                <div className="text-slate-200 font-semibold">{value}</div>
                              </div>
                            ))}
                          </div>
                          {model.comparison_notes && (
                            <div className="mt-3 text-xs text-slate-500 bg-slate-900/50 rounded p-2">
                              {model.comparison_notes.split('. ').slice(0, 2).join('. ')}
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

'use client';

import { useEffect, useState } from 'react';
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Cell
} from 'recharts';
import { BarChart3, TrendingUp, Database, Filter, RefreshCw } from 'lucide-react';
import { getStats, getHistory, getFeatureImportance, getModelRegistry, HistoryItem, ModelRegistryEntry } from '@/lib/api';

export default function AnalyticsPage() {
  const [stats, setStats] = useState<any>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [featureImportance, setFeatureImportance] = useState<any[]>([]);
  const [registry, setRegistry] = useState<ModelRegistryEntry[]>([]);
  const [fraudOnly, setFraudOnly] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [s, h, fi, reg] = await Promise.all([
        getStats(),
        getHistory(100, fraudOnly),
        getFeatureImportance(),
        getModelRegistry()
      ]);
      setStats(s);
      setHistory(h.predictions);
      if (fi.feature_importance) {
        setFeatureImportance(fi.feature_importance.slice(0, 15));
      }
      setRegistry(reg.models);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [fraudOnly]);

  // Amount distribution (buckets)
  const amountBuckets = (() => {
    const buckets = [
      { range: '$0–50', min: 0, max: 50, normal: 0, fraud: 0 },
      { range: '$50–200', min: 50, max: 200, normal: 0, fraud: 0 },
      { range: '$200–500', min: 200, max: 500, normal: 0, fraud: 0 },
      { range: '$500–1k', min: 500, max: 1000, normal: 0, fraud: 0 },
      { range: '$1k–5k', min: 1000, max: 5000, normal: 0, fraud: 0 },
      { range: '>$5k', min: 5000, max: Infinity, normal: 0, fraud: 0 },
    ];
    history.forEach(h => {
      const amt = h.amount ?? 0;
      const bucket = buckets.find(b => amt >= b.min && amt < b.max);
      if (bucket) {
        if (h.is_fraud) bucket.fraud++;
        else bucket.normal++;
      }
    });
    return buckets;
  })();

  // Probability distribution
  const probBuckets = (() => {
    const buckets = Array.from({ length: 10 }, (_, i) => ({
      range: `${i * 10}–${i * 10 + 10}%`, count: 0, fraud: 0, normal: 0
    }));
    history.forEach(h => {
      const idx = Math.min(Math.floor(h.fraud_probability * 10), 9);
      buckets[idx].count++;
      if (h.is_fraud) buckets[idx].fraud++;
      else buckets[idx].normal++;
    });
    return buckets;
  })();

  // Model comparison radar data
  const radarData = registry.length > 0 ? [
    { metric: 'Accuracy', ...Object.fromEntries(registry.map(m => [m.version, m.accuracy * 100])) },
    { metric: 'Precision', ...Object.fromEntries(registry.map(m => [m.version, m.precision * 100])) },
    { metric: 'Recall', ...Object.fromEntries(registry.map(m => [m.version, m.recall * 100])) },
    { metric: 'F1', ...Object.fromEntries(registry.map(m => [m.version, m.f1_score * 100])) },
    { metric: 'ROC-AUC', ...Object.fromEntries(registry.map(m => [m.version, m.roc_auc * 100])) },
  ] : [];

  const COLORS = ['#818cf8', '#22d3ee', '#34d399', '#f59e0b', '#f472b6'];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="w-10 h-10 border-4 border-violet-500/30 border-t-violet-500 rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-1">
            <span className="gradient-text">Analytics</span> Dashboard
          </h1>
          <p className="text-slate-400">Deep insights into fraud patterns and model performance</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setFraudOnly(!fraudOnly)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-all ${
              fraudOnly ? 'bg-red-500/20 border-red-500/30 text-red-400' : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'
            }`}
          >
            <Filter className="w-4 h-4" />
            {fraudOnly ? 'Fraud Only' : 'All Transactions'}
          </button>
          <button onClick={fetchData} className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        {[
          { label: 'Total Analyzed', value: stats?.total_predictions ?? 0, color: 'text-violet-400' },
          { label: 'Fraud Cases', value: stats?.total_fraud ?? 0, color: 'text-red-400' },
          { label: 'Fraud Rate', value: `${stats?.fraud_rate_percent ?? 0}%`, color: 'text-amber-400' },
          { label: 'Avg Probability', value: `${((stats?.avg_fraud_probability ?? 0) * 100).toFixed(1)}%`, color: 'text-cyan-400' },
        ].map(({ label, value, color }) => (
          <div key={label} className="glass-card rounded-xl p-4">
            <div className={`text-2xl font-bold ${color} mb-1`}>{value}</div>
            <div className="text-xs text-slate-400">{label}</div>
          </div>
        ))}
      </div>

      {/* Feature Importance */}
      <div className="glass-card rounded-xl p-6 mb-8">
        <h2 className="text-lg font-semibold text-white mb-1 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-violet-400" />
          Feature Importance
          <span className="text-xs text-slate-500 ml-2">Top 15 features influencing fraud detection</span>
        </h2>
        {featureImportance.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={featureImportance} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis dataKey="feature" type="category" tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={false} width={50} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                formatter={(v: any) => [(v * 100).toFixed(3) + '%', 'Importance']}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {featureImportance.map((_, i) => (
                  <Cell key={i} fill={`hsl(${260 - i * 12}, 70%, 65%)`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-slate-400 text-center py-8">Feature importance not available</div>
        )}
      </div>

      {/* Amount Distribution + Probability Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="glass-card rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Database className="w-5 h-5 text-cyan-400" />
            Transaction Amount Distribution
          </h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={amountBuckets} margin={{ left: -20, right: 10, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="range" tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Bar dataKey="normal" name="Normal" fill="#818cf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="fraud" name="Fraud" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-card rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" />
            Fraud Probability Distribution
          </h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={probBuckets} margin={{ left: -20, right: 10, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="range" tick={{ fill: '#64748b', fontSize: 9 }} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Bar dataKey="normal" name="Normal" fill="#22d3ee" radius={[4, 4, 0, 0]} />
              <Bar dataKey="fraud" name="Fraud" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Comparison Radar */}
      {registry.length > 1 && (
        <div className="glass-card rounded-xl p-6 mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-amber-400" />
            Model Version Comparison (Radar)
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#1e293b" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <PolarRadiusAxis angle={30} domain={[80, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
              {registry.slice(0, 3).map((m, i) => (
                <Radar key={m.version} name={`${m.version} (${m.model_type})`}
                  dataKey={m.version} stroke={COLORS[i]} fill={COLORS[i]} fillOpacity={0.15} strokeWidth={2}
                />
              ))}
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                formatter={(v: any) => [`${v.toFixed(2)}%`]} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Daily Trend */}
      {stats?.daily_stats && (
        <div className="glass-card rounded-xl p-6 mb-8">
          <h2 className="text-lg font-semibold text-white mb-4">Daily Transaction Trend</h2>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={stats.daily_stats} margin={{ left: -20, right: 10, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Line type="monotone" dataKey="total" stroke="#818cf8" strokeWidth={2} dot={{ fill: '#818cf8', r: 3 }} name="Total" />
              <Line type="monotone" dataKey="fraud" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444', r: 3 }} name="Fraud" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Transaction History Table */}
      <div className="glass-card rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-slate-400" />
          Transaction History ({history.length} records)
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-500 text-left border-b border-slate-700/50">
                <th className="pb-3 pr-4 font-medium">#</th>
                <th className="pb-3 pr-4 font-medium">Result</th>
                <th className="pb-3 pr-4 font-medium">Amount</th>
                <th className="pb-3 pr-4 font-medium">Risk</th>
                <th className="pb-3 pr-4 font-medium">Probability</th>
                <th className="pb-3 pr-4 font-medium">Model</th>
                <th className="pb-3 font-medium">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {history.length === 0 ? (
                <tr><td colSpan={7} className="text-center py-8 text-slate-500">No predictions yet</td></tr>
              ) : (
                history.map((item, idx) => (
                  <tr key={item.id} className={`hover:bg-slate-800/20 transition-colors ${item.is_fraud ? 'bg-red-500/3' : ''}`}>
                    <td className="py-2.5 pr-4 text-slate-600 text-xs">{idx + 1}</td>
                    <td className="py-2.5 pr-4">
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                        item.is_fraud ? 'bg-red-500/15 text-red-400' : 'bg-emerald-500/15 text-emerald-400'
                      }`}>
                        {item.is_fraud ? '⚠ FRAUD' : '✓ SAFE'}
                      </span>
                    </td>
                    <td className="py-2.5 pr-4 text-slate-300">${item.amount?.toFixed(2) ?? '—'}</td>
                    <td className="py-2.5 pr-4">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        item.risk_level === 'LOW' ? 'risk-low' : item.risk_level === 'MEDIUM' ? 'risk-medium' :
                        item.risk_level === 'HIGH' ? 'risk-high' : 'risk-critical'
                      }`}>{item.risk_level}</span>
                    </td>
                    <td className="py-2.5 pr-4">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-amber-500 to-red-500"
                            style={{ width: `${item.fraud_probability * 100}%` }} />
                        </div>
                        <span className="text-slate-400 text-xs font-mono">
                          {(item.fraud_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 pr-4 text-slate-500 text-xs">{item.model_version}</td>
                    <td className="py-2.5 text-slate-500 text-xs">
                      {item.created_at ? new Date(item.created_at).toLocaleTimeString() : '—'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

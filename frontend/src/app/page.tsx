'use client';

import { useEffect, useState } from 'react';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import {
  Shield, AlertTriangle, CheckCircle, Activity, TrendingUp,
  Clock, Cpu, Database, RefreshCw, ChevronRight
} from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { getMetrics, getStats, getHistory, MetricsData, StatsData, HistoryItem } from '@/lib/api';

const RISK_COLORS: Record<string, string> = {
  LOW: '#10b981', MEDIUM: '#f59e0b', HIGH: '#f97316', CRITICAL: '#ef4444'
};

function StatCard({
  title, value, subtitle, icon: Icon, color, trend
}: {
  title: string; value: string | number; subtitle?: string;
  icon: React.ElementType; color: string; trend?: string;
}) {
  const colorMap: Record<string, string> = {
    violet: 'from-violet-500/20 to-violet-600/5 border-violet-500/30 text-violet-400',
    cyan: 'from-cyan-500/20 to-cyan-600/5 border-cyan-500/30 text-cyan-400',
    emerald: 'from-emerald-500/20 to-emerald-600/5 border-emerald-500/30 text-emerald-400',
    red: 'from-red-500/20 to-red-600/5 border-red-500/30 text-red-400',
    amber: 'from-amber-500/20 to-amber-600/5 border-amber-500/30 text-amber-400',
  };
  const cls = colorMap[color] || colorMap.violet;

  return (
    <div className={`stat-card bg-gradient-to-br ${cls} rounded-xl border p-5 backdrop-blur-sm`}>
      <div className="flex items-start justify-between mb-3">
        <div className={`p-2 rounded-lg bg-current/10`}>
          <Icon className="w-5 h-5" />
        </div>
        {trend && <span className="text-xs text-slate-400">{trend}</span>}
      </div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      <div className="text-sm font-medium text-slate-300">{title}</div>
      {subtitle && <div className="text-xs text-slate-500 mt-0.5">{subtitle}</div>}
    </div>
  );
}

function RiskBadge({ level }: { level: string }) {
  const cls: Record<string, string> = {
    LOW: 'risk-low', MEDIUM: 'risk-medium', HIGH: 'risk-high', CRITICAL: 'risk-critical'
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${cls[level] || 'risk-low'}`}>
      {level}
    </span>
  );
}

export default function Dashboard() {
  const router = useRouter();
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(new Date());

  const fetchData = async () => {
    try {
      const [m, s, h] = await Promise.all([getMetrics(), getStats(), getHistory(10)]);
      setMetrics(m);
      setStats(s);
      setHistory(h.predictions);
      setLastRefresh(new Date());
    } catch (e) {
      console.error('Failed to fetch dashboard data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const riskData = stats ? Object.entries(stats.risk_distribution).map(([name, value]) => ({
    name, value, fill: RISK_COLORS[name] || '#64748b'
  })) : [];

  const dailyData = stats?.daily_stats || [];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-violet-500/30 border-t-violet-500 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-400">Initializing MLOps System...</p>
          <p className="text-slate-500 text-sm mt-1">Training model on first run (may take ~30s)</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-1">
            Fraud Detection <span className="gradient-text">Dashboard</span>
          </h1>
          <p className="text-slate-400 text-sm">
            Real-time monitoring · Auto-retraining MLOps Pipeline
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* ── Demo Button ── */}
          <button
            onClick={() => router.push('/demo')}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-violet-500 to-cyan-500 hover:from-violet-600 hover:to-cyan-600 text-white font-bold text-sm shadow-lg shadow-violet-500/30 transition-all hover:scale-105 animate-glow-pulse"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Run Full Demo
          </button>
          <span className="text-xs text-slate-500">
            {lastRefresh.toLocaleTimeString()}
          </span>
          <button
            onClick={fetchData}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          title="Total Predictions"
          value={stats?.total_predictions.toLocaleString() ?? '0'}
          subtitle="All time transactions analyzed"
          icon={Activity}
          color="violet"
        />
        <StatCard
          title="Fraud Detected"
          value={stats?.total_fraud.toLocaleString() ?? '0'}
          subtitle={`${stats?.fraud_rate_percent ?? 0}% fraud rate`}
          icon={AlertTriangle}
          color="red"
        />
        <StatCard
          title="Model ROC-AUC"
          value={metrics?.metrics?.roc_auc ? `${(metrics.metrics.roc_auc * 100).toFixed(1)}%` : 'N/A'}
          subtitle={`${metrics?.model_type ?? 'No model'} · ${metrics?.version ?? ''}`}
          icon={TrendingUp}
          color="cyan"
        />
        <StatCard
          title="Avg Response"
          value={stats?.avg_processing_time_ms ? `${stats.avg_processing_time_ms.toFixed(1)}ms` : '—'}
          subtitle="Average prediction time"
          icon={Clock}
          color="emerald"
        />
      </div>

      {/* Model Performance + Risk Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">

        {/* Model Metrics */}
        <div className="lg:col-span-2 glass-card rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-violet-400" />
            Model Performance Metrics
            {metrics && (
              <span className="ml-auto text-xs px-2 py-1 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">
                {metrics.version} · {metrics.model_type}
              </span>
            )}
          </h2>
          {metrics?.metrics ? (
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
              {[
                { label: 'Accuracy', value: metrics.metrics.accuracy, color: '#818cf8' },
                { label: 'Precision', value: metrics.metrics.precision, color: '#22d3ee' },
                { label: 'Recall', value: metrics.metrics.recall, color: '#34d399' },
                { label: 'F1 Score', value: metrics.metrics.f1_score, color: '#f59e0b' },
                { label: 'ROC-AUC', value: metrics.metrics.roc_auc, color: '#f472b6' },
              ].map(({ label, value, color }) => (
                <div key={label} className="text-center">
                  <div className="relative w-20 h-20 mx-auto mb-2">
                    <svg className="w-20 h-20 -rotate-90" viewBox="0 0 36 36">
                      <circle cx="18" cy="18" r="15.9155" fill="none" stroke="#1e293b" strokeWidth="2.5" />
                      <circle
                        cx="18" cy="18" r="15.9155" fill="none"
                        stroke={color} strokeWidth="2.5"
                        strokeDasharray={`${(value * 100).toFixed(1)} 100`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <span className="absolute inset-0 flex items-center justify-center text-sm font-bold text-white">
                      {(value * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-xs text-slate-400">{label}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-slate-400 text-center py-8">No model metrics available</div>
          )}

          {/* Confusion Matrix */}
          {metrics?.metrics && (
            <div className="mt-6 grid grid-cols-2 gap-2 max-w-xs mx-auto">
              <div className="text-center text-xs text-slate-400 col-span-2 mb-1">Confusion Matrix</div>
              {[
                { label: 'True Neg', value: metrics.metrics.true_negatives, color: 'text-emerald-400' },
                { label: 'False Pos', value: metrics.metrics.false_positives, color: 'text-red-400' },
                { label: 'False Neg', value: metrics.metrics.false_negatives, color: 'text-amber-400' },
                { label: 'True Pos', value: metrics.metrics.true_positives, color: 'text-cyan-400' },
              ].map(({ label, value, color }) => (
                <div key={label} className="bg-slate-800/60 rounded-lg p-2 text-center">
                  <div className={`text-lg font-bold ${color}`}>{value}</div>
                  <div className="text-[10px] text-slate-500">{label}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Risk Distribution */}
        <div className="glass-card rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Shield className="w-5 h-5 text-cyan-400" />
            Risk Distribution
          </h2>
          {riskData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={riskData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" paddingAngle={3}>
                    {riskData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2 mt-2">
                {riskData.map(({ name, value, fill }) => (
                  <div key={name} className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ background: fill }} />
                      <span className="text-slate-300">{name}</span>
                    </div>
                    <span className="text-white font-semibold">{value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="text-slate-400 text-center py-12">
              <Shield className="w-10 h-10 mx-auto mb-2 opacity-30" />
              No predictions yet
            </div>
          )}
        </div>
      </div>

      {/* Daily Trend Chart */}
      <div className="glass-card rounded-xl p-6 mb-8">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-emerald-400" />
          Transaction Trend (Last 7 Days)
        </h2>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={dailyData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <defs>
              <linearGradient id="normalGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="fraudGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
              labelStyle={{ color: '#94a3b8' }}
            />
            <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
            <Area type="monotone" dataKey="normal" stroke="#818cf8" fill="url(#normalGrad)" name="Normal" strokeWidth={2} />
            <Area type="monotone" dataKey="fraud" stroke="#ef4444" fill="url(#fraudGrad)" name="Fraud" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Predictions Table */}
      <div className="glass-card rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-amber-400" />
            Recent Predictions
          </h2>
          <Link href="/analytics" className="text-sm text-violet-400 hover:text-violet-300 flex items-center gap-1">
            View all <ChevronRight className="w-3 h-3" />
          </Link>
        </div>
        {history.length === 0 ? (
          <div className="text-center py-12 text-slate-400">
            <Activity className="w-10 h-10 mx-auto mb-2 opacity-30" />
            <p>No predictions yet. Try the <Link href="/predict" className="text-violet-400 hover:underline">Live Predict</Link> page.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-500 text-left border-b border-slate-700/50">
                  <th className="pb-3 pr-4 font-medium">Transaction ID</th>
                  <th className="pb-3 pr-4 font-medium">Amount</th>
                  <th className="pb-3 pr-4 font-medium">Risk</th>
                  <th className="pb-3 pr-4 font-medium">Probability</th>
                  <th className="pb-3 pr-4 font-medium">Model</th>
                  <th className="pb-3 font-medium">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {history.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 pr-4">
                      <div className="flex items-center gap-2">
                        {item.is_fraud ? (
                          <AlertTriangle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                        ) : (
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                        )}
                        <span className="font-mono text-xs text-slate-300">
                          {item.transaction_id.slice(0, 8)}...
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 pr-4 text-slate-300">
                      ${item.amount?.toFixed(2) ?? '—'}
                    </td>
                    <td className="py-2.5 pr-4">
                      <RiskBadge level={item.risk_level} />
                    </td>
                    <td className="py-2.5 pr-4">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-slate-700 rounded-full max-w-16">
                          <div
                            className="h-1.5 rounded-full"
                            style={{
                              width: `${item.fraud_probability * 100}%`,
                              background: `hsl(${120 - item.fraud_probability * 120}, 80%, 50%)`
                            }}
                          />
                        </div>
                        <span className="text-slate-300 font-mono text-xs">
                          {(item.fraud_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 pr-4 text-slate-400 text-xs">{item.model_version}</td>
                    <td className="py-2.5 text-slate-500 text-xs">
                      {item.created_at ? new Date(item.created_at).toLocaleTimeString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
        {/* Demo - spans 2 cols, most prominent */}
        <Link
          href="/demo"
          className="col-span-2 bg-gradient-to-br from-violet-500 via-purple-600 to-cyan-500 rounded-xl p-5 group hover:shadow-2xl hover:shadow-violet-500/30 transition-all duration-200 hover:-translate-y-1 relative overflow-hidden"
        >
          <div className="absolute inset-0 opacity-20" style={{ background: 'radial-gradient(circle at 80% 50%, #22d3ee, transparent 60%)' }} />
          <div className="relative">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              <span className="text-xs font-medium text-white/70 uppercase tracking-wider">One-Click</span>
            </div>
            <div className="font-black text-lg text-white">Run Full Demo</div>
            <div className="text-xs text-white/70">Watch the entire MLOps pipeline live · 10 steps · animated</div>
          </div>
        </Link>
        {[
          { href: '/predict', label: 'Live Predict', desc: 'Test a transaction', color: 'from-indigo-500 to-indigo-600', icon: Zap },
          { href: '/retrain', label: 'Retrain', desc: 'Upgrade the model', color: 'from-emerald-500 to-emerald-600', icon: RefreshCw },
        ].map(({ href, label, desc, color, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={`bg-gradient-to-br ${color} rounded-xl p-4 group hover:shadow-lg transition-all duration-200 hover:-translate-y-0.5`}
          >
            <Icon className="w-5 h-5 text-white/80 mb-2" />
            <div className="font-semibold text-white">{label}</div>
            <div className="text-xs text-white/70">{desc}</div>
          </Link>
        ))}
      </div>
    </div>
  );
}

function Zap({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function BarChart3({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  );
}

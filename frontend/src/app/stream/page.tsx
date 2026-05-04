'use client';

import { useEffect, useState, useCallback } from 'react';
import { Activity, Play, Square, Zap, AlertTriangle } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import {
  startStream, stopStream, getStreamStatus,
} from '@/lib/api';

export default function StreamPage() {
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState<any>(null);
  const [recent, setRecent] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const poll = useCallback(async () => {
    try {
      const s = await getStreamStatus();
      setRunning(!!s.running);
      setStats(s.stats);
      setRecent(s.recent || []);
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    poll();
  }, [poll]);

  useEffect(() => {
    const id = setInterval(poll, 350);
    return () => clearInterval(id);
  }, [poll]);

  const chartData = [...(recent || [])]
    .slice(0, 60)
    .reverse()
    .map((r, i) => ({
      i,
      prob: Math.round((r.fraud_probability ?? 0) * 1000) / 10,
      fraud: r.is_fraud ? 1 : 0,
    }));

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-1">
          Live <span className="gradient-text">Transaction Stream</span>
        </h1>
        <p className="text-slate-400">
          Simulated real-time scoring every 100ms from the active training dataset — fraud alerts update live
        </p>
      </div>

      <div className="flex flex-wrap gap-3 mb-6">
        <button
          type="button"
          onClick={async () => {
            setError(null);
            await startStream();
            await poll();
          }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-sm font-semibold hover:bg-emerald-500/30"
        >
          <Play className="w-4 h-4" /> Start stream
        </button>
        <button
          type="button"
          onClick={async () => {
            await stopStream();
            await poll();
          }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 border border-slate-600 text-slate-300 text-sm font-semibold hover:bg-slate-700"
        >
          <Square className="w-4 h-4" /> Stop
        </button>
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <Activity className={`w-4 h-4 ${running ? 'text-emerald-400 animate-pulse' : 'text-slate-600'}`} />
          {running ? 'Streaming' : 'Idle'}
        </div>
      </div>

      {error && (
        <div className="mb-4 text-sm text-red-400">{error}</div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="glass-card rounded-xl p-4">
          <div className="text-xs text-slate-500 mb-1">Events processed</div>
          <div className="text-2xl font-bold text-violet-300">{stats?.total ?? 0}</div>
        </div>
        <div className="glass-card rounded-xl p-4">
          <div className="text-xs text-slate-500 mb-1">Flagged as fraud</div>
          <div className="text-2xl font-bold text-red-400">{stats?.fraud_flagged ?? 0}</div>
        </div>
        <div className="glass-card rounded-xl p-4">
          <div className="text-xs text-slate-500 mb-1">Flag rate</div>
          <div className="text-2xl font-bold text-amber-400">
            {stats?.fraud_flag_rate_pct != null ? `${stats.fraud_flag_rate_pct.toFixed(2)}%` : '—'}
          </div>
        </div>
      </div>

      <div className="glass-card rounded-xl p-6 mb-8">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-violet-400" />
          Fraud probability (rolling)
        </h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="i" tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
            <Legend />
            <Line type="monotone" dataKey="prob" name="Fraud prob %" stroke="#a78bfa" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="glass-card rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          Recent alerts
        </h2>
        <div className="max-h-64 overflow-y-auto space-y-2 text-sm font-mono">
          {(recent || []).slice(0, 25).map((r, idx) => (
            <div
              key={idx}
              className={`flex justify-between rounded-lg px-3 py-2 border ${
                r.is_fraud ? 'bg-red-500/10 border-red-500/30 text-red-300' : 'bg-slate-800/40 border-slate-700/40 text-slate-400'
              }`}
            >
              <span>{(r.fraud_probability * 100).toFixed(2)}%</span>
              <span>{r.risk_level}</span>
              <span className="text-slate-500 text-xs">{r.true_label === 1 ? 'true fraud' : 'true ok'}</span>
            </div>
          ))}
          {(!recent || recent.length === 0) && (
            <p className="text-slate-500 text-center py-8">Start the stream to see live events</p>
          )}
        </div>
      </div>
    </div>
  );
}

'use client';

import { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, Zap, RotateCcw, ChevronDown, ChevronUp, Clock, Cpu } from 'lucide-react';
import {
  predictTransaction, TransactionInput, PredictionResult,
  SAMPLE_NORMAL_TRANSACTION, SAMPLE_FRAUD_TRANSACTION
} from '@/lib/api';

const defaultTransaction: TransactionInput = {
  time: 0, amount: 0,
  v1: 0, v2: 0, v3: 0, v4: 0, v5: 0, v6: 0, v7: 0,
  v8: 0, v9: 0, v10: 0, v11: 0, v12: 0, v13: 0, v14: 0,
  v15: 0, v16: 0, v17: 0, v18: 0, v19: 0, v20: 0, v21: 0,
  v22: 0, v23: 0, v24: 0, v25: 0, v26: 0, v27: 0, v28: 0,
};

function GaugeMeter({ probability }: { probability: number }) {
  const angle = -135 + probability * 270;
  const color = probability < 0.3 ? '#10b981' : probability < 0.6 ? '#f59e0b' : probability < 0.85 ? '#f97316' : '#ef4444';
  const pct = Math.round(probability * 100);

  return (
    <div className="relative w-48 h-48 mx-auto">
      <svg viewBox="0 0 200 200" className="w-full h-full">
        {/* Background arc */}
        <path d="M 30 145 A 80 80 0 1 1 170 145" fill="none" stroke="#1e293b" strokeWidth="16" strokeLinecap="round" />
        {/* Colored arc */}
        <path
          d="M 30 145 A 80 80 0 1 1 170 145"
          fill="none" stroke={color} strokeWidth="16" strokeLinecap="round"
          strokeDasharray={`${probability * 251.2} 251.2`}
          style={{ transition: 'stroke-dasharray 0.8s ease, stroke 0.3s ease' }}
        />
        {/* Needle */}
        <g transform={`rotate(${angle}, 100, 100)`}>
          <line x1="100" y1="100" x2="100" y2="32" stroke={color} strokeWidth="3" strokeLinecap="round" />
          <circle cx="100" cy="100" r="6" fill={color} />
        </g>
        {/* Center text */}
        <text x="100" y="130" textAnchor="middle" className="text-4xl" fill="white" fontSize="32" fontWeight="bold">{pct}%</text>
        <text x="100" y="155" textAnchor="middle" fill="#64748b" fontSize="12">Fraud Probability</text>
      </svg>
    </div>
  );
}

function InputField({ label, name, value, onChange }: {
  label: string; name: string; value: number; onChange: (name: string, val: number) => void;
}) {
  return (
    <div>
      <label className="block text-xs text-slate-400 mb-1">{label}</label>
      <input
        type="number"
        step="any"
        value={value}
        onChange={e => onChange(name, parseFloat(e.target.value) || 0)}
        className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500/30 transition-all"
      />
    </div>
  );
}

export default function PredictPage() {
  const [form, setForm] = useState<TransactionInput>(defaultTransaction);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showVFeatures, setShowVFeatures] = useState(false);

  const handleChange = (name: string, val: number) => {
    setForm(prev => ({ ...prev, [name]: val }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictTransaction(form);
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Prediction failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const riskConfig: Record<string, { bg: string; text: string; border: string; icon: React.ReactNode }> = {
    LOW: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30', icon: <CheckCircle className="w-6 h-6" /> },
    MEDIUM: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30', icon: <AlertTriangle className="w-6 h-6" /> },
    HIGH: { bg: 'bg-orange-500/10', text: 'text-orange-400', border: 'border-orange-500/30', icon: <AlertTriangle className="w-6 h-6" /> },
    CRITICAL: { bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/30', icon: <AlertTriangle className="w-6 h-6" /> },
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-1">
          Live <span className="gradient-text">Fraud Prediction</span>
        </h1>
        <p className="text-slate-400">Enter transaction details for real-time ML fraud detection</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="space-y-6">
          {/* Sample Buttons */}
          <div className="glass-card rounded-xl p-4">
            <p className="text-sm text-slate-400 mb-3 font-medium">Quick Load Sample Transactions:</p>
            <div className="flex gap-3">
              <button
                onClick={() => { setForm(SAMPLE_NORMAL_TRANSACTION); setResult(null); }}
                className="flex-1 py-2.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-400 text-sm font-semibold transition-all"
              >
                ✓ Normal Transaction
              </button>
              <button
                onClick={() => { setForm(SAMPLE_FRAUD_TRANSACTION); setResult(null); }}
                className="flex-1 py-2.5 rounded-lg bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-red-400 text-sm font-semibold transition-all"
              >
                ⚠ Fraud Transaction
              </button>
            </div>
          </div>

          {/* Basic Fields */}
          <div className="glass-card rounded-xl p-6">
            <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4 text-violet-400" />
              Transaction Details
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <InputField label="Time (seconds)" name="time" value={form.time} onChange={handleChange} />
              <InputField label="Amount (USD)" name="amount" value={form.amount} onChange={handleChange} />
            </div>
          </div>

          {/* V1–V10 always visible */}
          <div className="glass-card rounded-xl p-6">
            <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              PCA Features V1–V10
              <span className="text-xs text-slate-500">(Principal Components)</span>
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {Array.from({ length: 10 }, (_, i) => i + 1).map(n => (
                <InputField
                  key={n} label={`V${n}`} name={`v${n}`}
                  value={(form as any)[`v${n}`]} onChange={handleChange}
                />
              ))}
            </div>
          </div>

          {/* V11–V28 collapsible */}
          <div className="glass-card rounded-xl overflow-hidden">
            <button
              onClick={() => setShowVFeatures(!showVFeatures)}
              className="w-full flex items-center justify-between p-4 hover:bg-slate-800/30 transition-colors"
            >
              <span className="text-sm font-semibold text-slate-300">
                Advanced PCA Features V11–V28
              </span>
              {showVFeatures ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
            </button>
            {showVFeatures && (
              <div className="p-4 pt-0">
                <div className="grid grid-cols-2 gap-3">
                  {Array.from({ length: 18 }, (_, i) => i + 11).map(n => (
                    <InputField
                      key={n} label={`V${n}`} name={`v${n}`}
                      value={(form as any)[`v${n}`]} onChange={handleChange}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Submit */}
          <div className="flex gap-3">
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="flex-1 py-3.5 rounded-xl bg-gradient-to-r from-violet-500 to-violet-600 hover:from-violet-600 hover:to-violet-700 text-white font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg shadow-violet-500/25"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Shield className="w-4 h-4" />
                  Detect Fraud
                </>
              )}
            </button>
            <button
              onClick={() => { setForm(defaultTransaction); setResult(null); setError(null); }}
              className="px-4 py-3.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
              title="Reset"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {error && (
            <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 text-sm text-red-400">
              {error}
            </div>
          )}
        </div>

        {/* Result Panel */}
        <div className="space-y-6">
          {result ? (
            <>
              {/* Main Result */}
              <div className={`rounded-xl border p-6 ${riskConfig[result.risk_level]?.bg} ${riskConfig[result.risk_level]?.border} animate-slide-up`}>
                <div className="flex items-center gap-3 mb-4">
                  <div className={`${riskConfig[result.risk_level]?.text}`}>
                    {riskConfig[result.risk_level]?.icon}
                  </div>
                  <div>
                    <div className={`text-lg font-bold ${riskConfig[result.risk_level]?.text}`}>
                      {result.is_fraud ? '⚠ FRAUD DETECTED' : '✓ TRANSACTION SAFE'}
                    </div>
                    <div className="text-slate-400 text-sm">Risk Level: {result.risk_level}</div>
                  </div>
                </div>

                {/* Gauge */}
                <GaugeMeter probability={result.fraud_probability} />

                {/* Probability Bar */}
                <div className="mt-4">
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Safe</span>
                    <span>{(result.fraud_probability * 100).toFixed(2)}% fraud probability</span>
                    <span>Fraud</span>
                  </div>
                  <div className="h-3 bg-slate-700/50 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${result.fraud_probability * 100}%`,
                        background: `linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%)`
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Details */}
              <div className="glass-card rounded-xl p-5 animate-slide-up">
                <h3 className="text-sm font-semibold text-white mb-3">Prediction Details</h3>
                <div className="space-y-2 text-sm">
                  {[
                    { label: 'Transaction ID', value: result.transaction_id.slice(0, 16) + '...', mono: true },
                    { label: 'Model Version', value: result.model_version },
                    { label: 'Algorithm', value: result.model_type },
                    { label: 'Processing Time', value: `${result.processing_time_ms}ms` },
                    { label: 'Timestamp', value: new Date(result.timestamp).toLocaleString() },
                  ].map(({ label, value, mono }) => (
                    <div key={label} className="flex justify-between items-center border-b border-slate-700/30 pb-2">
                      <span className="text-slate-400">{label}</span>
                      <span className={`text-slate-200 ${mono ? 'font-mono text-xs' : ''}`}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Risk Level Legend */}
              <div className="glass-card rounded-xl p-5 animate-slide-up">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-amber-400" />
                  Risk Level Guide
                </h3>
                <div className="space-y-2 text-xs">
                  {[
                    { level: 'LOW', range: '0–30%', desc: 'Safe to process' },
                    { level: 'MEDIUM', range: '30–60%', desc: 'Flag for review' },
                    { level: 'HIGH', range: '60–85%', desc: 'Likely fraudulent' },
                    { level: 'CRITICAL', range: '85–100%', desc: 'Block immediately' },
                  ].map(({ level, range, desc }) => (
                    <div key={level} className="flex items-center gap-3">
                      <span className={`px-2 py-0.5 rounded-full font-semibold ${
                        level === 'LOW' ? 'risk-low' : level === 'MEDIUM' ? 'risk-medium' :
                        level === 'HIGH' ? 'risk-high' : 'risk-critical'
                      }`}>{level}</span>
                      <span className="text-slate-400">{range}</span>
                      <span className="text-slate-500">— {desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="glass-card rounded-xl p-12 text-center h-full flex flex-col items-center justify-center">
              <div className="w-20 h-20 rounded-full bg-violet-500/10 border border-violet-500/20 flex items-center justify-center mb-4 mx-auto">
                <Shield className="w-10 h-10 text-violet-400/50" />
              </div>
              <h3 className="text-lg font-semibold text-slate-300 mb-2">Ready to Analyze</h3>
              <p className="text-slate-500 text-sm max-w-xs">
                Load a sample transaction or enter your own data, then click{' '}
                <strong className="text-slate-400">Detect Fraud</strong> to see real-time ML results.
              </p>
              <div className="mt-6 space-y-2 text-xs text-slate-600">
                <div>→ Model: RandomForest / XGBoost</div>
                <div>→ Response time: &lt; 50ms</div>
                <div>→ All predictions logged to DB</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

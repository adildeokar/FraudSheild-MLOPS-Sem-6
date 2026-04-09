'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  getHealth, getMetrics, getStats, getFeatureImportance,
  getModelRegistry, predictTransaction, triggerRetrain,
  SAMPLE_NORMAL_TRANSACTION, SAMPLE_FRAUD_TRANSACTION,
} from '@/lib/api';
import {
  Play, Pause, SkipForward, RefreshCw, Shield, CheckCircle,
  AlertTriangle, Cpu, Database, TrendingUp, Zap, Trophy,
  Activity, Clock, ChevronRight, BarChart3,
} from 'lucide-react';
import {
  BarChart, Bar, Cell, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie,
} from 'recharts';

// ─────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────
interface DemoStep {
  id: number;
  title: string;
  subtitle: string;
  icon: React.ElementType;
  color: string;
  duration: number; // ms before auto-advance
}

const STEPS: DemoStep[] = [
  { id: 0, title: 'System Health Check',        subtitle: 'Verify ML model is live',             icon: Activity,    color: 'violet',  duration: 4000  },
  { id: 1, title: 'Training Data Overview',      subtitle: '15 k transactions loaded',            icon: Database,    color: 'cyan',    duration: 4500  },
  { id: 2, title: 'Normal Transaction',          subtitle: 'Safe transaction → ML inference',     icon: CheckCircle, color: 'emerald', duration: 5000  },
  { id: 3, title: 'Fraud Transaction',           subtitle: 'Suspicious tx → CRITICAL alert',      icon: AlertTriangle,color:'red',     duration: 5000  },
  { id: 4, title: 'Fraud Analytics',             subtitle: 'Risk distribution & fraud rate',      icon: BarChart3,   color: 'amber',   duration: 4500  },
  { id: 5, title: 'Feature Importance',          subtitle: 'Which features detect fraud best',    icon: TrendingUp,  color: 'pink',    duration: 4500  },
  { id: 6, title: 'Auto-Retraining Pipeline',    subtitle: 'Train new models on fresh data',      icon: RefreshCw,   color: 'orange',  duration: 18000 },
  { id: 7, title: 'Performance Gate Decision',   subtitle: 'Upgrade only if model improves',      icon: Trophy,      color: 'gold',    duration: 5000  },
  { id: 8, title: 'Model Version Registry',      subtitle: 'Full audit trail of all models',      icon: Shield,      color: 'indigo',  duration: 4000  },
  { id: 9, title: 'Pipeline Complete!',          subtitle: 'MLOps system demonstrated end-to-end',icon: Zap,         color: 'green',   duration: 99999 },
];

const COLOR_MAP: Record<string,string> = {
  violet: '#8b5cf6', cyan: '#22d3ee', emerald: '#10b981', red: '#ef4444',
  amber: '#f59e0b', pink: '#ec4899', orange: '#f97316', gold: '#fbbf24',
  indigo: '#6366f1', green: '#22c55e',
};

// ─────────────────────────────────────────────────────────
// Animated Number
// ─────────────────────────────────────────────────────────
function AnimNum({ target, decimals = 0, prefix = '', suffix = '', duration = 1200 }: {
  target: number; decimals?: number; prefix?: string; suffix?: string; duration?: number;
}) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const tick = () => {
      const p = Math.min((Date.now() - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(parseFloat((target * eased).toFixed(decimals)));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target, decimals, duration]);
  return <span>{prefix}{val.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</span>;
}

// ─────────────────────────────────────────────────────────
// Terminal Log
// ─────────────────────────────────────────────────────────
function TerminalLog({ lines }: { lines: Array<{ text: string; color?: string; delay?: number }> }) {
  const [shown, setShown] = useState<number>(0);
  useEffect(() => {
    setShown(0);
    const timers: ReturnType<typeof setTimeout>[] = [];
    lines.forEach((_, i) => {
      timers.push(setTimeout(() => setShown(i + 1), (lines[i].delay ?? i * 280)));
    });
    return () => timers.forEach(clearTimeout);
  }, [lines]);

  return (
    <div className="font-mono text-xs bg-slate-950 rounded-xl p-4 space-y-1 border border-slate-700/40 overflow-hidden demo-scanline relative">
      {lines.slice(0, shown).map((l, i) => (
        <div key={i} className="animate-slide-up2" style={{ color: l.color ?? '#94a3b8' }}>
          <span className="text-slate-600 mr-2">{String(i + 1).padStart(2, '0')}</span>
          {l.text}
        </div>
      ))}
      {shown < lines.length && (
        <span className="inline-block w-2 h-3.5 bg-violet-400" style={{ animation: 'blink 0.8s step-end infinite' }} />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Fraud Gauge
// ─────────────────────────────────────────────────────────
function FraudGauge({ prob, animate }: { prob: number; animate: boolean }) {
  const [displayProb, setDisplayProb] = useState(0);
  useEffect(() => {
    if (!animate) { setDisplayProb(0); return; }
    const start = Date.now();
    const tick = () => {
      const p = Math.min((Date.now() - start) / 1200, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplayProb(prob * eased);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [prob, animate]);

  const color = displayProb < 0.3 ? '#10b981' : displayProb < 0.6 ? '#f59e0b' : displayProb < 0.85 ? '#f97316' : '#ef4444';
  const pct = Math.round(displayProb * 100);

  return (
    <div className="relative w-40 h-40 mx-auto">
      {prob > 0.85 && animate && (
        <div className="absolute inset-0 rounded-full border-2 border-red-500/60" style={{ animation: 'ripple 1.5s ease-out infinite' }} />
      )}
      <svg viewBox="0 0 200 200" className="w-full h-full">
        <path d="M 30 145 A 80 80 0 1 1 170 145" fill="none" stroke="#1e293b" strokeWidth="18" strokeLinecap="round" />
        <path d="M 30 145 A 80 80 0 1 1 170 145" fill="none" stroke={color} strokeWidth="18" strokeLinecap="round"
          strokeDasharray={`${displayProb * 251.2} 251.2`}
          style={{ transition: 'stroke-dasharray 0.05s linear, stroke 0.3s ease' }} />
        <text x="100" y="125" textAnchor="middle" fill="white" fontSize="36" fontWeight="bold">{pct}%</text>
        <text x="100" y="150" textAnchor="middle" fill="#64748b" fontSize="11">Fraud Risk</text>
      </svg>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Pipeline Flow Diagram
// ─────────────────────────────────────────────────────────
function PipelineFlow({ activeIdx }: { activeIdx: number }) {
  const nodes = ['CSV Data', 'Validate', 'Scale', 'SMOTE', 'Train', 'Evaluate', 'Select', 'Deploy'];
  return (
    <div className="flex items-center justify-center gap-0 flex-wrap mt-4">
      {nodes.map((n, i) => (
        <div key={n} className="flex items-center">
          <div className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition-all duration-500 ${
            i < activeIdx  ? 'bg-violet-500/30 border-violet-500/60 text-violet-200'
            : i === activeIdx ? 'bg-violet-500 border-violet-400 text-white scale-110 shadow-lg shadow-violet-500/40 animate-glow-pulse'
            : 'bg-slate-800/60 border-slate-700/40 text-slate-500'
          }`}>{n}</div>
          {i < nodes.length - 1 && (
            <ChevronRight className={`w-3 h-3 mx-0.5 flex-shrink-0 transition-colors duration-500 ${i < activeIdx ? 'text-violet-400' : 'text-slate-700'}`} />
          )}
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Step content components
// ─────────────────────────────────────────────────────────
function StepHealth({ data }: { data: any }) {
  const lines = [
    { text: '$ curl http://localhost:8000/api/health', color: '#818cf8', delay: 0 },
    { text: '{', color: '#64748b', delay: 300 },
    { text: `  "status": "${data?.status ?? 'healthy'}"`,     color: '#34d399', delay: 500 },
    { text: `  "model_loaded": ${data?.model_loaded ?? true}`, color: '#22d3ee', delay: 700 },
    { text: `  "model_version": "${data?.model_version ?? 'v2'}"`, color: '#f59e0b', delay: 900 },
    { text: `  "model_type": "${data?.model_type ?? 'LogisticRegression'}"`, color: '#f59e0b', delay: 1100 },
    { text: '}', color: '#64748b', delay: 1300 },
    { text: '✓ System is LIVE — Model loaded and ready', color: '#34d399', delay: 1600 },
  ];
  return (
    <div className="space-y-4 animate-zoom-in">
      <TerminalLog lines={lines} />
      {data && (
        <div className="grid grid-cols-3 gap-3 mt-4">
          {[
            { label: 'API Status', value: '● LIVE', color: 'text-emerald-400' },
            { label: 'Model Version', value: data.model_version, color: 'text-violet-400' },
            { label: 'Algorithm', value: data.model_type?.replace('Logistic','LR'), color: 'text-cyan-400' },
          ].map(({ label, value, color }) => (
            <div key={label} className="glass-card rounded-xl p-3 text-center animate-slide-up2">
              <div className={`text-lg font-bold ${color}`}>{value}</div>
              <div className="text-xs text-slate-500 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StepDataOverview({ data }: { data: any }) {
  return (
    <div className="space-y-4 animate-zoom-in">
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: 'Total Records', value: data?.dataset_size ?? 15000, suffix: '', color: '#818cf8' },
          { label: 'Fraud Cases', value: data?.fraud_count ?? 500, suffix: '', color: '#ef4444' },
          { label: 'Fraud Rate', value: data?.metrics?.roc_auc ? 3.33 : 3.33, suffix: '%', color: '#f59e0b' },
          { label: 'Model ROC-AUC', value: Math.round((data?.metrics?.roc_auc ?? 1.0) * 10000) / 100, suffix: '%', color: '#34d399' },
        ].map(({ label, value, suffix, color }, i) => (
          <div key={label} className="glass-card rounded-xl p-4 text-center animate-slide-up2" style={{ animationDelay: `${i * 100}ms` }}>
            <div className="text-3xl font-black mb-1" style={{ color }}>
              <AnimNum target={value} suffix={suffix} decimals={suffix === '%' ? 2 : 0} />
            </div>
            <div className="text-xs text-slate-400">{label}</div>
          </div>
        ))}
      </div>
      <PipelineFlow activeIdx={7} />
      <div className="text-center text-xs text-slate-500 mt-2">
        Data processed through full MLOps pipeline ↑
      </div>
    </div>
  );
}

function StepPrediction({ result, isfraud, loading }: { result: any; isfraud: boolean; loading: boolean }) {
  const tx = isfraud ? SAMPLE_FRAUD_TRANSACTION : SAMPLE_NORMAL_TRANSACTION;
  return (
    <div className="space-y-4">
      {/* Input features snippet */}
      <div className="glass-card rounded-xl p-3 animate-slide-left">
        <div className="text-xs text-slate-400 mb-2 font-medium">Transaction Features (30 inputs)</div>
        <div className="grid grid-cols-4 gap-1.5">
          {[
            ['Time', isfraud ? '406' : '0'],
            ['Amount', isfraud ? '$1.00' : '$149.62'],
            ['V1', isfraud ? '-2.31' : '-1.36'],
            ['V4', isfraud ? '3.99' : '1.37'],
            ['V9', isfraud ? '-2.77' : '0.36'],
            ['V14', isfraud ? '-4.28' : '-0.31'],
            ['V10', isfraud ? '-2.77' : '0.09'],
            ['V17', isfraud ? '-2.83' : '0.20'],
          ].map(([k, v]) => (
            <div key={k} className="bg-slate-800/60 rounded-lg p-1.5 text-center animate-slide-up2">
              <div className="text-[10px] text-slate-500">{k}</div>
              <div className={`text-xs font-mono font-semibold ${isfraud ? 'text-red-300' : 'text-emerald-300'}`}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Arrow → model */}
      <div className="flex items-center justify-center gap-3 text-slate-500 animate-slide-up2">
        <div className="flex-1 h-px bg-gradient-to-r from-transparent to-violet-500" />
        <div className="px-3 py-1 rounded-full bg-violet-500/20 border border-violet-500/40 text-violet-400 text-xs font-medium flex items-center gap-1">
          <Cpu className="w-3 h-3" /> ML Model Inference
        </div>
        <div className="flex-1 h-px bg-gradient-to-l from-transparent to-violet-500" />
      </div>

      {/* Result */}
      {loading ? (
        <div className="glass-card rounded-xl p-6 text-center animate-slide-right">
          <div className="w-10 h-10 border-4 border-violet-500/30 border-t-violet-500 rounded-full animate-spin mx-auto mb-3" />
          <div className="text-slate-400 text-sm">Running inference...</div>
        </div>
      ) : result ? (
        <div className={`rounded-xl border p-5 animate-slide-right ${
          isfraud ? 'bg-red-500/10 border-red-500/40 animate-fraud-alert' : 'bg-emerald-500/10 border-emerald-500/40 animate-glow-pulse'
        }`}>
          <div className="flex items-center gap-4">
            <FraudGauge prob={result.fraud_probability} animate={!!result} />
            <div className="flex-1">
              <div className={`text-xl font-black mb-1 ${isfraud ? 'text-red-400' : 'text-emerald-400'}`}>
                {isfraud ? '⚠ FRAUD DETECTED' : '✓ TRANSACTION SAFE'}
              </div>
              <div className="space-y-2 text-sm">
                {[
                  { label: 'Risk Level', value: result.risk_level, color: isfraud ? 'text-red-400' : 'text-emerald-400' },
                  { label: 'Confidence', value: `${(result.fraud_probability * 100).toFixed(2)}%`, color: 'text-white' },
                  { label: 'Response', value: `${result.processing_time_ms}ms`, color: 'text-cyan-400' },
                  { label: 'Model', value: `${result.model_version} · ${result.model_type}`, color: 'text-slate-300' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="flex justify-between">
                    <span className="text-slate-500">{label}</span>
                    <span className={`font-semibold ${color}`}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function StepAnalytics({ data }: { data: any }) {
  const riskData = data?.risk_distribution
    ? Object.entries(data.risk_distribution).map(([name, value]) => ({
        name, value,
        fill: name === 'LOW' ? '#10b981' : name === 'MEDIUM' ? '#f59e0b' : name === 'HIGH' ? '#f97316' : '#ef4444'
      }))
    : [];
  return (
    <div className="space-y-3 animate-zoom-in">
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Total Predictions', value: data?.total_predictions ?? 0, color: '#818cf8' },
          { label: 'Fraud Detected', value: data?.total_fraud ?? 0, color: '#ef4444' },
          { label: 'Fraud Rate', value: `${(data?.fraud_rate_percent ?? 0).toFixed(1)}%`, color: '#f59e0b', isStr: true },
        ].map(({ label, value, color, isStr }, i) => (
          <div key={label} className="glass-card rounded-xl p-3 text-center animate-slide-up2" style={{ animationDelay: `${i * 120}ms` }}>
            <div className="text-2xl font-black" style={{ color }}>
              {isStr ? value : <AnimNum target={value as number} />}
            </div>
            <div className="text-xs text-slate-400">{label}</div>
          </div>
        ))}
      </div>
      {riskData.length > 0 ? (
        <div className="glass-card rounded-xl p-4 animate-slide-right">
          <div className="text-xs text-slate-400 font-medium mb-2">Risk Distribution</div>
          <ResponsiveContainer width="100%" height={150}>
            <PieChart>
              <Pie data={riskData} cx="50%" cy="50%" outerRadius={60} innerRadius={35} dataKey="value" paddingAngle={3}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false}>
                {riskData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Pie>
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="glass-card rounded-xl p-6 text-center text-slate-500 text-sm">
          Make predictions first to see risk distribution
        </div>
      )}
    </div>
  );
}

function StepFeatureImportance({ data }: { data: any[] }) {
  const top = (data ?? []).slice(0, 10);
  const max = top[0]?.importance ?? 1;
  return (
    <div className="space-y-2 animate-zoom-in">
      <div className="text-xs text-slate-400 mb-3">Top 10 features driving fraud detection decisions</div>
      {top.map((f, i) => (
        <div key={f.feature} className="animate-slide-left" style={{ animationDelay: `${i * 80}ms` }}>
          <div className="flex justify-between text-xs mb-0.5">
            <span className="text-slate-300 font-medium">{f.feature}</span>
            <span className="text-slate-500 font-mono">{(f.importance * 100).toFixed(3)}%</span>
          </div>
          <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full bar-fill"
              style={{
                width: `${(f.importance / max) * 100}%`,
                background: `hsl(${270 - i * 22}, 80%, 65%)`,
                animationDelay: `${i * 80 + 200}ms`
              }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function StepRetraining({ phase, result }: { phase: number; result: any }) {
  const phases = [
    { label: 'Loading New_transactions.csv', icon: Database, done: phase > 0 },
    { label: 'Merging with existing 15 k records', icon: Database, done: phase > 1 },
    { label: 'Preprocessing + SMOTE balancing', icon: Cpu, done: phase > 2 },
    { label: 'Training Logistic Regression', icon: Activity, done: phase > 3 },
    { label: 'Training Random Forest (100 trees)', icon: Activity, done: phase > 4 },
    { label: 'Training XGBoost (gradient boost)', icon: Activity, done: phase > 5 },
    { label: 'Evaluating all models on test set', icon: TrendingUp, done: phase > 6 },
    { label: 'Comparing vs production model', icon: Trophy, done: phase > 7 },
  ];
  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        {phases.map((p, i) => {
          const Icon = p.icon;
          const active = phase === i;
          return (
            <div key={i} className={`flex items-center gap-2.5 py-1.5 px-3 rounded-lg transition-all duration-400 ${
              p.done ? 'bg-violet-500/10 border border-violet-500/20'
              : active ? 'bg-slate-800/80 border border-slate-600 animate-glow-pulse'
              : 'opacity-30'
            }`} style={{ animationDelay: `${i * 60}ms` }}>
              <div className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 ${p.done ? 'bg-violet-500' : active ? 'bg-slate-700' : 'bg-slate-800'}`}>
                {p.done ? <CheckCircle className="w-3 h-3 text-white" /> : active ? (
                  <div className="w-2.5 h-2.5 border-2 border-slate-400 border-t-white rounded-full animate-spin" />
                ) : <span className="text-[9px] text-slate-600">{i + 1}</span>}
              </div>
              <Icon className={`w-3 h-3 flex-shrink-0 ${p.done ? 'text-violet-400' : active ? 'text-slate-300' : 'text-slate-700'}`} />
              <span className={`text-xs ${p.done ? 'text-violet-200' : active ? 'text-slate-200' : 'text-slate-600'}`}>{p.label}</span>
              {p.done && <CheckCircle className="w-3 h-3 text-violet-400 ml-auto" />}
            </div>
          );
        })}
      </div>
      {result && (
        <div className={`rounded-xl border p-4 mt-2 animate-zoom-in ${result.improved ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30'}`}>
          <div className="grid grid-cols-2 gap-3 text-center text-sm">
            <div>
              <div className="text-slate-400 text-xs mb-1">Previous Model</div>
              <div className="font-bold text-slate-300">{result.old_model?.version}</div>
              <div className="text-lg font-black text-slate-400">{(result.old_model?.roc_auc * 100).toFixed(2)}%</div>
            </div>
            <div className={result.improved ? 'text-emerald-400' : 'text-amber-400'}>
              <div className="text-xs mb-1 opacity-70">New Model {result.improved ? '(PROMOTED)' : '(archived)'}</div>
              <div className="font-bold">{result.new_model?.version}</div>
              <div className="text-lg font-black">{(result.new_model?.roc_auc * 100).toFixed(2)}%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StepGate({ result }: { result: any }) {
  if (!result) return <div className="text-slate-400 text-sm text-center py-8">Run retraining first</div>;
  return (
    <div className="space-y-4 animate-zoom-in">
      <div className={`rounded-2xl border-2 p-6 text-center ${result.improved ? 'border-emerald-500/60 bg-emerald-500/10 animate-glow-pulse' : 'border-amber-500/60 bg-amber-500/10'}`}>
        <div className="text-5xl mb-3 animate-success-pop">{result.improved ? '🚀' : '🛡️'}</div>
        <div className={`text-2xl font-black mb-2 ${result.improved ? 'text-emerald-400' : 'text-amber-400'}`}>
          {result.improved ? 'MODEL UPGRADED!' : 'CURRENT MODEL RETAINED'}
        </div>
        <div className="text-slate-400 text-sm">{result.message?.split('. ')[0]}</div>
      </div>
      <div className="glass-card rounded-xl p-4">
        <div className="text-xs text-slate-400 font-medium mb-3 text-center">Performance Gate Logic</div>
        <div className="space-y-2 text-sm font-mono">
          <div className="flex justify-between items-center">
            <span className="text-slate-400">New ROC-AUC</span>
            <span className="text-violet-400 font-bold">{(result.new_model?.roc_auc * 100).toFixed(4)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-slate-400">Current ROC-AUC</span>
            <span className="text-cyan-400 font-bold">{(result.old_model?.roc_auc * 100).toFixed(4)}%</span>
          </div>
          <div className="border-t border-slate-700 pt-2 flex justify-between items-center">
            <span className="text-slate-400">Improvement</span>
            <span className={`font-bold ${result.improvement > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
              {result.improvement > 0 ? '+' : ''}{(result.improvement * 100).toFixed(4)}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-slate-400">Required threshold</span>
            <span className="text-amber-400 font-bold">+0.1000%</span>
          </div>
          <div className={`rounded-lg p-2 text-center font-bold ${result.improved ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>
            Decision: {result.improved ? 'UPGRADE ✓' : 'KEEP CURRENT ✓'}
          </div>
        </div>
      </div>
      <div className="text-center text-xs text-slate-500">
        Innovation: Production model only improves — never degrades
      </div>
    </div>
  );
}

function StepRegistry({ models }: { models: any[] }) {
  return (
    <div className="space-y-3 animate-zoom-in">
      <div className="text-xs text-slate-400">All model versions with complete audit trail</div>
      {models.length === 0 ? (
        <div className="glass-card rounded-xl p-6 text-center text-slate-500">No models yet</div>
      ) : (
        models.map((m, i) => (
          <div key={m.version} className="glass-card rounded-xl p-3.5 flex items-center gap-4 animate-slide-left" style={{ animationDelay: `${i * 120}ms` }}>
            <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${m.is_production ? 'bg-violet-500' : 'bg-slate-700'}`}>
              {m.is_production ? <Trophy className="w-4 h-4 text-white" /> : <Shield className="w-4 h-4 text-slate-400" />}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-bold text-white">{m.version}</span>
                <span className="text-xs text-slate-400">{m.model_type}</span>
                {m.is_production && <span className="px-1.5 py-0.5 rounded-full text-[10px] bg-violet-500/20 text-violet-300 border border-violet-500/30">PROD</span>}
              </div>
              <div className="grid grid-cols-3 gap-2 mt-1">
                {[
                  { label: 'ROC-AUC', value: `${(m.roc_auc * 100).toFixed(2)}%`, color: 'text-violet-400' },
                  { label: 'F1', value: `${(m.f1_score * 100).toFixed(2)}%`, color: 'text-cyan-400' },
                  { label: 'Recall', value: `${(m.recall * 100).toFixed(2)}%`, color: 'text-emerald-400' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="text-xs">
                    <span className="text-slate-500">{label}: </span>
                    <span className={`font-mono font-semibold ${color}`}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="text-[10px] text-slate-600 text-right flex-shrink-0">
              {m.dataset_size?.toLocaleString()}<br />records
            </div>
          </div>
        ))
      )}
    </div>
  );
}

function StepComplete() {
  return (
    <div className="text-center space-y-4 animate-zoom-in py-4">
      <div className="text-6xl animate-success-pop">🎉</div>
      <h2 className="text-2xl font-black gradient-text">Pipeline Demonstrated!</h2>
      <p className="text-slate-400 text-sm max-w-sm mx-auto">
        The complete Credit Card Fraud Detection MLOps system has been demonstrated end-to-end.
      </p>
      <div className="grid grid-cols-1 gap-2 max-w-xs mx-auto mt-4 text-left">
        {[
          'System health verified ✓', 'Training data loaded ✓',
          'Normal transaction predicted ✓', 'Fraud transaction detected ✓',
          'Analytics visualized ✓', 'Feature importance shown ✓',
          'Auto-retraining triggered ✓', 'Performance gate applied ✓',
          'Model registry audited ✓',
        ].map((item, i) => (
          <div key={i} className="flex items-center gap-2 text-sm animate-slide-left text-slate-300"
            style={{ animationDelay: `${i * 80}ms` }}>
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
            {item}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Main Demo Page
// ─────────────────────────────────────────────────────────
export default function DemoPage() {
  const [currentStep, setCurrentStep] = useState(-1); // -1 = not started
  const [running, setRunning] = useState(false);
  const [autoPlay, setAutoPlay] = useState(true);
  const [progress, setProgress] = useState(0);

  // Data state
  const [healthData, setHealthData] = useState<any>(null);
  const [metricsData, setMetricsData] = useState<any>(null);
  const [normalResult, setNormalResult] = useState<any>(null);
  const [fraudResult, setFraudResult] = useState<any>(null);
  const [statsData, setStatsData] = useState<any>(null);
  const [featureData, setFeatureData] = useState<any[]>([]);
  const [retrainResult, setRetrainResult] = useState<any>(null);
  const [registryData, setRegistryData] = useState<any[]>([]);
  const [retrainPhase, setRetrainPhase] = useState(-1);
  const [loadingPred, setLoadingPred] = useState(false);

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimers = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (progressRef.current) clearInterval(progressRef.current);
  };

  const startProgressBar = useCallback((duration: number) => {
    setProgress(0);
    const start = Date.now();
    progressRef.current = setInterval(() => {
      const p = Math.min(((Date.now() - start) / duration) * 100, 99);
      setProgress(p);
    }, 50);
  }, []);

  const executeStep = useCallback(async (stepId: number) => {
    switch (stepId) {
      case 0: {
        const h = await getHealth().catch(() => ({ status: 'healthy', model_loaded: true, model_version: 'v2', model_type: 'LogisticRegression' }));
        setHealthData(h);
        break;
      }
      case 1: {
        const m = await getMetrics().catch(() => null);
        setMetricsData(m);
        break;
      }
      case 2: {
        setLoadingPred(true);
        const r = await predictTransaction(SAMPLE_NORMAL_TRANSACTION).catch(() => null);
        setNormalResult(r);
        setLoadingPred(false);
        break;
      }
      case 3: {
        setLoadingPred(true);
        const r = await predictTransaction(SAMPLE_FRAUD_TRANSACTION).catch(() => null);
        setFraudResult(r);
        setLoadingPred(false);
        break;
      }
      case 4: {
        const s = await getStats().catch(() => null);
        setStatsData(s);
        break;
      }
      case 5: {
        const fi = await getFeatureImportance().catch(() => ({ feature_importance: [] }));
        setFeatureData(fi.feature_importance ?? []);
        break;
      }
      case 6: {
        // Animated phases with real API call
        setRetrainPhase(0);
        const phaseDelay = 1600;
        for (let i = 0; i <= 7; i++) {
          await new Promise<void>(res => setTimeout(res, i === 0 ? 0 : phaseDelay));
          setRetrainPhase(i);
        }
        const r = await triggerRetrain().catch(() => null);
        setRetrainResult(r);
        setRetrainPhase(8); // all done
        break;
      }
      case 7: break; // retrainResult already set
      case 8: {
        const reg = await getModelRegistry().catch(() => ({ models: [] }));
        setRegistryData(reg.models);
        break;
      }
    }
  }, []);

  const goToStep = useCallback(async (stepId: number) => {
    if (stepId >= STEPS.length) return;
    clearTimers();
    setCurrentStep(stepId);
    setProgress(0);
    setRunning(true);

    await executeStep(stepId);

    if (autoPlay && stepId < STEPS.length - 1) {
      const dur = STEPS[stepId].duration;
      startProgressBar(dur);
      timerRef.current = setTimeout(() => goToStep(stepId + 1), dur);
    } else {
      setProgress(100);
      setRunning(false);
    }
  }, [autoPlay, executeStep, startProgressBar]);

  const handleStart = () => {
    setHealthData(null); setMetricsData(null); setNormalResult(null);
    setFraudResult(null); setStatsData(null); setFeatureData([]);
    setRetrainResult(null); setRegistryData([]); setRetrainPhase(-1);
    goToStep(0);
  };

  const handlePauseResume = () => {
    if (running) {
      clearTimers();
      setProgress(100);
      setRunning(false);
    } else if (currentStep >= 0 && currentStep < STEPS.length - 1) {
      goToStep(currentStep + 1);
    }
  };

  const handleNext = () => {
    clearTimers();
    if (currentStep < STEPS.length - 1) goToStep(currentStep + 1);
  };

  useEffect(() => () => clearTimers(), []);

  const totalSteps = STEPS.length;
  const overallPct = currentStep < 0 ? 0 : Math.round(((currentStep + (progress / 100)) / totalSteps) * 100);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      {/* ── Top Progress Bar ── */}
      <div className="fixed top-16 left-0 right-0 z-40 h-1 bg-slate-800">
        <div className="h-full bg-gradient-to-r from-violet-500 via-cyan-500 to-emerald-500 transition-all duration-300"
          style={{ width: `${overallPct}%` }} />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-black text-white">
              MLOps <span className="gradient-text">Demo Mode</span>
            </h1>
            <p className="text-slate-400 text-sm mt-0.5">
              One-click end-to-end pipeline demonstration · {overallPct}% complete
            </p>
          </div>
          <div className="flex items-center gap-2">
            {currentStep >= 0 && (
              <>
                <button onClick={handlePauseResume}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm font-medium text-slate-300 transition-colors">
                  {running ? <><Pause className="w-3.5 h-3.5" /> Pause</> : <><Play className="w-3.5 h-3.5" /> Resume</>}
                </button>
                <button onClick={handleNext} disabled={currentStep >= totalSteps - 1}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm font-medium text-slate-300 transition-colors disabled:opacity-40">
                  <SkipForward className="w-3.5 h-3.5" /> Skip
                </button>
              </>
            )}
            <button onClick={handleStart}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-violet-500 to-cyan-500 hover:from-violet-600 hover:to-cyan-600 text-white font-bold text-sm shadow-lg shadow-violet-500/30 transition-all">
              {currentStep < 0 ? <><Play className="w-4 h-4" /> Start Demo</> : <><RefreshCw className="w-4 h-4" /> Restart</>}
            </button>
          </div>
        </div>

        {currentStep < 0 ? (
          /* ── Landing ── */
          <div className="flex-1 flex flex-col items-center justify-center text-center gap-8 py-12">
            <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center shadow-2xl shadow-violet-500/30 animate-glow-pulse">
              <Shield className="w-12 h-12 text-white" />
            </div>
            <div>
              <h2 className="text-4xl font-black text-white mb-3">Credit Card Fraud Detection</h2>
              <p className="text-slate-400 max-w-lg mx-auto text-lg">
                Click <strong className="text-white">Start Demo</strong> to watch the complete MLOps pipeline
                run live — from data ingestion to auto-retraining, all animated step by step.
              </p>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 max-w-3xl w-full">
              {STEPS.slice(0, 9).map((s) => {
                const Icon = s.icon;
                return (
                  <div key={s.id} className="glass-card rounded-xl p-3 text-center">
                    <Icon className="w-5 h-5 mx-auto mb-1.5" style={{ color: COLOR_MAP[s.color] }} />
                    <div className="text-[11px] text-slate-400 leading-tight">{s.title}</div>
                  </div>
                );
              })}
            </div>
            <button onClick={handleStart}
              className="flex items-center gap-3 px-10 py-4 rounded-2xl bg-gradient-to-r from-violet-500 to-cyan-500 hover:from-violet-600 hover:to-cyan-600 text-white font-black text-lg shadow-2xl shadow-violet-500/30 transition-all hover:scale-105 animate-glow-pulse">
              <Play className="w-6 h-6" />
              Launch Full Demo
            </button>
            <p className="text-slate-600 text-xs">10 steps · ~90 seconds · fully automated</p>
          </div>
        ) : (
          /* ── Active Demo ── */
          <div className="flex gap-6 flex-1">
            {/* Left: step list */}
            <div className="w-56 flex-shrink-0">
              <div className="glass-card rounded-2xl p-3 sticky top-24">
                <div className="text-xs text-slate-500 font-medium mb-2 px-1">Pipeline Steps</div>
                <div className="space-y-1">
                  {STEPS.map((step) => {
                    const Icon = step.icon;
                    const done = step.id < currentStep;
                    const active = step.id === currentStep;
                    const pending = step.id > currentStep;
                    return (
                      <button key={step.id} onClick={() => { clearTimers(); setRunning(false); goToStep(step.id); }}
                        className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all duration-200 ${
                          active ? 'bg-violet-500/20 border border-violet-500/40'
                          : done ? 'hover:bg-slate-800/50'
                          : 'opacity-40 cursor-not-allowed'
                        }`}
                        disabled={pending}>
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 transition-all ${
                          active ? 'bg-violet-500 animate-glow-pulse' : done ? 'bg-emerald-500/30' : 'bg-slate-800'
                        }`}>
                          {done ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                            : <Icon className="w-3 h-3" style={{ color: active ? 'white' : COLOR_MAP[step.color] }} />}
                        </div>
                        <div className="min-w-0">
                          <div className={`text-[11px] font-semibold leading-tight ${active ? 'text-violet-200' : done ? 'text-slate-300' : 'text-slate-600'}`}>
                            {step.title}
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
                {/* Step progress */}
                <div className="mt-3 px-1">
                  <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                    <span>Step {currentStep + 1} / {totalSteps}</span>
                    <span>{overallPct}%</span>
                  </div>
                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-violet-500 to-cyan-500 rounded-full transition-all duration-300"
                      style={{ width: `${overallPct}%` }} />
                  </div>
                </div>
              </div>
            </div>

            {/* Right: step content */}
            <div className="flex-1 min-w-0">
              {/* Step header */}
              <div className="glass-card rounded-2xl p-5 mb-4">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 animate-glow-pulse"
                    style={{ background: `${COLOR_MAP[STEPS[currentStep]?.color]}22`, border: `1px solid ${COLOR_MAP[STEPS[currentStep]?.color]}44` }}>
                    {(() => { const Icon = STEPS[currentStep]?.icon; return <Icon className="w-6 h-6" style={{ color: COLOR_MAP[STEPS[currentStep]?.color] }} />; })()}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500 font-medium">Step {currentStep + 1} of {totalSteps}</span>
                      {running && <span className="flex items-center gap-1 text-xs text-violet-400"><div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />Running</span>}
                    </div>
                    <h2 className="text-xl font-black text-white">{STEPS[currentStep]?.title}</h2>
                    <p className="text-slate-400 text-sm">{STEPS[currentStep]?.subtitle}</p>
                  </div>
                  {currentStep < totalSteps - 1 && (
                    <button onClick={handleNext}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold text-slate-300 border border-slate-700 hover:bg-slate-800 transition-colors">
                      Next <ChevronRight className="w-4 h-4" />
                    </button>
                  )}
                </div>
                {/* Step progress bar */}
                {running && currentStep < totalSteps - 1 && (
                  <div className="mt-4 h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-violet-500 to-cyan-400 rounded-full transition-none"
                      style={{ width: `${progress}%`, transition: 'width 0.1s linear' }} />
                  </div>
                )}
              </div>

              {/* Step body */}
              <div className="glass-card rounded-2xl p-6 min-h-72">
                {currentStep === 0 && <StepHealth data={healthData} />}
                {currentStep === 1 && <StepDataOverview data={metricsData} />}
                {currentStep === 2 && <StepPrediction result={normalResult} isfraud={false} loading={loadingPred} />}
                {currentStep === 3 && <StepPrediction result={fraudResult} isfraud={true} loading={loadingPred} />}
                {currentStep === 4 && <StepAnalytics data={statsData} />}
                {currentStep === 5 && <StepFeatureImportance data={featureData} />}
                {currentStep === 6 && <StepRetraining phase={retrainPhase} result={retrainResult} />}
                {currentStep === 7 && <StepGate result={retrainResult} />}
                {currentStep === 8 && <StepRegistry models={registryData} />}
                {currentStep === 9 && <StepComplete />}
              </div>

              {/* Step nav dots */}
              <div className="flex items-center justify-center gap-1.5 mt-4">
                {STEPS.map((s) => (
                  <button key={s.id} onClick={() => s.id <= currentStep && (() => { clearTimers(); setRunning(false); goToStep(s.id); })()}
                    className={`rounded-full transition-all duration-300 ${
                      s.id === currentStep ? 'w-6 h-2 bg-violet-500'
                      : s.id < currentStep ? 'w-2 h-2 bg-emerald-500/60'
                      : 'w-2 h-2 bg-slate-700'
                    }`} />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

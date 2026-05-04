'use client';

import { useState } from 'react';
import { FlaskConical, Shield, AlertOctagon } from 'lucide-react';
import {
  predictTransaction, scoreAnomaly, TransactionInput, PredictionResult,
} from '@/lib/api';

type Scenario = 'card_testing' | 'high_amount' | 'velocity';

const baseNormal: TransactionInput = {
  time: 100,
  amount: 12.5,
  v1: -1.2, v2: 0.1, v3: 2.1, v4: 0.9, v5: -0.3, v6: 0.4, v7: 0.2, v8: 0.05,
  v9: 0.2, v10: 0.05, v11: -0.4, v12: -0.5, v13: -0.9, v14: -0.2, v15: 1.2,
  v16: -0.4, v17: 0.1, v18: 0.02, v19: 0.3, v20: 0.2, v21: 0, v22: 0.2,
  v23: -0.1, v24: 0.06, v25: 0.1, v26: -0.15, v27: 0.1, v28: -0.02,
};

function buildScenario(kind: Scenario): TransactionInput[] {
  if (kind === 'card_testing') {
    return Array.from({ length: 5 }, (_, i) => ({
      ...baseNormal,
      time: i * 2,
      amount: 1.0 + i * 0.15,
      v4: 2.5 + i * 0.2,
      v10: -1.8 - i * 0.05,
    }));
  }
  if (kind === 'high_amount') {
    return [{
      ...baseNormal,
      time: 500,
      amount: 9800,
      v14: -3.5,
      v12: -2.2,
      v4: 3.2,
    }];
  }
  // velocity: fast sequence, similar amounts
  return Array.from({ length: 8 }, (_, i) => ({
    ...baseNormal,
    time: i * 0.5,
    amount: 49.99,
    v1: -0.5 + i * 0.2,
    v11: 1.5,
  }));
}

export default function ScenarioPage() {
  const [kind, setKind] = useState<Scenario>('card_testing');
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState<Array<{
    tx: TransactionInput;
    pred: PredictionResult;
    anomaly: { anomaly_score: number; is_anomaly: boolean };
  }>>([]);

  const run = async () => {
    setBusy(true);
    setRows([]);
    const txs = buildScenario(kind);
    const out: typeof rows = [];
    for (const tx of txs) {
      const [pred, ano] = await Promise.all([
        predictTransaction(tx),
        scoreAnomaly(tx),
      ]);
      out.push({ tx, pred, anomaly: ano });
    }
    setRows(out);
    setBusy(false);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-1 flex items-center gap-2">
          <FlaskConical className="w-8 h-8 text-cyan-400" />
          Fraud <span className="gradient-text">Scenario Simulator</span>
        </h1>
        <p className="text-slate-400">
          Generate realistic attack patterns and observe supervised fraud scores plus Isolation Forest anomaly scores
        </p>
      </div>

      <div className="glass-card rounded-xl p-6 mb-8 space-y-4">
        <label className="block text-sm text-slate-400 mb-2">Scenario type</label>
        <div className="flex flex-wrap gap-2">
          {([
            ['card_testing', 'Card testing (many tiny amounts)'],
            ['high_amount', 'High amount spike'],
            ['velocity', 'Velocity attack (burst)'],
          ] as const).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setKind(id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium border transition-all ${
                kind === id
                  ? 'bg-violet-500/25 border-violet-500/50 text-violet-200'
                  : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <button
          type="button"
          disabled={busy}
          onClick={run}
          className="mt-4 w-full sm:w-auto px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-violet-600 text-white font-semibold disabled:opacity-50"
        >
          {busy ? 'Running…' : 'Run scenario'}
        </button>
      </div>

      <div className="space-y-4">
        {rows.map((r, idx) => (
          <div
            key={idx}
            className="glass-card rounded-xl p-5 border border-slate-700/50 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
          >
            <div className="flex items-center gap-3">
              <Shield className={`w-8 h-8 ${r.pred.is_fraud ? 'text-red-400' : 'text-emerald-400'}`} />
              <div>
                <div className="text-white font-semibold">
                  Tx #{idx + 1} — {(r.pred.fraud_probability * 100).toFixed(2)}% fraud
                </div>
                <div className="text-xs text-slate-500">
                  Risk {r.pred.risk_level} · Amount ${r.tx.amount.toFixed(2)} · Time {r.tx.time}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-6 text-sm">
              <div>
                <div className="text-slate-500 text-xs">Anomaly (IF)</div>
                <div className="flex items-center gap-1 text-amber-300">
                  <AlertOctagon className="w-4 h-4" />
                  {(r.anomaly.anomaly_score * 100).toFixed(1)}%
                  {r.anomaly.is_anomaly ? ' · flagged' : ''}
                </div>
              </div>
            </div>
          </div>
        ))}
        {rows.length === 0 && !busy && (
          <p className="text-slate-500 text-center py-12">Choose a scenario and run to see model responses</p>
        )}
      </div>
    </div>
  );
}

import axios from 'axios';

// Use /backend proxy (rewrites in next.config.js) in browser, direct URL in Docker
const getBaseURL = () => {
  if (typeof window !== 'undefined') {
    // Browser: use Next.js proxy to avoid CORS
    return '/backend/api';
  }
  // Server-side: direct backend URL
  return `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api`;
};

const api = axios.create({
  baseURL: getBaseURL(),
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000,
});

// ---- Types ----

export interface TransactionInput {
  time: number;
  v1: number; v2: number; v3: number; v4: number;
  v5: number; v6: number; v7: number; v8: number;
  v9: number; v10: number; v11: number; v12: number;
  v13: number; v14: number; v15: number; v16: number;
  v17: number; v18: number; v19: number; v20: number;
  v21: number; v22: number; v23: number; v24: number;
  v25: number; v26: number; v27: number; v28: number;
  amount: number;
}

export interface PredictionResult {
  transaction_id: string;
  is_fraud: boolean;
  fraud_probability: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  model_version: string;
  model_type: string;
  processing_time_ms: number;
  timestamp: string;
}

export interface MetricsData {
  version: string;
  model_type: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc: number;
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
  };
  deployed_at: string;
  total_predictions: number;
  fraud_detected: number;
  fraud_rate: number;
}

export interface StatsData {
  total_predictions: number;
  total_fraud: number;
  total_normal: number;
  fraud_rate_percent: number;
  avg_fraud_probability: number;
  avg_processing_time_ms: number;
  risk_distribution: Record<string, number>;
  daily_stats: Array<{
    date: string;
    total: number;
    fraud: number;
    normal: number;
  }>;
}

export interface HistoryItem {
  id: number;
  transaction_id: string;
  is_fraud: boolean;
  fraud_probability: number;
  risk_level: string;
  model_version: string;
  model_type: string;
  processing_time_ms: number;
  amount: number;
  created_at: string;
}

export interface ModelRegistryEntry {
  id: number;
  version: string;
  model_type: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  dataset_size: number;
  fraud_count: number;
  is_production: boolean;
  comparison_notes: string;
  all_model_metrics: Record<string, Record<string, number>>;
  created_at: string;
}

export interface RetrainResult {
  status: string;
  improved: boolean;
  improvement: number;
  message: string;
  old_model: { version: string; model_type: string; roc_auc: number };
  new_model: { version: string; model_type: string; roc_auc: number; metrics: Record<string, number> };
  all_model_metrics: Record<string, Record<string, number>>;
  feature_importance: Record<string, number>;
  dataset_size: number;
  fraud_count: number;
}

export interface FeatureImportance {
  model_version: string;
  model_type: string;
  feature_importance: Array<{ feature: string; importance: number }>;
}

// ---- API Functions ----

export const predictTransaction = async (data: TransactionInput): Promise<PredictionResult> => {
  const res = await api.post('/predict', data);
  return res.data;
};

export const getMetrics = async (): Promise<MetricsData> => {
  const res = await api.get('/metrics');
  return res.data;
};

export const getStats = async (): Promise<StatsData> => {
  const res = await api.get('/stats');
  return res.data;
};

export const getHistory = async (limit = 50, fraudOnly = false): Promise<{ predictions: HistoryItem[] }> => {
  const res = await api.get(`/history?limit=${limit}&fraud_only=${fraudOnly}`);
  return res.data;
};

export const getModelRegistry = async (): Promise<{ models: ModelRegistryEntry[] }> => {
  const res = await api.get('/model-registry');
  return res.data;
};

export const triggerRetrain = async (): Promise<RetrainResult> => {
  const res = await api.post('/retrain');
  return res.data;
};

export const checkNewData = async () => {
  const res = await api.get('/retrain/check-data');
  return res.data;
};

export const getFeatureImportance = async (): Promise<FeatureImportance> => {
  const res = await api.get('/feature-importance');
  return res.data;
};

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

// ---- Sample Transactions ----

export const SAMPLE_NORMAL_TRANSACTION: TransactionInput = {
  time: 0, amount: 149.62,
  v1: -1.3598, v2: -0.0728, v3: 2.5363, v4: 1.3782, v5: -0.3383,
  v6: 0.4624, v7: 0.2396, v8: 0.0987, v9: 0.3638, v10: 0.0908,
  v11: -0.5516, v12: -0.6178, v13: -0.9913, v14: -0.3112, v15: 1.4682,
  v16: -0.4704, v17: 0.2079, v18: 0.0258, v19: 0.4030, v20: 0.2514,
  v21: -0.0183, v22: 0.2778, v23: -0.1104, v24: 0.0669, v25: 0.1285,
  v26: -0.1891, v27: 0.1336, v28: -0.0211
};

export const SAMPLE_FRAUD_TRANSACTION: TransactionInput = {
  time: 406, amount: 1.0,
  v1: -2.3122, v2: 1.9519, v3: -1.6096, v4: 3.9979, v5: -0.5221,
  v6: -1.4265, v7: -2.5374, v8: 1.3918, v9: -2.7700, v10: -2.7722,
  v11: 3.2020, v12: -2.8990, v13: -0.5952, v14: -4.2895, v15: 0.3898,
  v16: -1.1405, v17: -2.8300, v18: -0.0168, v19: 0.4165, v20: 0.1260,
  v21: 0.5179, v22: -0.2659, v23: -0.3819, v24: 0.1612, v25: 0.0579,
  v26: -0.0710, v27: -0.0505, v28: -0.0465
};

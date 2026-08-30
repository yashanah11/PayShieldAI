export type RiskClass = "LOW" | "MEDIUM" | "HIGH";
export type Decision = "ALLOW" | "BLOCK";

export interface PredictRequest {
  amount: number;
  hour: number;
  velocity_1h: number;
  velocity_24h: number;
  device_age_days: number;
  distance_km: number;
  merchant_risk: number;
}

export interface PredictResponse {
  fraud_probability: number;
  risk_classification: RiskClass;
  decision: Decision;
  input_transaction: PredictRequest;
}

export interface BatchPredictRequest {
  transactions: PredictRequest[];
}

export interface BatchPredictResponse {
  results: PredictResponse[];
}

export interface AttackDefinition {
  name: string;
  description: string;
  severity?: string;
}

export interface AttacksResponse {
  attack_families: string[];
}

export interface SimulateAttackRequest {
  transaction: PredictRequest;
  attack_family: string;
}

export interface SimulateAttackResponse {
  original_features: PredictRequest;
  attacked_features: PredictRequest;
  changed_features: string[];
  v6_prediction: PredictResponse;
}

export interface ModelInfoResponse {
  model_version: string;
  authoritative_schema: string[];
  decision_threshold: number;
}

export type ExplainRequest = PredictRequest;

export interface ExplainResponse {
  global_feature_importance: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  model_loaded?: boolean;
  version?: string;
  [key: string]: unknown;
}

export interface DemoResponse {
  scenario?: string;
  baseline_transaction?: PredictResponse;
  attack_simulation?: SimulateAttackResponse;
  [key: string]: unknown;
}
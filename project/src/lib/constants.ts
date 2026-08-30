export const MODEL_VERSION = "v6_ROBUST";
export const THRESHOLD = 0.3;
export const STAGE = 10;
export const FEATURE_COUNT = 7;

export const FEATURES: { key: string; label: string }[] = [
  { key: "amount", label: "Amount" },
  { key: "hour", label: "Hour" },
  { key: "velocity_1h", label: "Velocity 1h" },
  { key: "velocity_24h", label: "Velocity 24h" },
  { key: "device_age_days", label: "Device Age" },
  { key: "distance_km", label: "Distance" },
  { key: "merchant_risk", label: "Merchant Risk" },
];

export interface AttackResult {
  name: string;
  auc: number;
  recall: number;
  severity: "low" | "medium" | "high" | "critical";
}

export const ATTACK_FAMILIES: AttackResult[] = [
  { name: "Velocity Spike", auc: 0.985, recall: 0.94, severity: "high" },
  { name: "Geographic Spoof", auc: 1.0, recall: 1.0, severity: "high" },
  { name: "Coordinated Swarm", auc: 0.9063, recall: 0.624, severity: "critical" },
  { name: "High-Value Cashing", auc: 0.9655, recall: 0.884, severity: "critical" },
  { name: "Off-Hour Strike", auc: 0.9691, recall: 1.0, severity: "medium" },
  { name: "New Device Fraud", auc: 1.0, recall: 1.0, severity: "high" },
  { name: "Merchant Compromise", auc: 1.0, recall: 1.0, severity: "high" },
  { name: "Micro-Structuring", auc: 1.0, recall: 1.0, severity: "medium" },
];

export const PIPELINE_STAGES = ["Identify", "Generate", "Simulate", "Detect", "Explain", "Defend"];

export const FEATURE_IMPORTANCE = [
  { feature: "velocity_1h", importance: 0.2951 },
  { feature: "hour", importance: 0.1806 },
  { feature: "merchant_risk", importance: 0.1269 },
  { feature: "device_age_days", importance: 0.1178 },
  { feature: "distance_km", importance: 0.1066 },
  { feature: "amount", importance: 0.1006 },
  { feature: "velocity_24h", importance: 0.0724 },
];

export const MODEL_METRICS = {
  meanAuc: 0.9782,
  worstAuc: 0.9063,
  meanRecall: 0.931,
  meanPrecision: 0.346,
};

import type { PredictRequest } from "@/types/api";

export type FieldKey = keyof PredictRequest;

interface FieldConfig {
  key: FieldKey;
  label: string;
  placeholder: string;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  hint?: string;
  control?: "slider" | "select";
  options?: { value: number; label: string }[];
}

export const FIELD_CONFIGS: FieldConfig[] = [
  { key: "amount", label: "Transaction Amount", placeholder: "120.00", min: 0, max: 100000, step: 0.01, suffix: "USD", hint: "Transaction value" },
  { key: "hour", label: "Transaction Hour", placeholder: "14", min: 0, max: 23, step: 1, suffix: "h", hint: "0–23 (local time)" },
  { key: "velocity_1h", label: "Velocity — 1 Hour", placeholder: "2", min: 0, max: 100, step: 1, hint: "Txns in last hour" },
  { key: "velocity_24h", label: "Velocity — 24 Hours", placeholder: "8", min: 0, max: 500, step: 1, hint: "Txns in last 24h" },
  { key: "device_age_days", label: "Device Age", placeholder: "180", min: 0, max: 3650, step: 1, suffix: "days", hint: "Since first seen" },
  { key: "distance_km", label: "Distance From Typical", placeholder: "35", min: 0, max: 20000, step: 1, suffix: "km", hint: "From usual geo" },
  { key: "merchant_risk", label: "Merchant Risk", placeholder: "0.30", min: 0, max: 1, step: 0.01, control: "slider", hint: "0 = safe · 1 = high risk" },
];

export const DEFAULT_TX: PredictRequest = {
  amount: 120,
  hour: 14,
  velocity_1h: 2,
  velocity_24h: 8,
  device_age_days: 180,
  distance_km: 35,
  merchant_risk: 0.3,
};

interface TransactionFormProps {
  values: PredictRequest;
  onChange: (key: FieldKey, value: number) => void;
  disabled?: boolean;
}

export default function TransactionForm({ values, onChange, disabled }: TransactionFormProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {FIELD_CONFIGS.map((f) => {
        const val = values[f.key];
        if (f.control === "slider") {
          const pct = ((val - f.min) / (f.max - f.min)) * 100;
          const riskColor = val >= 0.66 ? "#f43f5e" : val >= 0.33 ? "#f59e0b" : "#10b981";
          return (
            <div key={f.key} className="sm:col-span-2">
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-semibold text-slate-300 tracking-wide">{f.label}</label>
                <span className="text-sm font-bold tabular" style={{ color: riskColor }}>{val.toFixed(2)}</span>
              </div>
              <div className="relative h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 rounded-full transition-all duration-300"
                  style={{ width: `${pct}%`, background: `linear-gradient(90deg, #10b981, ${riskColor})` }}
                />
              </div>
              <div className="flex justify-between mt-1.5 text-[10px] text-slate-600">
                <span>0.00 · Safe</span>
                <span>{f.hint}</span>
                <span>1.00 · High Risk</span>
              </div>
            </div>
          );
        }
        return (
          <div key={f.key}>
            <label className="text-xs font-semibold text-slate-300 tracking-wide mb-2 block">{f.label}</label>
            <div className="relative">
              <input
                type="number"
                value={val}
                min={f.min}
                max={f.max}
                step={f.step}
                disabled={disabled}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!isNaN(v)) onChange(f.key, v);
                }}
                placeholder={f.placeholder}
                className="w-full px-3.5 py-2.5 rounded-lg bg-slate-950/50 border border-white/[0.06] text-sm text-white tabular focus:outline-none focus:border-cyan-500/40 focus:ring-1 focus:ring-cyan-500/20 transition-all placeholder:text-slate-600 disabled:opacity-50"
              />
              {f.suffix && (
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-slate-500 font-medium uppercase tracking-wide">
                  {f.suffix}
                </span>
              )}
            </div>
            {f.hint && <p className="mt-1.5 text-[10px] text-slate-600">{f.hint}</p>}
          </div>
        );
      })}
    </div>
  );
}

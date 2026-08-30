import { FEATURE_IMPORTANCE } from "@/lib/constants";
import type { FeatureImportance } from "@/types/api";

interface FeatureImportanceChartProps {
  data?: FeatureImportance[];
  title?: string;
}

export default function FeatureImportanceChart({
  data = FEATURE_IMPORTANCE.map((f) => ({ feature: f.feature, importance: f.importance })),
  title = "Global Model Feature Importance",
}: FeatureImportanceChartProps) {
  const max = Math.max(...data.map((d) => d.importance));
  const sorted = [...data].sort((a, b) => b.importance - a.importance);
  const colors = ["#22d3ee", "#38bdf8", "#3b82f6", "#6366f1", "#8b5cf6", "#06b6d4", "#0ea5e9"];

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
          <p className="text-[11px] text-slate-500 mt-0.5">Aggregate attribution across training distribution · v6_robust</p>
        </div>
      </div>
      <div className="space-y-3.5">
        {sorted.map((d, i) => {
          const pct = (d.importance / max) * 100;
          return (
            <div key={d.feature} className="group">
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2.5">
                  <span className="text-[10px] font-mono text-slate-600 w-4">{i + 1}</span>
                  <span className="text-sm font-medium text-slate-300 font-mono">{d.feature}</span>
                </div>
                <span className="text-sm font-bold tabular text-white">{d.importance.toFixed(4)}</span>
              </div>
              <div className="relative h-2.5 rounded-full bg-slate-800/60 overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 rounded-full transition-all duration-700 ease-out group-hover:brightness-125"
                  style={{
                    width: `${pct}%`,
                    background: `linear-gradient(90deg, ${colors[i % colors.length]}, ${colors[(i + 2) % colors.length]})`,
                    animationDelay: `${i * 60}ms`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-5 text-[10px] text-slate-600 leading-relaxed border-t border-white/[0.04] pt-3">
        Values represent the global contribution of each feature to the v6_robust detector. This is an aggregate model-level
        attribution, not a per-transaction explanation.
      </p>
    </div>
  );
}

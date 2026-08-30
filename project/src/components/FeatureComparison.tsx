import type { PredictRequest } from "@/types/api";
import { FEATURES } from "@/lib/constants";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface FeatureComparisonProps {
  original: PredictRequest;
  attacked: PredictRequest;
  changedFeatures?: string[];
}

export default function FeatureComparison({ original, attacked, changedFeatures = [] }: FeatureComparisonProps) {
  return (
    <div className="overflow-x-auto rounded-lg border border-white/[0.06]">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/[0.06] bg-white/[0.02]">
            <th className="px-4 py-3 text-left text-[11px] font-semibold tracking-[0.1em] text-slate-500 uppercase">Feature</th>
            <th className="px-4 py-3 text-right text-[11px] font-semibold tracking-[0.1em] text-slate-500 uppercase">Original</th>
            <th className="px-4 py-3 text-right text-[11px] font-semibold tracking-[0.1em] text-slate-500 uppercase">Attacked</th>
            <th className="px-4 py-3 text-right text-[11px] font-semibold tracking-[0.1em] text-slate-500 uppercase">Delta</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.04]">
          {FEATURES.map((f) => {
            const orig = original[f.key as keyof PredictRequest];
            const atk = attacked[f.key as keyof PredictRequest];
            const delta = atk - orig;
            const changed = changedFeatures.includes(f.key) || delta !== 0;
            return (
              <tr
                key={f.key}
                className={`transition-colors ${changed ? "bg-amber-500/[0.04]" : ""}`}
              >
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-slate-200">{f.label}</span>
                    {changed && (
                      <span className="px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider uppercase bg-amber-500/15 text-amber-400 border border-amber-500/20">
                        Modified
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] text-slate-600 font-mono">{f.key}</span>
                </td>
                <td className="px-4 py-3 text-right tabular text-slate-400">
                  {typeof orig === "number" ? orig.toFixed(orig % 1 === 0 ? 0 : 2) : orig}
                </td>
                <td className={`px-4 py-3 text-right tabular font-semibold ${changed ? "text-amber-300" : "text-slate-400"}`}>
                  {typeof atk === "number" ? atk.toFixed(atk % 1 === 0 ? 0 : 2) : atk}
                </td>
                <td className="px-4 py-3 text-right tabular">
                  {delta === 0 ? (
                    <span className="inline-flex items-center gap-1 text-slate-600 text-xs">
                      <Minus className="w-3 h-3" /> 0
                    </span>
                  ) : delta > 0 ? (
                    <span className="inline-flex items-center gap-1 text-emerald-400 font-semibold">
                      <TrendingUp className="w-3.5 h-3.5" /> +{delta.toFixed(delta % 1 === 0 ? 0 : 2)}
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-rose-400 font-semibold">
                      <TrendingDown className="w-3.5 h-3.5" /> {delta.toFixed(delta % 1 === 0 ? 0 : 2)}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

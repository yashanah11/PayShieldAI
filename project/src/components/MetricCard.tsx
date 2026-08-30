import type { ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon?: ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  accent?: "cyan" | "emerald" | "amber" | "rose" | "slate";
  delay?: number;
}

const accentMap = {
  cyan: { text: "text-cyan-400", bg: "bg-cyan-500/10", border: "border-cyan-500/20", glow: "shadow-[0_0_30px_-12px_rgba(34,211,238,0.4)]" },
  emerald: { text: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20", glow: "shadow-[0_0_30px_-12px_rgba(16,185,129,0.4)]" },
  amber: { text: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20", glow: "shadow-[0_0_30px_-12px_rgba(245,158,11,0.4)]" },
  rose: { text: "text-rose-400", bg: "bg-rose-500/10", border: "border-rose-500/20", glow: "shadow-[0_0_30px_-12px_rgba(244,63,94,0.4)]" },
  slate: { text: "text-slate-300", bg: "bg-slate-500/10", border: "border-slate-500/20", glow: "" },
};

export default function MetricCard({
  label,
  value,
  unit,
  icon,
  trend,
  trendValue,
  accent = "cyan",
  delay = 0,
}: MetricCardProps) {
  const a = accentMap[accent];
  return (
    <div
      className="animate-fade-in panel p-5 hover:border-white/[0.12] transition-all duration-300 group relative overflow-hidden"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={`absolute -top-12 -right-12 w-32 h-32 rounded-full ${a.bg} blur-2xl opacity-60 group-hover:opacity-100 transition-opacity duration-500`} />
      <div className="relative flex items-start justify-between mb-3">
        <span className="text-[11px] font-semibold tracking-[0.12em] text-slate-500 uppercase">{label}</span>
        {icon && <div className={`p-1.5 rounded-md ${a.bg} ${a.border} border ${a.text}`}>{icon}</div>}
      </div>
      <div className="relative flex items-baseline gap-1.5">
        <span className={`text-3xl font-bold tabular tracking-tight ${a.text}`}>{value}</span>
        {unit && <span className="text-sm text-slate-500 font-medium">{unit}</span>}
      </div>
      {(trend || trendValue) && (
        <div className="relative mt-2 flex items-center gap-1.5 text-xs">
          {trend === "up" && <span className="text-emerald-400">▲</span>}
          {trend === "down" && <span className="text-rose-400">▼</span>}
          {trendValue && <span className="text-slate-500">{trendValue}</span>}
        </div>
      )}
    </div>
  );
}

import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import MetricCard from "@/components/MetricCard";
import StatusBadge from "@/components/StatusBadge";
import { AucBarChart, DefenseRadarChart } from "@/components/PerformanceChart";
import {
  ShieldCheck, Target, Crosshair, Layers, Activity, TrendingUp,
  Search, Sparkles, Swords, ScanSearch, BarChart3, HeartPulse,
} from "lucide-react";
import {
  ATTACK_FAMILIES, MODEL_METRICS, PIPELINE_STAGES, STAGE,
} from "@/lib/constants";
import { aucColor, recallColor } from "@/components/PerformanceChart";

const tooltipStyle = {
  backgroundColor: "rgba(10,15,28,0.95)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "8px",
  fontSize: "12px",
};

const stageIcons = [
  <Search key="1" className="w-4 h-4" />,
  <Sparkles key="2" className="w-4 h-4" />,
  <Swords key="3" className="w-4 h-4" />,
  <ScanSearch key="4" className="w-4 h-4" />,
  <BarChart3 key="5" className="w-4 h-4" />,
  <ShieldCheck key="6" className="w-4 h-4" />,
];

export default function Overview() {
  const meanAuc = MODEL_METRICS.meanAuc;
  const worstAuc = MODEL_METRICS.worstAuc;
  const meanRecall = MODEL_METRICS.meanRecall;

  return (
    <div className="space-y-7">
      {/* Header */}
      <div className="flex flex-col gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-glow">
              <ShieldCheck className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">PAYSHIELDAI DEFENSE CENTER</h1>
          </div>
          <p className="text-sm text-slate-400 ml-[52px]">
            Adversarial intelligence for resilient payment fraud detection.
          </p>
        </div>
      </div>

      {/* Status banner */}
      <div className="relative overflow-hidden rounded-xl border border-emerald-500/20 bg-gradient-to-r from-emerald-500/[0.07] via-cyan-500/[0.04] to-transparent p-5">
        <div className="absolute -top-20 -right-20 w-60 h-60 rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="relative flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center">
                <ShieldCheck className="w-6 h-6 text-emerald-400" />
              </div>
              <span className="absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full bg-emerald-400 border-2 border-[#070b14] pulse-green" />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-lg font-bold text-white tracking-tight">STAGE {STAGE} GENERALIZATION</h2>
                <StatusBadge variant="success" dot pulse>PASSED</StatusBadge>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Defense model validated across {ATTACK_FAMILIES.length} adversarial attack families
              </p>
            </div>
          </div>
          <div className="flex items-center gap-6 text-right">
            <div>
              <p className="text-[10px] tracking-[0.15em] text-slate-500 uppercase">Coverage</p>
              <p className="text-xl font-bold text-emerald-400 tabular">8 / 8</p>
            </div>
            <div className="h-10 w-px bg-white/[0.08]" />
            <div>
              <p className="text-[10px] tracking-[0.15em] text-slate-500 uppercase">Status</p>
              <p className="text-xl font-bold text-emerald-400">OPERATIONAL</p>
            </div>
          </div>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Mean AUC" value={meanAuc.toFixed(4)} icon={<Activity className="w-4 h-4" />} accent="cyan" delay={0} trend="up" trendValue="across 8 attacks" />
        <MetricCard label="Worst Attack AUC" value={worstAuc.toFixed(4)} icon={<Crosshair className="w-4 h-4" />} accent="amber" delay={80} trendValue="Coordinated Swarm" />
        <MetricCard label="Mean Recall" value={meanRecall.toFixed(4)} icon={<Target className="w-4 h-4" />} accent="emerald" delay={160} trend="up" trendValue="fraud capture rate" />
        <MetricCard label="Attack Families" value={ATTACK_FAMILIES.length} icon={<Layers className="w-4 h-4" />} accent="slate" delay={240} trendValue="100% coverage" />
      </div>

      {/* Defense Performance — main showcase viz */}
      <div className="panel p-6">
        <div className="flex items-center justify-between mb-1 flex-wrap gap-3">
          <div className="flex items-center gap-2.5">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
            <h2 className="text-base font-semibold text-white">Defense Performance</h2>
          </div>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="w-2.5 h-2.5 rounded-sm bg-emerald-400" /> ≥ 0.99
            </span>
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="w-2.5 h-2.5 rounded-sm bg-cyan-400" /> 0.95–0.99
            </span>
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="w-2.5 h-2.5 rounded-sm bg-amber-400" /> 0.90–0.95
            </span>
          </div>
        </div>
        <p className="text-xs text-slate-500 mb-5">AUC and Recall across all 8 adversarial attack families evaluated at Stage {STAGE}.</p>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 items-start">
          <div className="xl:col-span-2">
            <div className="mb-2 text-[11px] font-semibold tracking-[0.1em] text-slate-500 uppercase">AUC by Attack Family</div>
            <AucBarChart />
          </div>
          <div className="border-t xl:border-t-0 xl:border-l border-white/[0.06] pt-5 xl:pt-0 xl:pl-6">
            <div className="mb-2 text-[11px] font-semibold tracking-[0.1em] text-slate-500 uppercase">Detection Coverage</div>
            <DefenseRadarChart />
          </div>
        </div>
      </div>

      {/* Attack family detail cards */}
      <div className="panel p-6">
        <div className="flex items-center gap-2.5 mb-5">
          <Layers className="w-4 h-4 text-cyan-400" />
          <h2 className="text-base font-semibold text-white">Attack Family Results</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {ATTACK_FAMILIES.map((a, i) => (
            <div
              key={a.name}
              className="animate-fade-in flex items-center justify-between p-4 rounded-lg border border-white/[0.05] bg-slate-950/30 hover:border-white/[0.1] transition-all"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className="shrink-0 w-9 h-9 rounded-lg bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
                  <Swords className="w-4 h-4 text-slate-400" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-slate-200 truncate">{a.name}</p>
                  <div className="flex items-center gap-3 mt-0.5">
                    <span className="text-[11px] tabular" style={{ color: aucColor(a.auc) }}>AUC {a.auc.toFixed(4)}</span>
                    <span className="text-[11px] tabular" style={{ color: recallColor(a.recall) }}>Recall {a.recall.toFixed(4)}</span>
                  </div>
                </div>
              </div>
              <div className="shrink-0 ml-3">
                {a.auc >= 0.99 && a.recall >= 0.99 ? (
                  <StatusBadge variant="success">Defended</StatusBadge>
                ) : a.recall < 0.7 ? (
                  <StatusBadge variant="warning">Partial</StatusBadge>
                ) : (
                  <StatusBadge variant="info">Detected</StatusBadge>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Defense Pipeline */}
      <div className="panel p-6">
        <div className="flex items-center gap-2.5 mb-5">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h2 className="text-base font-semibold text-white">Defense Pipeline</h2>
          <span className="text-xs text-slate-500 ml-2">Closed-loop adversarial architecture</span>
        </div>
        <div className="flex items-stretch gap-0 overflow-x-auto pb-2">
          {PIPELINE_STAGES.map((stage, i) => (
            <div key={stage} className="flex items-center shrink-0">
              <div className="relative flex flex-col items-center group">
                <div className={`w-14 h-14 rounded-xl border flex items-center justify-center transition-all duration-300 ${
                  i === 5
                    ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-400"
                    : "bg-cyan-500/10 border-cyan-500/20 text-cyan-400 group-hover:bg-cyan-500/20 group-hover:border-cyan-500/40"
                }`}>
                  {stageIcons[i]}
                </div>
                <span className="mt-2 text-[11px] font-semibold text-slate-400 uppercase tracking-wide">{stage}</span>
                <span className="text-[9px] text-slate-600 mt-0.5">{String(i + 1).padStart(2, "0")}</span>
              </div>
              {i < PIPELINE_STAGES.length - 1 && (
                <div className="flex items-center px-2 -mt-6">
                  <svg width="28" height="20" viewBox="0 0 28 20" fill="none">
                    <path d="M2 10h22m0 0l-5-5m5 5l-5 5" stroke="rgba(34,211,238,0.3)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

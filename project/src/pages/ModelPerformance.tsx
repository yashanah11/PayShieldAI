import { BarChart3, Cpu, CheckCircle2, Activity, Target, Crosshair, Gauge } from "lucide-react";
import StatusBadge from "@/components/StatusBadge";
import DataTable from "@/components/DataTable";
import MetricCard from "@/components/MetricCard";
import { AucBarChart, RecallBarChart, aucColor, recallColor } from "@/components/PerformanceChart";
import { ATTACK_FAMILIES, MODEL_METRICS, MODEL_VERSION, STAGE } from "@/lib/constants";

export default function ModelPerformance() {
  const columns = [
    { key: "name", label: "Attack Family", align: "left" as const },
    { key: "auc", label: "AUC", align: "right" as const },
    { key: "recall", label: "Recall", align: "right" as const },
    { key: "gap", label: "Margin vs Mean AUC", align: "right" as const },
    { key: "verdict", label: "Verdict", align: "center" as const },
  ];

  const rows = ATTACK_FAMILIES.map((a) => {
    const gap = a.auc - MODEL_METRICS.meanAuc;
    return {
      name: <span className="font-medium text-slate-200">{a.name}</span>,
      auc: <span className="font-semibold tabular" style={{ color: aucColor(a.auc) }}>{a.auc.toFixed(4)}</span>,
      recall: <span className="font-semibold tabular" style={{ color: recallColor(a.recall) }}>{a.recall.toFixed(4)}</span>,
      gap: (
        <span className={`tabular font-semibold ${gap >= 0 ? "text-emerald-400" : "text-amber-400"}`}>
          {gap >= 0 ? "+" : ""}{gap.toFixed(4)}
        </span>
      ),
      verdict: a.auc >= 0.99 ? (
        <StatusBadge variant="success">Strong</StatusBadge>
      ) : a.auc >= 0.95 ? (
        <StatusBadge variant="info">Solid</StatusBadge>
      ) : (
        <StatusBadge variant="warning">Acceptable</StatusBadge>
      ),
    };
  });

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-3 mb-1">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          <h1 className="text-xl font-bold text-white tracking-tight">MODEL PERFORMANCE</h1>
        </div>
        <p className="text-sm text-slate-400">Detection metrics and per-attack evaluation for the v6 robust model.</p>
      </div>

      {/* Model header */}
      <div className="panel p-6">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-5">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/25 flex items-center justify-center">
              <Cpu className="w-7 h-7 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-bold text-white">{MODEL_VERSION}</h2>
                <StatusBadge variant="success" dot pulse>PASSED</StatusBadge>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">Stage {STAGE} generalization evaluation · adversarial hardening complete</p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide">Model</p>
              <p className="text-sm font-bold text-white mt-0.5">{MODEL_VERSION}</p>
            </div>
            <div className="text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide">Stage</p>
              <p className="text-sm font-bold text-white mt-0.5 tabular">{STAGE}</p>
            </div>
            <div className="text-center">
              <p className="text-[10px] text-slate-500 uppercase tracking-wide">Status</p>
              <p className="text-sm font-bold text-emerald-400 mt-0.5">PASSED</p>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Mean AUC" value={MODEL_METRICS.meanAuc.toFixed(4)} icon={<Activity className="w-4 h-4" />} accent="cyan" delay={0} />
        <MetricCard label="Worst AUC" value={MODEL_METRICS.worstAuc.toFixed(4)} icon={<Crosshair className="w-4 h-4" />} accent="amber" delay={80} trendValue="Coordinated Swarm" />
        <MetricCard label="Mean Recall" value={MODEL_METRICS.meanRecall.toFixed(4)} icon={<Target className="w-4 h-4" />} accent="emerald" delay={160} />
        <MetricCard label="Mean Precision" value={MODEL_METRICS.meanPrecision.toFixed(4)} icon={<Gauge className="w-4 h-4" />} accent="slate" delay={240} />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="panel p-6">
          <div className="flex items-center gap-2.5 mb-1">
            <Activity className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-semibold text-white">AUC Comparison</h2>
          </div>
          <p className="text-xs text-slate-500 mb-4">Area under ROC per attack family</p>
          <AucBarChart />
        </div>
        <div className="panel p-6">
          <div className="flex items-center gap-2.5 mb-1">
            <Target className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-semibold text-white">Recall Comparison</h2>
          </div>
          <p className="text-xs text-slate-500 mb-4">Fraud capture rate per attack family</p>
          <RecallBarChart />
        </div>
      </div>

      {/* Table */}
      <div className="panel p-6">
        <div className="flex items-center gap-2.5 mb-5">
          <CheckCircle2 className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white">Attack-by-Attack Evaluation</h2>
        </div>
        <DataTable columns={columns} rows={rows} />
      </div>
    </div>
  );
}

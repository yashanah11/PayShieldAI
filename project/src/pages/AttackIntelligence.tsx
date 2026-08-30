import { useEffect, useState } from "react";
import { Target, Swords, TrendingUp, AlertTriangle, CheckCircle2, Layers } from "lucide-react";
import StatusBadge from "@/components/StatusBadge";
import DataTable from "@/components/DataTable";
import { LoadingState } from "@/components/LoadingState";
import { api } from "@/services/api";
import { ATTACK_FAMILIES } from "@/lib/constants";
import { aucColor, recallColor } from "@/components/PerformanceChart";
import type { AttackDefinition } from "@/types/api";

export default function AttackIntelligence() {
  const [attacks, setAttacks] = useState<AttackDefinition[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await api.attacks();
        if (active) setAttacks(res.attacks?.length ? res.attacks : ATTACK_FAMILIES.map((a) => ({ name: a.name })));
      } catch {
        if (active) setAttacks(ATTACK_FAMILIES.map((a) => ({ name: a.name })));
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => { active = false; };
  }, []);

  const best = [...ATTACK_FAMILIES].sort((a, b) => b.auc - a.auc).slice(0, 3);
  const weakest = ATTACK_FAMILIES.find((a) => a.name === "Coordinated Swarm")!;

  const columns = [
    { key: "name", label: "Attack Family", align: "left" as const },
    { key: "auc", label: "AUC", align: "right" as const },
    { key: "recall", label: "Recall", align: "right" as const },
    { key: "severity", label: "Severity", align: "center" as const },
    { key: "status", label: "Detection", align: "center" as const },
  ];

  const rows = ATTACK_FAMILIES.map((a) => ({
    name: (
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-white/[0.03] border border-white/[0.06] flex items-center justify-center shrink-0">
          <Swords className="w-3.5 h-3.5 text-slate-400" />
        </div>
        <span className="font-medium text-slate-200">{a.name}</span>
      </div>
    ),
    auc: <span className="font-semibold tabular" style={{ color: aucColor(a.auc) }}>{a.auc.toFixed(4)}</span>,
    recall: <span className="font-semibold tabular" style={{ color: recallColor(a.recall) }}>{a.recall.toFixed(4)}</span>,
    severity: (
      <StatusBadge
        variant={a.severity === "critical" ? "danger" : a.severity === "high" ? "warning" : a.severity === "medium" ? "info" : "success"}
      >
        {a.severity.toUpperCase()}
      </StatusBadge>
    ),
    status: a.auc >= 0.99 && a.recall >= 0.99 ? (
      <StatusBadge variant="success" dot>Defended</StatusBadge>
    ) : a.recall < 0.7 ? (
      <StatusBadge variant="warning" dot>Partial</StatusBadge>
    ) : (
      <StatusBadge variant="info" dot>Detected</StatusBadge>
    ),
  }));

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-3 mb-1">
          <Target className="w-5 h-5 text-cyan-400" />
          <h1 className="text-xl font-bold text-white tracking-tight">ATTACK INTELLIGENCE</h1>
        </div>
        <p className="text-sm text-slate-400">Threat family coverage and detection efficacy across the adversarial evaluation suite.</p>
      </div>

      {/* Coverage banner */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="panel p-5 md:col-span-2 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-200 mb-1">Attack Coverage</h2>
            <p className="text-xs text-slate-500">All adversarial families evaluated against v6_robust</p>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-center">
              <p className="text-3xl font-bold tabular text-emerald-400">8 / 8</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-wide mt-0.5">Families</p>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold tabular text-emerald-400">100%</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-wide mt-0.5">Coverage</p>
            </div>
          </div>
        </div>
        <div className="panel p-5 flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center shrink-0">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-200">Evaluation Passed</p>
            <p className="text-xs text-slate-500">Stage 10 generalization complete</p>
          </div>
        </div>
      </div>

      {/* Best + weakest */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="panel p-5 lg:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-semibold text-white">Best Detected Attacks</h2>
          </div>
          <div className="space-y-3">
            {best.map((a, i) => (
              <div key={a.name} className="flex items-center justify-between p-3 rounded-lg bg-emerald-500/[0.03] border border-emerald-500/10">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-bold tabular text-emerald-400 w-5">#{i + 1}</span>
                  <span className="text-sm font-semibold text-slate-200">{a.name}</span>
                </div>
                <div className="flex items-center gap-4 text-xs tabular">
                  <span className="text-slate-400">AUC <span className="text-emerald-400 font-semibold">{a.auc.toFixed(4)}</span></span>
                  <span className="text-slate-400">Recall <span className="text-emerald-400 font-semibold">{a.recall.toFixed(4)}</span></span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel p-5 border-amber-500/20">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <h2 className="text-sm font-semibold text-white">Weakest Detection</h2>
          </div>
          <div className="p-4 rounded-lg bg-amber-500/[0.04] border border-amber-500/15">
            <p className="text-sm font-bold text-amber-300">{weakest.name}</p>
            <p className="text-[11px] text-slate-500 mt-1">Hardest-to-detect family — still above operational floor.</p>
            <div className="grid grid-cols-2 gap-3 mt-4">
              <div className="text-center p-2 rounded-md bg-slate-950/40">
                <p className="text-[9px] text-slate-500 uppercase">AUC</p>
                <p className="text-lg font-bold tabular text-amber-400">{weakest.auc.toFixed(4)}</p>
              </div>
              <div className="text-center p-2 rounded-md bg-slate-950/40">
                <p className="text-[9px] text-slate-500 uppercase">Recall</p>
                <p className="text-lg font-bold tabular text-amber-400">{weakest.recall.toFixed(4)}</p>
              </div>
            </div>
            <p className="text-[10px] text-slate-500 mt-3 leading-relaxed">
              Coordinated swarm remains the most challenging pattern. The model maintains AUC above 0.90, indicating the defense holds but recall improvement is a target for future hardening.
            </p>
          </div>
        </div>
      </div>

      {/* Full table */}
      <div className="panel p-6">
        <div className="flex items-center gap-2.5 mb-5">
          <Layers className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white">All Attack Families</h2>
          {loading && <span className="text-xs text-slate-500">syncing…</span>}
        </div>
        <DataTable columns={columns} rows={rows} />
      </div>
    </div>
  );
}

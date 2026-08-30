import { useEffect, useState } from "react";
import { HeartPulse, Server, Cpu, ShieldCheck, Swords, Database, Activity, CheckCircle2, XCircle } from "lucide-react";
import StatusBadge from "@/components/StatusBadge";
import { LoadingState, ErrorState } from "@/components/LoadingState";
import { api } from "@/services/api";
import { MODEL_VERSION, THRESHOLD, FEATURE_COUNT } from "@/lib/constants";
import type { HealthResponse, ModelInfoResponse } from "@/types/api";

interface HealthRow {
  label: string;
  value: string;
  status: "ok" | "warn" | "err";
  icon: React.ReactNode;
}

export default function SystemHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, m] = await Promise.all([api.health(), api.modelInfo()]);
      setHealth(h);
      setModelInfo(m);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load system status.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const apiOnline = health?.status?.toLowerCase() === "ok" || health?.status?.toLowerCase() === "online" || health?.status?.toLowerCase() === "healthy";
  const modelLoaded = modelInfo?.model_version != null || health?.model_loaded === true;

  const rows: HealthRow[] = [
    { label: "API Status", value: apiOnline ? "ONLINE" : loading ? "CHECKING" : "OFFLINE", status: apiOnline ? "ok" : "err", icon: <Server className="w-4 h-4" /> },
    { label: "Model Status", value: modelLoaded ? "LOADED" : "NOT LOADED", status: modelLoaded ? "ok" : "err", icon: <Cpu className="w-4 h-4" /> },
    { label: "Model Version", value: modelInfo?.model_version || MODEL_VERSION, status: "ok", icon: <ShieldCheck className="w-4 h-4" /> },
    { label: "Decision Threshold", value: (modelInfo?.threshold ?? THRESHOLD).toFixed(2), status: "ok", icon: <Activity className="w-4 h-4" /> },
    { label: "Feature Schema", value: `${modelInfo?.features?.length || FEATURE_COUNT} FEATURES`, status: "ok", icon: <Database className="w-4 h-4" /> },
  ];

  const systems = [
    { name: "Backend API", detail: "FastAPI · /health", ok: apiOnline, icon: <Server className="w-5 h-5" /> },
    { name: "Detection Model", detail: modelInfo?.model_version || MODEL_VERSION, ok: modelLoaded, icon: <Cpu className="w-5 h-5" /> },
    { name: "Red-Team Engine", detail: "/simulate-attack available", ok: apiOnline, icon: <Swords className="w-5 h-5" /> },
  ];

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-3 mb-1">
          <HeartPulse className="w-5 h-5 text-cyan-400" />
          <h1 className="text-xl font-bold text-white tracking-tight">SYSTEM HEALTH</h1>
        </div>
        <p className="text-sm text-slate-400">Backend connectivity, model status, and engine availability.</p>
      </div>

      {loading && (
        <div className="panel p-6">
          <LoadingState label="Probing backend services" />
        </div>
      )}

      {error && !loading && (
        <div className="panel p-6">
          <ErrorState message={error} onRetry={load} />
        </div>
      )}

      {!loading && !error && (
        <>
          {/* Status grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {rows.map((r, i) => (
              <div
                key={r.label}
                className="animate-fade-in panel p-5"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[10px] font-semibold tracking-[0.12em] text-slate-500 uppercase">{r.label}</span>
                  <span className={`p-1.5 rounded-md ${
                    r.status === "ok" ? "bg-emerald-500/10 text-emerald-400" : r.status === "warn" ? "bg-amber-500/10 text-amber-400" : "bg-rose-500/10 text-rose-400"
                  }`}>
                    {r.icon}
                  </span>
                </div>
                <p className={`text-lg font-bold tabular tracking-tight ${
                  r.status === "ok" ? "text-white" : r.status === "warn" ? "text-amber-400" : "text-rose-400"
                }`}>
                  {r.value}
                </p>
              </div>
            ))}
          </div>

          {/* Systems */}
          <div className="panel p-6">
            <div className="flex items-center gap-2.5 mb-5">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h2 className="text-sm font-semibold text-white">Subsystem Status</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {systems.map((s) => (
                <div
                  key={s.name}
                  className={`flex items-center gap-4 p-4 rounded-lg border ${
                    s.ok ? "border-emerald-500/20 bg-emerald-500/[0.03]" : "border-rose-500/20 bg-rose-500/[0.03]"
                  }`}
                >
                  <div className={`w-11 h-11 rounded-xl flex items-center justify-center shrink-0 ${
                    s.ok ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
                  }`}>
                    {s.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-200">{s.name}</p>
                    <p className="text-xs text-slate-500 truncate">{s.detail}</p>
                  </div>
                  {s.ok ? (
                    <div className="flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      <span className="text-xs font-semibold text-emerald-400">{apiOnline ? "Connected" : "Available"}</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5">
                      <XCircle className="w-4 h-4 text-rose-400" />
                      <span className="text-xs font-semibold text-rose-400">Offline</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Raw health response */}
          <div className="panel p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-white">Raw Service Response</h2>
              <StatusBadge variant={apiOnline ? "success" : "danger"} dot pulse>
                {apiOnline ? "Healthy" : "Unreachable"}
              </StatusBadge>
            </div>
            <pre className="text-xs font-mono text-slate-400 bg-slate-950/60 rounded-lg p-4 overflow-x-auto border border-white/[0.04]">
{JSON.stringify(
  {
    health: health,
    model_info: modelInfo,
  },
  null,
  2,
)}
            </pre>
          </div>
        </>
      )}
    </div>
  );
}

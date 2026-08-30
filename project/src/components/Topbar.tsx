import { useEffect, useState } from "react";
import { Bell, Cpu, Activity } from "lucide-react";
import { useNav, type PageId } from "./NavContext";
import { MODEL_VERSION } from "@/lib/constants";
import { api } from "@/services/api";

const pageTitles: Record<PageId, { title: string; subtitle: string }> = {
  overview: { title: "Defense Center", subtitle: "Adversarial intelligence overview" },
  detector: { title: "Transaction Detector", subtitle: "Real-time fraud scoring" },
  redteam: { title: "Red-Team Simulator", subtitle: "Adversarial attack injection" },
  intelligence: { title: "Attack Intelligence", subtitle: "Threat family coverage" },
  performance: { title: "Model Performance", subtitle: "Detection metrics & evaluation" },
  explainability: { title: "Explainability", subtitle: "Global feature attribution" },
  health: { title: "System Health", subtitle: "Backend & model status" },
};

export default function Topbar() {
  const { page } = useNav();
  const { title, subtitle } = pageTitles[page];
  const [status, setStatus] = useState<"online" | "offline" | "checking">("checking");

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        await api.health();
        if (active) setStatus("online");
      } catch {
        if (active) setStatus("offline");
      }
    };
    check();
    const id = setInterval(check, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  return (
    <header className="sticky top-0 z-30 h-16 flex items-center justify-between px-6 border-b border-white/[0.06] bg-[#070b14]/80 backdrop-blur-xl">
      <div className="min-w-0">
        <h1 className="text-[15px] font-semibold text-white tracking-tight truncate">{title}</h1>
        <p className="text-xs text-slate-500 truncate hidden sm:block">{subtitle}</p>
      </div>

      <div className="flex items-center gap-3">
        {/* Model badge */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg glass text-xs">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-semibold text-slate-200 tracking-wide">{MODEL_VERSION}</span>
        </div>

        {/* Status */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg glass text-xs">
          <Activity className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400">API</span>
          <span className={`font-semibold ${status === "online" ? "text-emerald-400" : status === "offline" ? "text-rose-400" : "text-amber-400"}`}>
            {status === "online" ? "ONLINE" : status === "offline" ? "OFFLINE" : "CHECKING"}
          </span>
          <span className={`w-1.5 h-1.5 rounded-full ${status === "online" ? "bg-emerald-400 pulse-green" : status === "offline" ? "bg-rose-400" : "bg-amber-400"}`} />
        </div>

        <button className="relative p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] transition-colors">
          <Bell className="w-4.5 h-4.5" />
          <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-cyan-400" />
        </button>
      </div>
    </header>
  );
}

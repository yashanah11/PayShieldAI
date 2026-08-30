import { ShieldCheck, ChevronLeft, Radio } from "lucide-react";
import { useNav, type PageId } from "./NavContext";
import { MODEL_VERSION, THRESHOLD } from "@/lib/constants";
import {
  LayoutDashboard,
  ScanSearch,
  Swords,
  Target,
  BarChart3,
  Sparkles,
  HeartPulse,
} from "lucide-react";

const navItems: { id: PageId; label: string; icon: React.ReactNode }[] = [
  { id: "overview", label: "Overview", icon: <LayoutDashboard className="w-[18px] h-[18px]" /> },
  { id: "detector", label: "Transaction Detector", icon: <ScanSearch className="w-[18px] h-[18px]" /> },
  { id: "redteam", label: "Red-Team Simulator", icon: <Swords className="w-[18px] h-[18px]" /> },
  { id: "intelligence", label: "Attack Intelligence", icon: <Target className="w-[18px] h-[18px]" /> },
  { id: "performance", label: "Model Performance", icon: <BarChart3 className="w-[18px] h-[18px]" /> },
  { id: "explainability", label: "Explainability", icon: <Sparkles className="w-[18px] h-[18px]" /> },
  { id: "health", label: "System Health", icon: <HeartPulse className="w-[18px] h-[18px]" /> },
];

export default function Sidebar() {
  const { page, navigate, collapsed, setCollapsed } = useNav();

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex flex-col border-r border-white/[0.06] bg-[#0a0f1c]/80 backdrop-blur-xl transition-[width] duration-300 ease-out ${
        collapsed ? "w-[72px]" : "w-[248px]"
      }`}
    >
      {/* Logo */}
      <div className="h-16 flex items-center gap-3 px-5 border-b border-white/[0.06] shrink-0">
        <div className="relative shrink-0">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-glow">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>
        </div>
        {!collapsed && (
          <div className="animate-fade-in overflow-hidden">
            <div className="flex items-baseline gap-1">
              <span className="text-[15px] font-bold tracking-tight text-white">PAYSHIELD</span>
              <span className="text-[11px] font-bold tracking-wider text-cyan-400">AI</span>
            </div>
            <p className="text-[9px] tracking-[0.18em] text-slate-500 font-medium uppercase mt-0.5">
              Adversarial Fraud Defense
            </p>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const active = page === item.id;
          return (
            <button
              key={item.id}
              onClick={() => navigate(item.id)}
              title={collapsed ? item.label : undefined}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group relative ${
                active
                  ? "bg-cyan-500/10 text-cyan-300"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.03]"
              }`}
            >
              {active && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-gradient-to-b from-cyan-400 to-blue-500" />
              )}
              <span className={`shrink-0 ${active ? "text-cyan-400" : ""}`}>{item.icon}</span>
              {!collapsed && <span className="truncate">{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Footer status */}
      <div className="border-t border-white/[0.06] p-3 space-y-2 shrink-0">
        {!collapsed ? (
          <div className="space-y-1.5 px-2">
            <div className="flex items-center gap-2">
              <span className="relative flex w-2 h-2">
                <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60 animate-ping" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
              </span>
              <span className="text-[11px] font-semibold text-emerald-400 tracking-wide">SYSTEM ONLINE</span>
            </div>
            <div className="flex items-center justify-between text-[10px] text-slate-500 font-medium">
              <span className="text-slate-400 tracking-wide">{MODEL_VERSION}</span>
              <span className="tabular">{THRESHOLD.toFixed(2)} THRESHOLD</span>
            </div>
          </div>
        ) : (
          <div className="flex justify-center">
            <span className="relative flex w-2 h-2">
              <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60 animate-ping" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
            </span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center gap-1.5 py-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-white/[0.03] transition-colors text-xs"
        >
          <ChevronLeft className={`w-4 h-4 transition-transform ${collapsed ? "rotate-180" : ""}`} />
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}

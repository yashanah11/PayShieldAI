import { Swords, AlertTriangle, Info } from "lucide-react";
import type { AttackDefinition } from "@/types/api";
import { ATTACK_FAMILIES } from "@/lib/constants";

const severityStyle: Record<string, { border: string; text: string; bg: string; label: string }> = {
  low: { border: "border-emerald-500/30", text: "text-emerald-400", bg: "bg-emerald-500/5", label: "LOW" },
  medium: { border: "border-amber-500/30", text: "text-amber-400", bg: "bg-amber-500/5", label: "MEDIUM" },
  high: { border: "border-rose-500/30", text: "text-rose-400", bg: "bg-rose-500/5", label: "HIGH" },
  critical: { border: "border-fuchsia-500/40", text: "text-fuchsia-400", bg: "bg-fuchsia-500/5", label: "CRITICAL" },
};

function getSeverity(name: string): string {
  const found = ATTACK_FAMILIES.find((a) => a.name.toLowerCase() === name.toLowerCase());
  return found?.severity ?? "high";
}

interface AttackSelectorProps {
  attacks: AttackDefinition[];
  selected: string | null;
  onSelect: (name: string) => void;
  disabled?: boolean;
}

export default function AttackSelector({ attacks, selected, onSelect, disabled }: AttackSelectorProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
      {attacks.map((a) => {
        const sev = getSeverity(a.name);
        const s = severityStyle[sev] ?? severityStyle.high;
        const active = selected === a.name;
        return (
          <button
            key={a.name}
            disabled={disabled}
            onClick={() => onSelect(a.name)}
            className={`group relative text-left p-3.5 rounded-lg border transition-all duration-200 ${
              active
                ? `${s.border} ${s.bg} ring-1 ring-offset-0 ring-cyan-500/30`
                : "border-white/[0.06] bg-slate-950/40 hover:border-white/[0.12] hover:bg-white/[0.02]"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <div className="flex items-center justify-between gap-2 mb-1.5">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className={`shrink-0 p-1.5 rounded-md ${active ? s.bg + " " + s.border + " border" : "bg-white/[0.03] border border-white/[0.06]"}`}>
                  <Swords className={`w-3.5 h-3.5 ${active ? s.text : "text-slate-500"}`} />
                </div>
                <span className={`text-sm font-semibold truncate ${active ? "text-white" : "text-slate-300"}`}>{a.name}</span>
              </div>
              <span className={`shrink-0 px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider ${s.bg} ${s.text} border ${s.border}`}>
                {s.label}
              </span>
            </div>
            {a.description && (
              <p className="text-[11px] text-slate-500 leading-relaxed line-clamp-2 ml-8">{a.description}</p>
            )}
          </button>
        );
      })}
    </div>
  );
}

import type { ReactNode } from "react";

type Variant = "success" | "danger" | "warning" | "info" | "neutral";

const styles: Record<Variant, { bg: string; text: string; border: string; dot: string }> = {
  success: { bg: "bg-emerald-500/10", text: "text-emerald-400", border: "border-emerald-500/25", dot: "bg-emerald-400" },
  danger: { bg: "bg-rose-500/10", text: "text-rose-400", border: "border-rose-500/25", dot: "bg-rose-400" },
  warning: { bg: "bg-amber-500/10", text: "text-amber-400", border: "border-amber-500/25", dot: "bg-amber-400" },
  info: { bg: "bg-cyan-500/10", text: "text-cyan-400", border: "border-cyan-500/25", dot: "bg-cyan-400" },
  neutral: { bg: "bg-slate-500/10", text: "text-slate-300", border: "border-slate-500/20", dot: "bg-slate-400" },
};

export default function StatusBadge({
  variant = "neutral",
  children,
  dot = false,
  pulse = false,
  className = "",
}: {
  variant?: Variant;
  children: ReactNode;
  dot?: boolean;
  pulse?: boolean;
  className?: string;
}) {
  const s = styles[variant];
  return (
    <span className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-md text-xs font-semibold ${s.bg} ${s.text} border ${s.border} ${className}`}>
      {dot && (
        <span className={`relative flex w-1.5 h-1.5 ${pulse ? "animate-pulse" : ""}`}>
          {pulse && <span className={`absolute inline-flex h-full w-full rounded-full ${s.dot} opacity-60 animate-ping`} />}
          <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${s.dot}`} />
        </span>
      )}
      {children}
    </span>
  );
}

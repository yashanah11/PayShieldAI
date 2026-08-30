import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts";
import { ATTACK_FAMILIES } from "@/lib/constants";
import type { AttackResult } from "@/lib/constants";

const CHART_FONT = { fontSize: 11, fill: "#64748b", fontFamily: "ui-monospace, monospace" };

const tooltipStyle = {
  backgroundColor: "rgba(10,15,28,0.95)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "8px",
  fontSize: "12px",
  backdropFilter: "blur(8px)",
};

function aucColor(auc: number) {
  if (auc >= 0.99) return "#10b981";
  if (auc >= 0.95) return "#22d3ee";
  if (auc >= 0.9) return "#f59e0b";
  return "#f43f5e";
}

function recallColor(r: number) {
  if (r >= 0.95) return "#10b981";
  if (r >= 0.8) return "#22d3ee";
  if (r >= 0.6) return "#f59e0b";
  return "#f43f5e";
}

export function AucBarChart({ data = ATTACK_FAMILIES }: { data?: AttackResult[] }) {
  const chartData = data.map((d) => ({
    name: d.name.replace(/\s+/g, "\n"),
    auc: Number(d.auc.toFixed(4)),
  }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={chartData} margin={{ top: 12, right: 8, left: -10, bottom: 8 }}>
        <defs>
          <linearGradient id="auc-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.9} />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.5} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
        <XAxis dataKey="name" tick={CHART_FONT} tickLine={false} axisLine={false} interval={0} height={50} />
        <YAxis domain={[0.85, 1.01]} tick={CHART_FONT} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={{ color: "#cbd5e1", fontWeight: 600 }}
          itemStyle={{ color: "#22d3ee" }}
          cursor={{ fill: "rgba(34,211,238,0.05)" }}
        />
        <Bar dataKey="auc" radius={[4, 4, 0, 0]} maxBarSize={48}>
          {chartData.map((d, i) => (
            <Cell key={i} fill={aucColor(d.auc)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RecallBarChart({ data = ATTACK_FAMILIES }: { data?: AttackResult[] }) {
  const chartData = data.map((d) => ({
    name: d.name.replace(/\s+/g, "\n"),
    recall: Number(d.recall.toFixed(4)),
  }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={chartData} margin={{ top: 12, right: 8, left: -10, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
        <XAxis dataKey="name" tick={CHART_FONT} tickLine={false} axisLine={false} interval={0} height={50} />
        <YAxis domain={[0.5, 1.01]} tick={CHART_FONT} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={{ color: "#cbd5e1", fontWeight: 600 }}
          itemStyle={{ color: "#10b981" }}
          cursor={{ fill: "rgba(16,185,129,0.05)" }}
        />
        <Bar dataKey="recall" radius={[4, 4, 0, 0]} maxBarSize={48}>
          {chartData.map((d, i) => (
            <Cell key={i} fill={recallColor(d.recall)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DefenseRadarChart({ data = ATTACK_FAMILIES }: { data?: AttackResult[] }) {
  const chartData = data.map((d) => ({
    attack: d.name.split(" ").map((w, i) => i === 0 ? w : w.slice(0, 4) + ".").join(" "),
    AUC: Number((d.auc * 100).toFixed(1)),
    Recall: Number((d.recall * 100).toFixed(1)),
  }));
  return (
    <ResponsiveContainer width="100%" height={340}>
      <RadarChart data={chartData} margin={{ top: 12, right: 30, left: 30, bottom: 12 }}>
        <PolarGrid stroke="rgba(148,163,184,0.12)" />
        <PolarAngleAxis dataKey="attack" tick={{ ...CHART_FONT, fontSize: 10 }} />
        <PolarRadiusAxis domain={[50, 100]} tick={{ ...CHART_FONT, fontSize: 9 }} stroke="rgba(148,163,184,0.1)" />
        <Radar name="AUC" dataKey="AUC" stroke="#22d3ee" strokeWidth={2} fill="#22d3ee" fillOpacity={0.15} />
        <Radar name="Recall" dataKey="Recall" stroke="#10b981" strokeWidth={2} fill="#10b981" fillOpacity={0.1} />
        <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
        <Tooltip contentStyle={tooltipStyle} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

export { aucColor, recallColor };

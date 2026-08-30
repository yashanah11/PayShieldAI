interface DataTableProps {
  columns: { key: string; label: string; align?: "left" | "right" | "center"; width?: string }[];
  rows: Record<string, React.ReactNode>[];
}

export default function DataTable({ columns, rows }: DataTableProps) {
  return (
    <div className="overflow-x-auto rounded-lg border border-white/[0.06]">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/[0.06] bg-white/[0.02]">
            {columns.map((c) => (
              <th
                key={c.key}
                style={{ width: c.width }}
                className={`px-4 py-3 text-[11px] font-semibold tracking-[0.1em] text-slate-500 uppercase ${
                  c.align === "right" ? "text-right" : c.align === "center" ? "text-center" : "text-left"
                }`}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.04]">
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-white/[0.02] transition-colors">
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={`px-4 py-3 text-slate-300 ${
                    c.align === "right" ? "text-right tabular" : c.align === "center" ? "text-center" : "text-left"
                  }`}
                >
                  {row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

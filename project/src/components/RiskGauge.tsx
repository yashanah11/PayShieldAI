interface RiskGaugeProps {
  value: number;
  threshold?: number;
  size?: number;
}

export default function RiskGauge({
  value,
  threshold = 0.3,
  size = 220,
}: RiskGaugeProps) {

  const pct = Math.max(
    0,
    Math.min(1, value)
  );

  const radius = 80;
  const strokeWidth = 10;

  const circumference =
    2 * Math.PI * radius;

  const arcLength =
    circumference * 0.75;

  const progress =
    arcLength * pct;

  const blocked =
    pct >= threshold;

  const color =
    blocked
      ? "#f43f5e"
      : pct >= threshold * 0.6
      ? "#f59e0b"
      : "#10b981";

  const percentage =
    (pct * 100).toFixed(1);

  return (
    <div
      className="relative flex items-center justify-center"
      style={{
        width: size,
        height: size * 0.8,
      }}
    >

      <svg
        width={size}
        height={size}
        viewBox="0 0 200 200"
        className="overflow-visible"
      >

        {/* Background arc */}

        <circle
          cx="100"
          cy="100"
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${arcLength} ${circumference}`}
          transform="rotate(135 100 100)"
        />

        {/* Progress */}

        <circle
          cx="100"
          cy="100"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${progress} ${circumference}`}
          transform="rotate(135 100 100)"
          className="transition-all duration-700 ease-out"
          style={{
            filter: `drop-shadow(0 0 8px ${color})`,
          }}
        />

        {/* Threshold marker */}

        <circle
          cx={
            100 +
            radius *
              Math.cos(
                Math.PI * (0.75 + threshold * 0.75)
              )
          }
          cy={
            100 +
            radius *
              Math.sin(
                Math.PI * (0.75 + threshold * 0.75)
              )
          }
          r="3"
          fill="#ffffff"
          opacity="0.8"
        />

      </svg>


      {/* Center */}

      <div className="absolute inset-0 flex flex-col items-center justify-center pt-4">

        <span
          className="text-4xl font-bold tabular tracking-tight"
          style={{ color }}
        >
          {percentage}%
        </span>

        <span className="text-[10px] uppercase tracking-[0.15em] text-slate-500 mt-1">
          Fraud Risk
        </span>

      </div>

    </div>
  );
}
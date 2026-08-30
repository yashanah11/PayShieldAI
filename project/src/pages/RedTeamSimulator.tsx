import { useEffect, useState } from "react";
import {
  Swords,
  Zap,
  ShieldCheck,
  ShieldAlert,
  ChevronRight,
  Cpu,
  AlertCircle,
  RotateCcw,
} from "lucide-react";

import TransactionForm, {
  DEFAULT_TX,
  type FieldKey,
} from "@/components/TransactionForm";

import AttackSelector from "@/components/AttackSelector";
import FeatureComparison from "@/components/FeatureComparison";
import RiskGauge from "@/components/RiskGauge";
import StatusBadge from "@/components/StatusBadge";
import Button from "@/components/Button";
import {
  LoadingState,
  ErrorState,
  EmptyState,
} from "@/components/LoadingState";

import { api } from "@/services/api";
import { useToast } from "@/components/Toast";

import type {
  PredictRequest,
  PredictResponse,
  AttackDefinition,
  SimulateAttackResponse,
} from "@/types/api";

const FALLBACK_ATTACKS: AttackDefinition[] = [
  {
    name: "Velocity Spike",
    description: "Sudden burst of transactions in a short window.",
    severity: "HIGH",
  },
  {
    name: "Geographic Spoof",
    description: "Impossible travel distance between transactions.",
    severity: "HIGH",
  },
  {
    name: "Coordinated Swarm",
    description:
      "Many actors transacting in coordination to evade velocity limits.",
    severity: "CRITICAL",
  },
  {
    name: "High-Value Cashing",
    description: "Large amount withdrawal/cashout pattern.",
    severity: "CRITICAL",
  },
  {
    name: "Off-Hour Strike",
    description: "Transactions at unusual nighttime hours.",
    severity: "MEDIUM",
  },
  {
    name: "New Device Fraud",
    description: "Fraud from a freshly-seen device with no history.",
    severity: "HIGH",
  },
  {
    name: "Merchant Compromise",
    description: "Compromised merchant with elevated risk profile.",
    severity: "HIGH",
  },
  {
    name: "Micro-Structuring",
    description:
      "Many small transactions structured to avoid thresholds.",
    severity: "MEDIUM",
  },
];

export default function RedTeamSimulator() {
  const [tx, setTx] = useState<PredictRequest>(DEFAULT_TX);

  const [attacks, setAttacks] =
    useState<AttackDefinition[]>(FALLBACK_ATTACKS);

  const [attacksLoading, setAttacksLoading] = useState(true);

  const [selectedAttack, setSelectedAttack] =
    useState<string | null>(null);

  const [result, setResult] =
    useState<SimulateAttackResponse | null>(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const toast = useToast();

  /*
   * Load attack families from backend.
   *
   * Backend response:
   * {
   *   "attack_families": [
   *      "Velocity Spike",
   *      ...
   *   ]
   * }
   */
  useEffect(() => {
    let active = true;

    async function loadAttacks() {
      try {
        const response = await api.attacks();

        if (!active) return;

        const backendAttacks = response.attack_families || [];

        const list: AttackDefinition[] =
          backendAttacks.length > 0
            ? backendAttacks.map((name) => {
                const fallback = FALLBACK_ATTACKS.find(
                  (attack) => attack.name === name,
                );

                return (
                  fallback || {
                    name,
                    description: "Backend attack family.",
                  }
                );
              })
            : FALLBACK_ATTACKS;

        setAttacks(list);

        if (list.length > 0) {
          setSelectedAttack(list[0].name);
        }
      } catch (error) {
        if (!active) return;

        setAttacks(FALLBACK_ATTACKS);
        setSelectedAttack(FALLBACK_ATTACKS[0].name);

        toast.warning(
          "Attack library unavailable",
          "Using the built-in attack definitions.",
        );
      } finally {
        if (active) {
          setAttacksLoading(false);
        }
      }
    }

    loadAttacks();

    return () => {
      active = false;
    };
  }, [toast]);

  const update = (
    key: FieldKey,
    value: number,
  ) => {
    setTx((previous) => ({
      ...previous,
      [key]: value,
    }));
  };

  const validate = (): string | null => {
    if (
      tx.hour < 0 ||
      tx.hour > 23 ||
      !Number.isInteger(tx.hour)
    ) {
      return "Hour must be an integer 0–23.";
    }

    if (tx.amount < 0) {
      return "Amount cannot be negative.";
    }

    if (tx.velocity_1h < 0) {
      return "Velocity 1h cannot be negative.";
    }

    if (tx.velocity_24h < 0) {
      return "Velocity 24h cannot be negative.";
    }

    if (tx.device_age_days < 0) {
      return "Device age cannot be negative.";
    }

    if (tx.distance_km < 0) {
      return "Distance cannot be negative.";
    }

    if (
      tx.merchant_risk < 0 ||
      tx.merchant_risk > 1
    ) {
      return "Merchant risk must be between 0 and 1.";
    }

    return null;
  };

  const simulate = async () => {
    if (!selectedAttack) {
      toast.warning(
        "No attack selected",
        "Choose an attack family first.",
      );
      return;
    }

    const validationError = validate();

    if (validationError) {
      toast.warning(
        "Invalid transaction",
        validationError,
      );
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      /*
       * IMPORTANT:
       *
       * Backend expects:
       *
       * {
       *   transaction: {...},
       *   attack_family: "Velocity Spike"
       * }
       */
      const response = await api.simulateAttack({
        transaction: tx,
        attack_family: selectedAttack,
      });

      setResult(response);

      const decision =
        response.v6_prediction.decision;

      if (decision === "BLOCK") {
        toast.success(
          "Attack defended",
          `${selectedAttack} was detected and blocked.`,
        );
      } else {
        toast.error(
          "Attack bypassed",
          `${selectedAttack} was not blocked by the detector.`,
        );
      }
    } catch (e) {
      const message =
        e instanceof Error
          ? e.message
          : "Unknown simulation error.";

      setError(message);

      toast.error(
        "Simulation failed",
        message,
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
    setTx(DEFAULT_TX);
  };

  const originalPrediction: PredictResponse | null =
    result
      ? {
          fraud_probability: 0,
          risk_classification: "LOW",
          decision: "ALLOW",
          input_transaction: result.original_features,
        }
      : null;

  const attackedPrediction =
    result?.v6_prediction || null;

  const probDelta =
    originalPrediction && attackedPrediction
      ? attackedPrediction.fraud_probability -
        originalPrediction.fraud_probability
      : 0;

  const attackedBlocked =
    attackedPrediction?.decision === "BLOCK";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <Swords className="w-5 h-5 text-cyan-400" />

          <h1 className="text-xl font-bold text-white tracking-tight">
            RED-TEAM ATTACK SIMULATOR
          </h1>
        </div>

        <p className="text-sm text-slate-400">
          Inject controlled adversarial patterns and observe how the
          v6 robust defense responds.
        </p>
      </div>

      {/* Input + Attack selection */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Transaction */}
        <div className="panel p-6">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-sm font-semibold text-slate-200">
              Base Transaction
            </h2>

            <StatusBadge variant="info" dot>
              Benign Seed
            </StatusBadge>
          </div>

          <TransactionForm
            values={tx}
            onChange={update}
            disabled={loading}
          />
        </div>

        {/* Attack selector */}
        <div className="panel p-6">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-sm font-semibold text-slate-200">
              Select Attack Vector
            </h2>

            <span className="text-[10px] text-slate-500">
              {attacks.length} FAMILIES
            </span>
          </div>

          {attacksLoading ? (
            <LoadingState label="Loading attack library" />
          ) : (
            <AttackSelector
              attacks={attacks}
              selected={selectedAttack}
              onSelect={setSelectedAttack}
              disabled={loading}
            />
          )}

          <div className="mt-5 pt-5 border-t border-white/[0.06]">
            <Button
              onClick={simulate}
              loading={loading}
              size="lg"
              className="w-full"
            >
              <Zap className="w-4 h-4" />
              Simulate Attack
            </Button>

            {result && (
              <button
                onClick={reset}
                className="mt-3 mx-auto block text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1.5"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Run new simulation
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="panel p-6">
          <LoadingState label="Injecting adversarial perturbations" />
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="panel p-6">
          <ErrorState
            message={error}
            onRetry={simulate}
          />
        </div>
      )}

      {/* Empty */}
      {!loading && !error && !result && (
        <div className="panel p-8">
          <EmptyState
            icon={<Swords className="w-10 h-10" />}
            title="No simulation yet"
            subtitle="Configure a base transaction and attack vector, then run the simulator to see the before/after defense response."
          />
        </div>
      )}

      {/* Result */}
      {!loading && !error && result && (
        <div className="space-y-6 animate-fade-in">
          {/* Attack Flow */}
          <div className="panel p-6">
            <div className="flex items-center gap-2.5 mb-5">
              <AlertCircle className="w-4 h-4 text-amber-400" />

              <h2 className="text-sm font-semibold text-white">
                Attack Flow — {selectedAttack}
              </h2>

              <StatusBadge
                variant="warning"
                className="ml-auto"
              >
                Adversarial Injection
              </StatusBadge>
            </div>

            <div className="flex items-center justify-between gap-2 overflow-x-auto pb-2">
              <FlowStep
                icon={<Zap className="w-4 h-4" />}
                label="Attack Injected"
                tone="amber"
              />

              <FlowArrow />

              <FlowStep
                icon={<Cpu className="w-4 h-4" />}
                label="v6 Robust Detector"
                tone="cyan"
              />

              <FlowArrow />

              <FlowStep
                icon={<GaugeMini />}
                label={`Fraud Prob ${result.v6_prediction.fraud_probability.toFixed(4)}`}
                tone={attackedBlocked ? "rose" : "emerald"}
              />

              <FlowArrow />

              <FlowStep
                icon={<ShieldAlert className="w-4 h-4" />}
                label={result.v6_prediction.risk_classification}
                tone={attackedBlocked ? "rose" : "emerald"}
              />

              <FlowArrow />

              <FlowStep
                icon={
                  attackedBlocked ? (
                    <ShieldAlert className="w-4 h-4" />
                  ) : (
                    <ShieldCheck className="w-4 h-4" />
                  )
                }
                label={result.v6_prediction.decision}
                tone={attackedBlocked ? "rose" : "emerald"}
                large
              />
            </div>
          </div>

          {/* Feature comparison */}
          <div className="panel p-6">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-sm font-semibold text-white">
                Feature Perturbation
              </h2>

              <span className="text-xs text-slate-500">
                {result.changed_features.length} feature(s) modified
              </span>
            </div>

            <FeatureComparison
              original={result.original_features}
              attacked={result.attacked_features}
              changedFeatures={result.changed_features}
            />

            {result.changed_features.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                <span className="text-[10px] text-slate-500 tracking-wide uppercase mr-1 self-center">
                  Changed:
                </span>

                {result.changed_features.map((feature) => (
                  <span
                    key={feature}
                    className="px-2 py-1 rounded-md text-[11px] font-mono font-medium bg-amber-500/10 text-amber-300 border border-amber-500/20"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Predictions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {originalPrediction && (
              <PredictionCard
                title="Before Attack"
                prediction={originalPrediction}
                dim
              />
            )}

            <PredictionCard
              title="After Attack"
              prediction={result.v6_prediction}
              highlight
              probDelta={probDelta}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function FlowStep({
  icon,
  label,
  tone,
  large,
}: {
  icon: React.ReactNode;
  label: string;
  tone: "amber" | "cyan" | "rose" | "emerald";
  large?: boolean;
}) {
  const toneMap = {
    amber:
      "bg-amber-500/10 border-amber-500/25 text-amber-400",
    cyan:
      "bg-cyan-500/10 border-cyan-500/25 text-cyan-400",
    rose:
      "bg-rose-500/10 border-rose-500/25 text-rose-400",
    emerald:
      "bg-emerald-500/10 border-emerald-500/25 text-emerald-400",
  };

  return (
    <div className="flex flex-col items-center shrink-0">
      <div
        className={`${
          large ? "w-14 h-14" : "w-11 h-11"
        } rounded-xl border flex items-center justify-center ${toneMap[tone]}`}
      >
        {icon}
      </div>

      <span
        className={`mt-2 ${
          large
            ? "text-xs font-bold"
            : "text-[10px] font-semibold"
        } text-slate-400 uppercase tracking-wide text-center max-w-[90px]`}
      >
        {label}
      </span>
    </div>
  );
}

function FlowArrow() {
  return (
    <div className="flex items-center shrink-0 -mt-5">
      <ChevronRight className="w-5 h-5 text-slate-700" />
    </div>
  );
}

function GaugeMini() {
  return (
    <svg
      className="w-4 h-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        d="M12 14l3-3M3 18a9 9 0 1118 0"
      />
    </svg>
  );
}

function PredictionCard({
  title,
  prediction,
  dim,
  highlight,
  probDelta,
}: {
  title: string;
  prediction: PredictResponse;
  dim?: boolean;
  highlight?: boolean;
  probDelta?: number;
}) {
  const blocked =
    prediction.decision === "BLOCK";

  return (
    <div
      className={`panel p-6 relative overflow-hidden ${
        highlight
          ? blocked
            ? "border-rose-500/30"
            : "border-emerald-500/30"
          : ""
      } ${dim ? "opacity-80" : ""}`}
    >
      {highlight && (
        <div
          className={`absolute -top-16 -right-16 w-40 h-40 rounded-full blur-3xl ${
            blocked
              ? "bg-rose-500/10"
              : "bg-emerald-500/10"
          }`}
        />
      )}

      <div className="flex items-center justify-between mb-4 relative">
        <h3 className="text-sm font-semibold text-slate-200">
          {title}
        </h3>

        {highlight &&
          probDelta !== undefined && (
            <span
              className={`text-xs font-semibold tabular ${
                probDelta >= 0
                  ? "text-rose-400"
                  : "text-emerald-400"
              }`}
            >
              Δ {probDelta >= 0 ? "+" : ""}
              {probDelta.toFixed(4)}
            </span>
          )}
      </div>

      <div className="flex flex-col items-center relative">
        <RiskGauge
          value={prediction.fraud_probability}
          threshold={0.3}
          size={180}
        />

        <div
          className={`mt-4 px-4 py-2 rounded-lg border w-full text-center ${
            blocked
              ? "border-rose-500/30 bg-rose-500/[0.06]"
              : "border-emerald-500/30 bg-emerald-500/[0.06]"
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            {blocked ? (
              <ShieldAlert className="w-4 h-4 text-rose-400" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            )}

            <span
              className={`text-lg font-bold ${
                blocked
                  ? "text-rose-400"
                  : "text-emerald-400"
              }`}
            >
              {prediction.decision}
            </span>
          </div>

          <p className="text-[10px] text-slate-500 mt-1">
            Risk: {prediction.risk_classification} ·
            Threshold 0.30
          </p>
        </div>
      </div>
    </div>
  );
}
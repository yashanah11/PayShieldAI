import { useState } from "react";
import {
  ScanSearch,
  RotateCcw,
  ShieldCheck,
  ShieldAlert,
  Gauge,
  Activity,
  Zap,
} from "lucide-react";

import TransactionForm, {
  DEFAULT_TX,
  type FieldKey,
} from "@/components/TransactionForm";

import RiskGauge from "@/components/RiskGauge";
import Button from "@/components/Button";
import { LoadingState, ErrorState } from "@/components/LoadingState";
import { api } from "@/services/api";
import { useToast } from "@/components/Toast";
import type { PredictRequest, PredictResponse } from "@/types/api";
import { MODEL_VERSION, THRESHOLD } from "@/lib/constants";

export default function TransactionDetector() {
  const [tx, setTx] = useState<PredictRequest>(DEFAULT_TX);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toast = useToast();

  const update = (key: FieldKey, value: number) => {
    setTx((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const validate = (): string | null => {
    if (
      tx.hour < 0 ||
      tx.hour > 23 ||
      !Number.isInteger(tx.hour)
    ) {
      return "Hour must be an integer between 0 and 23.";
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

  const analyze = async () => {
    const validationError = validate();

    if (validationError) {
      toast.warning("Invalid input", validationError);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log("Sending transaction:", tx);

      const response = await api.predict(tx);

      console.log("Backend response:", response);

      setResult(response);

      toast.success(
        "Analysis complete",
        `Fraud probability: ${(response.fraud_probability * 100).toFixed(2)}%`
      );
    } catch (e) {
      console.error("Prediction error:", e);

      const message =
        e instanceof Error
          ? e.message
          : "Unable to analyze transaction.";

      setError(message);

      toast.error(
        "Detection failed",
        message
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
    setTx({ ...DEFAULT_TX });
  };

  const riskClass = result?.risk_classification;

  const blocked =
    result?.decision === "BLOCK";

  const probability =
    result?.fraud_probability ?? 0;

  const probabilityPercent =
    probability * 100;

  const getRiskColor = () => {
    if (riskClass === "HIGH") {
      return "text-rose-400";
    }

    if (riskClass === "MEDIUM") {
      return "text-amber-400";
    }

    return "text-emerald-400";
  };

  const getRiskBackground = () => {
    if (riskClass === "HIGH") {
      return "border-rose-500/30 bg-rose-500/[0.06]";
    }

    if (riskClass === "MEDIUM") {
      return "border-amber-500/30 bg-amber-500/[0.06]";
    }

    return "border-emerald-500/30 bg-emerald-500/[0.06]";
  };

  return (
    <div className="space-y-6">

      {/* ========================================= */}
      {/* PAGE HEADER */}
      {/* ========================================= */}

      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">

        <div>
          <div className="flex items-center gap-3 mb-2">

            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
              <ScanSearch className="w-5 h-5 text-cyan-400" />
            </div>

            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">
                TRANSACTION DETECTOR
              </h1>

              <p className="text-xs text-slate-500 mt-0.5">
                Real-time fraud risk analysis
              </p>
            </div>

          </div>

          <p className="text-sm text-slate-400">
            Analyze payment behavior using the{" "}
            <span className="text-cyan-400 font-semibold">
              {MODEL_VERSION}
            </span>{" "}
            fraud detection model.
          </p>
        </div>

        {/* Model status */}

        <div className="flex items-center gap-3">

          <div className="flex items-center gap-2 px-3 py-2 rounded-lg border border-emerald-500/20 bg-emerald-500/[0.05]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
            </span>

            <span className="text-xs font-medium text-emerald-400">
              MODEL ONLINE
            </span>
          </div>

          <div className="px-3 py-2 rounded-lg border border-white/[0.06] bg-white/[0.02]">
            <span className="text-xs text-slate-400">
              Threshold{" "}
              <span className="font-bold text-white">
                {THRESHOLD.toFixed(2)}
              </span>
            </span>
          </div>

        </div>

      </div>


      {/* ========================================= */}
      {/* MAIN CONTENT */}
      {/* ========================================= */}

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">

        {/* ======================================= */}
        {/* INPUT PANEL */}
        {/* ======================================= */}

        <div className="xl:col-span-2 panel p-6">

          <div className="flex items-center justify-between mb-6">

            <div>
              <h2 className="text-sm font-semibold text-white">
                Transaction Input
              </h2>

              <p className="text-[11px] text-slate-500 mt-1">
                Configure the seven transaction features
              </p>
            </div>

            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-cyan-500/10 border border-cyan-500/10">

              <Activity className="w-3 h-3 text-cyan-400" />

              <span className="text-[9px] font-bold tracking-wider text-cyan-400">
                7 FEATURES
              </span>

            </div>

          </div>


          {/* Form */}

          <TransactionForm
            values={tx}
            onChange={update}
            disabled={loading}
          />


          {/* Buttons */}

          <div className="mt-7 flex gap-3">

            <Button
              onClick={analyze}
              loading={loading}
              className="flex-1"
              size="lg"
            >
              {!loading && (
                <ScanSearch className="w-4 h-4" />
              )}

              {loading
                ? "Analyzing..."
                : "Analyze Transaction"}
            </Button>

            <Button
              variant="secondary"
              onClick={reset}
              size="lg"
              disabled={loading}
            >
              <RotateCcw className="w-4 h-4" />
            </Button>

          </div>


          {/* API indicator */}

          <div className="mt-5 pt-4 border-t border-white/[0.05] flex items-center justify-between">

            <div className="flex items-center gap-2">

              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />

              <span className="text-[10px] text-slate-500">
                FastAPI backend connected
              </span>

            </div>

            <span className="text-[10px] font-mono text-slate-600">
              POST /predict
            </span>

          </div>

        </div>


        {/* ======================================= */}
        {/* RESULT PANEL */}
        {/* ======================================= */}

        <div className="xl:col-span-3 panel p-6 min-h-[520px] flex flex-col">

          {/* Result header */}

          <div className="flex items-center justify-between mb-6">

            <div>
              <h2 className="text-sm font-semibold text-white">
                Detection Result
              </h2>

              <p className="text-[11px] text-slate-500 mt-1">
                AI-powered transaction risk assessment
              </p>
            </div>

            <div className="flex items-center gap-2">

              <Zap className="w-3.5 h-3.5 text-cyan-400" />

              <span className="text-[10px] text-slate-500">
                LIVE ANALYSIS
              </span>

            </div>

          </div>


          {/* =================================== */}
          {/* LOADING */}
          {/* =================================== */}

          {loading && (
            <div className="flex-1 flex flex-col items-center justify-center">

              <LoadingState
                label="Running fraud detection model"
                className="w-full"
              />

              <p className="text-[10px] text-slate-600 mt-4">
                Sending transaction to {MODEL_VERSION}
              </p>

            </div>
          )}


          {/* =================================== */}
          {/* ERROR */}
          {/* =================================== */}

          {!loading && error && (
            <div className="flex-1 flex flex-col items-center justify-center">

              <ErrorState
                message={error}
                onRetry={analyze}
              />

              <button
                onClick={reset}
                className="mt-4 text-xs text-slate-500 hover:text-slate-300 transition-colors"
              >
                Reset transaction
              </button>

            </div>
          )}


          {/* =================================== */}
          {/* EMPTY STATE */}
          {/* =================================== */}

          {!loading &&
            !error &&
            !result && (
              <div className="flex-1 flex flex-col items-center justify-center text-center">

                <div className="relative mb-6">

                  <div className="absolute inset-0 rounded-3xl bg-cyan-500/10 blur-xl" />

                  <div className="relative w-20 h-20 rounded-2xl bg-white/[0.03] border border-white/[0.07] flex items-center justify-center">

                    <Gauge className="w-9 h-9 text-slate-600" />

                  </div>

                </div>

                <p className="text-sm font-semibold text-slate-400">
                  Awaiting transaction
                </p>

                <p className="text-xs text-slate-600 mt-2 max-w-sm leading-relaxed">
                  Configure the transaction features on the left
                  and run the detector to calculate the fraud risk.
                </p>

                <div className="mt-6 flex items-center gap-2 text-[10px] text-slate-600">

                  <div className="w-1.5 h-1.5 rounded-full bg-slate-600" />

                  READY FOR ANALYSIS

                </div>

              </div>
            )}


          {/* =================================== */}
          {/* RESULT */}
          {/* =================================== */}

          {!loading &&
            !error &&
            result && (

              <div className="flex-1 flex flex-col">

                {/* Gauge */}

                <div className="flex flex-col items-center">

                  <RiskGauge
                    value={probability}
                    threshold={THRESHOLD}
                  />

                  <div className="mt-1 text-center">

                    <p className="text-[10px] tracking-[0.15em] text-slate-500 uppercase">
                      Fraud Probability
                    </p>

                    <p className={`text-2xl font-bold tabular mt-1 ${getRiskColor()}`}>
                      {probabilityPercent.toFixed(2)}%
                    </p>

                  </div>

                </div>


                {/* Risk + Decision */}

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-6">

                  {/* Risk */}

                  <div className={`rounded-xl border p-4 ${getRiskBackground()}`}>

                    <p className="text-[9px] tracking-[0.14em] text-slate-500 uppercase mb-3">
                      Risk Classification
                    </p>

                    <div className="flex items-center justify-between">

                      <span className={`text-xl font-bold ${getRiskColor()}`}>
                        {riskClass}
                      </span>

                      <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                        riskClass === "HIGH"
                          ? "bg-rose-500/10"
                          : riskClass === "MEDIUM"
                          ? "bg-amber-500/10"
                          : "bg-emerald-500/10"
                      }`}>

                        <Activity className={`w-4 h-4 ${getRiskColor()}`} />

                      </div>

                    </div>

                  </div>


                  {/* Decision */}

                  <div
                    className={`rounded-xl border p-4 ${
                      blocked
                        ? "border-rose-500/30 bg-rose-500/[0.06]"
                        : "border-emerald-500/30 bg-emerald-500/[0.06]"
                    }`}
                  >

                    <p className="text-[9px] tracking-[0.14em] text-slate-500 uppercase mb-3">
                      Security Decision
                    </p>

                    <div className="flex items-center justify-between">

                      <div className="flex items-center gap-2">

                        {blocked ? (
                          <ShieldAlert className="w-5 h-5 text-rose-400" />
                        ) : (
                          <ShieldCheck className="w-5 h-5 text-emerald-400" />
                        )}

                        <span
                          className={`text-xl font-bold ${
                            blocked
                              ? "text-rose-400"
                              : "text-emerald-400"
                          }`}
                        >
                          {result.decision}
                        </span>

                      </div>

                      <span
                        className={`text-[9px] font-bold tracking-wider px-2 py-1 rounded ${
                          blocked
                            ? "bg-rose-500/10 text-rose-400"
                            : "bg-emerald-500/10 text-emerald-400"
                        }`}
                      >
                        {blocked ? "BLOCKED" : "SAFE"}
                      </span>

                    </div>

                  </div>

                </div>


                {/* Metrics */}

                <div className="grid grid-cols-3 gap-3 mt-4">

                  <div className="rounded-lg border border-white/[0.06] bg-slate-950/40 p-3">

                    <p className="text-[9px] tracking-[0.1em] text-slate-500 uppercase">
                      Probability
                    </p>

                    <p className="text-sm font-bold tabular text-white mt-1">
                      {result.fraud_probability.toFixed(4)}
                    </p>

                  </div>


                  <div className="rounded-lg border border-white/[0.06] bg-slate-950/40 p-3">

                    <p className="text-[9px] tracking-[0.1em] text-slate-500 uppercase">
                      Threshold
                    </p>

                    <p className="text-sm font-bold tabular text-white mt-1">
                      {THRESHOLD.toFixed(2)}
                    </p>

                  </div>


                  <div className="rounded-lg border border-white/[0.06] bg-slate-950/40 p-3">

                    <p className="text-[9px] tracking-[0.1em] text-slate-500 uppercase">
                      Model
                    </p>

                    <p className="text-sm font-bold text-cyan-400 mt-1 truncate">
                      {MODEL_VERSION}
                    </p>

                  </div>

                </div>


                {/* Decision explanation */}

                <div className="mt-4 p-3 rounded-lg border border-white/[0.05] bg-white/[0.015]">

                  <div className="flex items-start gap-2">

                    {blocked ? (
                      <ShieldAlert className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
                    ) : (
                      <ShieldCheck className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
                    )}

                    <p className="text-[11px] leading-relaxed text-slate-500">

                      {blocked
                        ? `The model classified this transaction as high risk. The fraud probability of ${probabilityPercent.toFixed(
                            2
                          )}% meets or exceeds the ${THRESHOLD.toFixed(
                            2
                          )} decision threshold.`
                        : `The model classified this transaction as ${riskClass?.toLowerCase()} risk. The fraud probability of ${probabilityPercent.toFixed(
                            2
                          )}% is below the ${THRESHOLD.toFixed(
                            2
                          )} decision threshold.`}

                    </p>

                  </div>

                </div>


                {/* Footer */}

                <div className="mt-auto pt-5 flex items-center justify-between">

                  <div className="flex items-center gap-2">

                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />

                    <span className="text-[10px] text-slate-600">
                      Prediction completed successfully
                    </span>

                  </div>

                  <button
                    onClick={reset}
                    className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1.5"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Analyze another
                  </button>

                </div>

              </div>
            )}

        </div>

      </div>

    </div>
  );
}
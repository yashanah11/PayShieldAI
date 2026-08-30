import { useEffect, useState } from "react";
import { Sparkles, Cpu, AlertCircle, RefreshCw } from "lucide-react";
import FeatureImportanceChart from "@/components/FeatureImportanceChart";
import StatusBadge from "@/components/StatusBadge";
import { LoadingState, ErrorState } from "@/components/LoadingState";
import { api } from "@/services/api";
import { MODEL_VERSION } from "@/lib/constants";
import TransactionForm, {
  DEFAULT_TX,
  type FieldKey,
} from "@/components/TransactionForm";
import type { FeatureImportance, PredictRequest } from "@/types/api";

export default function Explainability() {
  const [tx, setTx] = useState<PredictRequest>(DEFAULT_TX);
  const [data, setData] = useState<FeatureImportance[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const update = (key: FieldKey, value: number) => {
    setTx((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const load = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await api.explain(tx);

      const importance = res.global_feature_importance;

      if (!importance || typeof importance !== "object") {
        throw new Error(
          "Backend returned invalid feature importance data.",
        );
      }

      const formattedData: FeatureImportance[] = Object.entries(
        importance,
      )
        .map(([feature, value]) => ({
          feature,
          importance: Number(value),
        }))
        .filter((item) => Number.isFinite(item.importance))
        .sort((a, b) => b.importance - a.importance);

      if (formattedData.length === 0) {
        throw new Error(
          "Backend returned no feature importance values.",
        );
      }

      setData(formattedData);
    } catch (e) {
      setData(null);
      setError(
        e instanceof Error
          ? e.message
          : "Failed to load feature importance.",
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <Sparkles className="w-5 h-5 text-cyan-400" />

          <h1 className="text-xl font-bold text-white tracking-tight">
            MODEL EXPLAINABILITY
          </h1>
        </div>

        <p className="text-sm text-slate-400">
          Understand which transaction features influence the detector.
        </p>
      </div>

      {/* Transaction Input */}
      <div className="panel p-6">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2.5">
            <Cpu className="w-4 h-4 text-cyan-400" />

            <h2 className="text-sm font-semibold text-white">
              Transaction Features
            </h2>
          </div>

          <StatusBadge variant="neutral">
            7 features
          </StatusBadge>
        </div>

        <TransactionForm
          values={tx}
          onChange={update}
          disabled={loading}
        />

        <div className="mt-5 flex justify-end">
          <button
            type="button"
            onClick={load}
            disabled={loading}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-semibold hover:bg-cyan-500/15 hover:text-cyan-200 disabled:opacity-50 transition-all"
          >
            <RefreshCw
              className={`w-4 h-4 ${
                loading ? "animate-spin" : ""
              }`}
            />

            {loading ? "Computing..." : "Analyze Features"}
          </button>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="panel p-6">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-5">
          <div className="flex items-center gap-2.5">
            <Sparkles className="w-4 h-4 text-cyan-400" />

            <h2 className="text-sm font-semibold text-white">
              Global Feature Attribution
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <StatusBadge variant="info">
              {MODEL_VERSION}
            </StatusBadge>

            <StatusBadge variant="neutral">
              {data?.length ?? 7} features
            </StatusBadge>
          </div>
        </div>

        {loading && (
          <LoadingState
            label="Computing feature importance"
          />
        )}

        {!loading && error && (
          <ErrorState
            message={error}
            onRetry={load}
          />
        )}

        {!loading && !error && data && data.length > 0 && (
          <FeatureImportanceChart data={data} />
        )}

        {!loading && !error && (!data || data.length === 0) && (
          <div className="py-12 text-center">
            <Cpu className="w-8 h-8 mx-auto text-slate-600 mb-3" />

            <p className="text-sm font-semibold text-slate-400">
              No feature importance available
            </p>
          </div>
        )}

        {/* Explanation */}
        <div className="mt-6 flex items-start gap-3 p-4 rounded-lg bg-amber-500/[0.04] border border-amber-500/15">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />

          <div>
            <p className="text-xs font-semibold text-amber-300">
              Important Note
            </p>

            <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
              This chart displays{" "}
              <span className="font-semibold text-slate-200">
                GLOBAL MODEL FEATURE IMPORTANCE
              </span>{" "}
              returned by the v6_robust detector.
              It represents the aggregate importance of each
              transaction feature to the model.
            </p>

            <p className="text-[11px] text-slate-500 mt-2 leading-relaxed">
              This is{" "}
              <span className="font-semibold text-slate-300">
                not
              </span>{" "}
              a SHAP analysis and{" "}
              <span className="font-semibold text-slate-300">
                not
              </span>{" "}
              a per-transaction SHAP explanation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
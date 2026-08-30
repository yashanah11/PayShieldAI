import type {
  PredictRequest,
  PredictResponse,
  BatchPredictRequest,
  BatchPredictResponse,
  AttacksResponse,
  SimulateAttackRequest,
  SimulateAttackResponse,
  ModelInfoResponse,
  ExplainRequest,
  ExplainResponse,
  HealthResponse,
  DemoResponse,
} from "@/types/api";

const BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let res: Response;

  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
  } catch {
    throw new ApiError(
      "Cannot reach backend server. Make sure FastAPI is running on port 8000.",
      0,
    );
  }

  const body = await res.json().catch(() => null);

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;

    // FastAPI validation errors
    if (Array.isArray(body?.detail)) {
      detail = body.detail
        .map((item: unknown) => {
          if (
            typeof item === "object" &&
            item !== null &&
            "msg" in item
          ) {
            return String((item as { msg?: unknown }).msg ?? "Validation error");
          }

          return JSON.stringify(item);
        })
        .join(", ");
    }
    // Normal FastAPI error
    else if (typeof body?.detail === "string") {
      detail = body.detail;
    }
    // Other API error format
    else if (typeof body?.message === "string") {
      detail = body.message;
    }

    throw new ApiError(detail, res.status);
  }

  return body as T;
}

export const api = {
  health: () =>
    request<HealthResponse>("/health"),

  predict: (tx: PredictRequest) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(tx),
    }),

  batchPredict: (data: BatchPredictRequest) =>
    request<BatchPredictResponse>("/batch-predict", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  attacks: () =>
    request<AttacksResponse>("/attacks"),

  simulateAttack: (data: SimulateAttackRequest) =>
    request<SimulateAttackResponse>("/simulate-attack", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  modelInfo: () =>
    request<ModelInfoResponse>("/model-info"),

  explain: (data: ExplainRequest) =>
    request<ExplainResponse>("/explain", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  demo: () =>
    request<DemoResponse>("/demo", {
      method: "POST",
      body: JSON.stringify({}),
    }),
};

export { ApiError, BASE_URL };
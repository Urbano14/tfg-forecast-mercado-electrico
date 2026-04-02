import { apiFetch } from "./client";

export interface ModelMetric {
  id: string;
  name: string;
  type: string;
  mae: number;
  rmse: number;
}

export interface MetricsResponse {
  metrics: ModelMetric[];
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  return apiFetch<MetricsResponse>("/metrics");
}

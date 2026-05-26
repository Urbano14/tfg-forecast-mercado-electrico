import { apiFetch } from "./client";

// Métricas visibles de un modelo.
export interface ModelMetric {
  id: string;
  name: string;
  type: string;
  mae: number;
  rmse: number;
}

// El backend devuelve las métricas dentro de la propiedad "metrics".
export interface MetricsResponse {
  metrics: ModelMetric[];
}

// Consulta los valores de métricas expuestos por el backend.
export async function fetchMetrics(): Promise<MetricsResponse> {
  return apiFetch<MetricsResponse>("/metrics");
}

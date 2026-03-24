import { apiFetch } from "./client";

export interface ForecastPoint {
  timestamp: string;
  value: number;
}

export interface ForecastResponse {
  model: string;
  model_type: string;
  requested_date: string;
  horizon_hours: number;
  forecast: ForecastPoint[];
}

export async function fetchForecast(
  date: string,
  model: string
): Promise<ForecastResponse> {
  const params = new URLSearchParams({ date, model });
  return apiFetch<ForecastResponse>(`/forecast?${params.toString()}`);
}
import { apiFetch } from "./client";

export interface HistoricalRangeResponse {
  start: string;
  end: string;
}

export interface HistoricalDataPoint {
  timestamp: string;
  price: number;
  demand_forecast: number | null;
  wind_forecast: number | null;
  solar_forecast: number | null;
  hydro_programmed: number | null;
}

export async function fetchHistoricalRange(): Promise<HistoricalRangeResponse> {
  return apiFetch<HistoricalRangeResponse>("/historical/range");
}

export async function fetchHistoricalData(
  start: string,
  end: string,
  limit?: number
): Promise<HistoricalDataPoint[]> {
  const params = new URLSearchParams({ start, end });

  if (limit !== undefined) {
    params.append("limit", String(limit));
  }

  return apiFetch<HistoricalDataPoint[]>(`/historical?${params.toString()}`);
}
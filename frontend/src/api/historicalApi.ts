import { apiFetch } from "./client";

// Respuesta del endpoint /historical/range, indica el primer y último timestamp disponible en la base de datos.
export interface HistoricalRangeResponse {
  start: string;
  end: string;
}

// Cada punto representa una hora del mercado eléctrico con precio y variables exógenas.
export interface HistoricalDataPoint {
  timestamp: string;
  price: number;
  demand_forecast: number | null;
  wind_forecast: number | null;
  solar_forecast: number | null;
  hydro_programmed: number | null;
}

// Obtiene el rango completo de fechas disponibles.
export async function fetchHistoricalRange(): Promise<HistoricalRangeResponse> {
  return apiFetch<HistoricalRangeResponse>("/historical/range");
}

// Obtiene datos históricos entre dos fechas.
export async function fetchHistoricalData(
  start: string,
  end: string,
  limit?: number
): Promise<HistoricalDataPoint[]> {
  // Construye los parámetros de la URL de forma segura, si se indica un límite, se añade a la petición.

  const params = new URLSearchParams({ start, end });

  if (limit !== undefined) {
    params.append("limit", String(limit));
  }

  // Llama al endpoint de histórico y devuelve una lista de puntos horarios.
  return apiFetch<HistoricalDataPoint[]>(`/historical?${params.toString()}`);
}

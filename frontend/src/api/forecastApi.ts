import { apiFetch } from "./client";

// Punto individual de predicción devuelto por el backend.
// Cada punto tiene una hora futura y el valor predicho para esa hora.
export interface ForecastPoint {
  timestamp: string;
  value: number;
}

// Representa la respuesta completa del endpoint /forecast.
// Incluye información del modelo usado.
export interface ForecastResponse {
  model: string;
  model_type: string;
  requested_date: string;
  horizon_hours: number;
  forecast: ForecastPoint[];
}

// Llama al endpoint de predicción del backend.
// Recibe la fecha base de predicción y el identificador del modelo seleccionado en el frontend:
export async function fetchForecast(
  date: string,
  model: string
): Promise<ForecastResponse> {
  // Construye los parámetros de la URL de forma segura.
  const params = new URLSearchParams({ date, model });

  // Usa el cliente común apiFetch para llamar al backend.
  return apiFetch<ForecastResponse>(`/forecast?${params.toString()}`);
}

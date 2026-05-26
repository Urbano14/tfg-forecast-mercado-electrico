import { apiFetch } from "./client";

// Información de un modelo disponible en la aplicación.
export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  horizon_hours: number;
}

// Respuesta completa del endpoint /models, el backend devuelve la lista dentro de la propiedad "models".
export interface ModelsResponse {
  models: ModelInfo[];
}

// Obtiene del backend el catálogo de modelos disponibles.
export async function fetchModels(): Promise<ModelsResponse> {
  return apiFetch<ModelsResponse>("/models");
}

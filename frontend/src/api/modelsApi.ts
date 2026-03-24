import { apiFetch } from "./client";

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  horizon_hours: number;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export async function fetchModels(): Promise<ModelsResponse> {
  return apiFetch<ModelsResponse>("/models");
}
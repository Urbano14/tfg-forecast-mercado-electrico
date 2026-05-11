const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/v1";

const API_ERROR_TRANSLATIONS: Record<string, string> = {
  "start must be earlier than end": "La fecha de inicio debe ser anterior a la fecha de fin.",
  "Date must be aligned to full hour (e.g., 2022-01-01T00:00:00)":
    "La fecha debe estar alineada a una hora completa, por ejemplo 2022-01-01T00:00:00.",
  "No historical data available": "No hay datos historicos disponibles.",
  "Requested date is too early": "La fecha solicitada es demasiado temprana.",
  "Requested date is beyond available data":
    "La fecha solicitada esta fuera del rango de datos disponible.",
  "Not enough historical data: need previous 24 hours":
    "No hay suficientes datos historicos: se necesitan las 24 horas anteriores.",
  "Missing lag_1 data for XGBoost": "Faltan datos de lag_1 para XGBoost.",
  "Missing lag_24 data for XGBoost": "Faltan datos de lag_24 para XGBoost.",
  "Missing lag_168 data for XGBoost": "Faltan datos de lag_168 para XGBoost.",
  "Missing exogenous data for XGBoost": "Faltan datos exogenos para XGBoost.",
  "No historical data available for XGBoost":
    "No hay datos historicos disponibles para XGBoost.",
  "Requested date not present in historical data for XGBoost":
    "La fecha solicitada no esta presente en los datos historicos para XGBoost.",
  "No historical data available for Chronos":
    "No hay datos historicos disponibles para Chronos.",
  "Requested date is too early for Chronos":
    "La fecha solicitada es demasiado temprana para Chronos.",
  "Requested date not present in historical data for Chronos":
    "La fecha solicitada no esta presente en los datos historicos para Chronos.",
  "Missing price data for Chronos": "Faltan datos de precio para Chronos.",
  "Missing exogenous data for Chronos": "Faltan datos exogenos para Chronos.",
  "No valid historical data available for Chronos":
    "No hay datos historicos validos disponibles para Chronos.",
  "Not enough future covariates for Chronos (need 24 hours)":
    "No hay suficientes covariables futuras para Chronos: se necesitan 24 horas.",
  "Missing future exogenous data for Chronos":
    "Faltan datos exogenos futuros para Chronos.",
  "Chronos output does not include mean predictions":
    "La salida de Chronos no incluye predicciones medias.",
  "Chronos did not return 24 future steps":
    "Chronos no devolvio 24 pasos futuros.",
  "Chronos prediction returned empty output":
    "Chronos devolvio una prediccion vacia.",
};

function translateApiErrorMessage(message: string): string {
  if (API_ERROR_TRANSLATIONS[message]) {
    return API_ERROR_TRANSLATIONS[message];
  }

  if (message.startsWith("Model '") && message.endsWith("' is not supported")) {
    const modelId = message.slice(7, message.indexOf("' is not supported"));
    return `El modelo '${modelId}' no es compatible.`;
  }

  if (message.startsWith("Chronos prediction failed: ")) {
    const detail = message.replace("Chronos prediction failed: ", "");
    return `La prediccion de Chronos ha fallado: ${detail}`;
  }

  if (message.startsWith("Chronos output missing item_id 'price': ")) {
    const detail = message.replace("Chronos output missing item_id 'price': ", "");
    return `La salida de Chronos no contiene el item_id 'price': ${detail}`;
  }

  return message;
}

export async function apiFetch<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);

  if (!response.ok) {
    let detail = "";

    try {
      const data = (await response.json()) as { detail?: string };
      detail = data.detail ? translateApiErrorMessage(data.detail) : "";
    } catch {
      detail = "";
    }

    throw new Error(detail || `Error de API: ${response.status}`);
  }

  return response.json();
}

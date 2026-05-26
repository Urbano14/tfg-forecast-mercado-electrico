// Página que muestra los modelos disponibles y sus métricas principales.

import { useEffect, useMemo, useState } from "react";
import { fetchMetrics, type ModelMetric } from "../api/metricsApi";
import { fetchModels, type ModelInfo } from "../api/modelsApi";
import { formatNumber } from "../utils/number";

function ModelsPage() {
  // Lista de modelos disponibles que devuelve el backend.
  const [models, setModels] = useState<ModelInfo[]>([]);

  // Métricas asociadas a cada modelo: MAE, RMSE y horizonte.
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);

  // Estados de carga y error para controlar lo que se muestra en pantalla.
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modelo seleccionado visualmente en la página.
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);

        // Carga en paralelo los modelos disponibles y sus métricas.
        const [modelsResponse, metricsResponse] = await Promise.all([
          fetchModels(),
          fetchMetrics(),
        ]);

        setModels(modelsResponse.models);
        setMetrics(metricsResponse.metrics);

        // Si todavía no hay modelo seleccionado, selecciona por defecto el primero recibido.
        setSelectedModelId((current) => current ?? modelsResponse.models[0]?.id ?? null);
      } catch (err) {
        // Si falla alguna llamada al backend, se muestra el error en la página.
        setError(err instanceof Error ? err.message : "Error desconocido");
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  // Crea un mapa para acceder rápido a las métricas de cada modelo por su id.
  // Ejemplo: "xgboost" -> métricas de XGBoost.
  const metricsById = useMemo(() => {
    return new Map(metrics.map((metric) => [metric.id, metric]));
  }, [metrics]);

  // Traduce los tipos internos del backend a textos más claros para la interfaz.
  const modelTypeLabels: Record<string, string> = {
    baseline: "modelo base",
    machine_learning: "aprendizaje automatico",
    foundation_model: "modelo fundacional",
  };

  // Clases visuales para alternar el color de las tarjetas de modelos.
  const cardToneClasses = ["models-card--blue", "models-card--amber", "models-card--teal"];

  return (
    <div className="page">
      <header className="page__header">
        <div>
          <h1 className="page__title">Modelos disponibles</h1>
          <p className="page__subtitle">
            Comparativa global de rendimiento y tipo de modelo.
          </p>
        </div>
      </header>

      <section className="card">
        <div className="card__header">
          <h2>Resumen de modelos</h2>
          <p>Metricas MAE y RMSE agregadas por modelo.</p>
        </div>

        {/* Estado de carga mientras se recuperan modelos y métricas del backend. */}
        {loading && (
          <div className="loading-stack" aria-live="polite">
            <div className="loading-bar" />
            <p className="status">Cargando modelos...</p>
          </div>
        )}

        {/* Mensaje de error si falla la carga desde la API. */}
        {!loading && error && (
          <p className="status status--error">Error al cargar modelos: {error}</p>
        )}

        {/* Cuando ya hay datos, se pinta una tarjeta por cada modelo disponible. */}
        {!loading && !error && (
          <div className="models-grid">
            {models.map((model, index) => {
              const metric = metricsById.get(model.id);

              const toneClass = cardToneClasses[index % cardToneClasses.length];

              const isSelected = selectedModelId === model.id;

              return (
                <button
                  className={`models-card models-card--button ${toneClass}${
                    isSelected ? " models-card--selected" : ""
                  }`}
                  key={model.id}
                  onClick={() => setSelectedModelId(model.id)}
                  type="button"
                >
                  {/* Tipo de modelo: baseline, machine learning o modelo fundacional. */}
                  <div className="models-card__tag">
                    {modelTypeLabels[model.type] ?? model.type}
                  </div>

                  {/* Nombre visible del modelo. */}
                  <h3 className="models-card__title">{model.name}</h3>

                  {/* Todos los modelos de la app trabajan con horizonte de 24 horas. */}
                  <p className="models-card__meta">Horizonte: {model.horizon_hours}h</p>

                  {/* Métricas principales mostradas al usuario. */}
                  <div className="models-metrics">
                    <div className="models-metrics__item">
                      <span className="models-metrics__label">MAE</span>
                      <strong className="models-metrics__value">
                        {formatNumber(metric?.mae ?? null, 4)}
                      </strong>
                    </div>
                    <div className="models-metrics__item">
                      <span className="models-metrics__label">RMSE</span>
                      <strong className="models-metrics__value models-metrics__value--warm">
                        {formatNumber(metric?.rmse ?? null, 4)}
                      </strong>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

export default ModelsPage;

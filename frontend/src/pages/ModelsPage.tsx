import { useEffect, useMemo, useState } from "react";
import { fetchMetrics, type ModelMetric } from "../api/metricsApi";
import { fetchModels, type ModelInfo } from "../api/modelsApi";
import { formatNumber } from "../utils/number";

function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);

        const [modelsResponse, metricsResponse] = await Promise.all([
          fetchModels(),
          fetchMetrics(),
        ]);

        setModels(modelsResponse.models);
        setMetrics(metricsResponse.metrics);
        setSelectedModelId((current) => current ?? modelsResponse.models[0]?.id ?? null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  const metricsById = useMemo(() => {
    return new Map(metrics.map((metric) => [metric.id, metric]));
  }, [metrics]);

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

        {loading && (
          <div className="loading-stack" aria-live="polite">
            <div className="loading-bar" />
            <p className="status">Cargando modelos...</p>
          </div>
        )}
        {!loading && error && (
          <p className="status status--error">Error al cargar modelos: {error}</p>
        )}

        {!loading && !error && (
          <div className="models-grid">
            {models.map((model, index) => {
              const metric = metricsById.get(model.id);
              const toneClass = cardToneClasses[index % cardToneClasses.length];
              const isSelected = selectedModelId === model.id;

              return (
                <button
                  // This keeps the requested selected glow without altering routing or data flow.
                  className={`models-card models-card--button ${toneClass}${
                    isSelected ? " models-card--selected" : ""
                  }`}
                  key={model.id}
                  onClick={() => setSelectedModelId(model.id)}
                  type="button"
                >
                  <div className="models-card__tag">{model.type}</div>
                  <h3 className="models-card__title">{model.name}</h3>
                  <p className="models-card__meta">Horizon: {model.horizon_hours}h</p>
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

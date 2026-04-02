import { useEffect, useMemo, useState } from "react";
import { fetchMetrics, type ModelMetric } from "../api/metricsApi";
import { fetchModels, type ModelInfo } from "../api/modelsApi";
import { formatNumber } from "../utils/number";

function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

        {loading && <p className="status">Cargando modelos...</p>}
        {!loading && error && (
          <p className="status status--error">Error al cargar modelos: {error}</p>
        )}

        {!loading && !error && (
          <div className="models-grid">
            {models.map((model) => {
              const metric = metricsById.get(model.id);
              return (
                <div className="models-card" key={model.id}>
                  <div className="models-card__tag">{model.type}</div>
                  <h3 className="models-card__title">{model.name}</h3>
                  <p className="models-card__meta">
                    Horizon: {model.horizon_hours}h
                  </p>
                  <div className="models-metrics">
                    <div className="models-metrics__item">
                      <span className="models-metrics__label">MAE</span>
                      <strong className="models-metrics__value">
                        {formatNumber(metric?.mae ?? null, 4)}
                      </strong>
                    </div>
                    <div className="models-metrics__item">
                      <span className="models-metrics__label">RMSE</span>
                      <strong className="models-metrics__value">
                        {formatNumber(metric?.rmse ?? null, 4)}
                      </strong>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

export default ModelsPage;

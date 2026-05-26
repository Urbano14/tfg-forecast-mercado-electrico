// Pagina principal del dashboard con resumen de indicadores clave y accesos rapidos a secciones principales

import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { fetchHistoricalRange } from "../api/historicalApi";
import { fetchMetrics, type ModelMetric } from "../api/metricsApi";
import { fetchModels, type ModelInfo } from "../api/modelsApi";
import { APP_TIMEZONE, formatDate } from "../utils/date";
import { formatNumber } from "../utils/number";

function DashboardPage() {
  // Guarda el rango temporal disponible en la base de datos.
  const [range, setRange] = useState<{ start: string; end: string } | null>(null);

  // Lista de modelos disponibles que devuelve el endpoint /models.
  const [models, setModels] = useState<ModelInfo[]>([]);

  // Métricas visibles de los modelos que devuelve el endpoint /metrics.
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);

  // Estado de carga para mostrar guiones o mensajes mientras llegan los datos.
  const [loading, setLoading] = useState(true);

  // Mensaje de error si alguna llamada al backend falla.
  const [error, setError] = useState<string | null>(null);

  // Al cargar el dashboard, se piden al backend los datos necesarios para el resumen.
  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);

        // Ejecuta las tres peticiones en paralelo para no esperar una detrás de otra.
        const [rangeResponse, modelsResponse, metricsResponse] = await Promise.all([
          fetchHistoricalRange(),
          fetchModels(),
          fetchMetrics(),
        ]);

        // Guarda las respuestas en el estado para que React vuelva a pintar la página.
        setRange(rangeResponse);
        setModels(modelsResponse.models);
        setMetrics(metricsResponse.metrics);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Error desconocido");
      } finally {
        // Termina la carga tanto si todo ha ido bien como si ha fallado.
        setLoading(false);
      }
    }

    loadData();
  }, []);

  // Crea un mapa id -> modelo para poder recuperar el nombre visible de cada modelo.
  const modelById = useMemo(() => {
    return new Map(models.map((model) => [model.id, model]));
  }, [models]);

  // Busca el modelo con menor MAE entre las métricas cargadas.
  const bestMaeMetric = useMemo<ModelMetric | null>(() => {
    let best: ModelMetric | null = null;
    metrics.forEach((metric) => {
      if (metric.mae === null || metric.mae === undefined) {
        return;
      }

      if (!best || metric.mae < (best.mae ?? Number.POSITIVE_INFINITY)) {
        best = metric;
      }
    });
    return best;
  }, [metrics]);

  // Busca el modelo con menor RMSE entre las métricas cargadas.
  const bestRmseMetric = useMemo<ModelMetric | null>(() => {
    let best: ModelMetric | null = null;
    metrics.forEach((metric) => {
      if (metric.rmse === null || metric.rmse === undefined) {
        return;
      }

      if (!best || metric.rmse < (best.rmse ?? Number.POSITIVE_INFINITY)) {
        best = metric;
      }
    });
    return best;
  }, [metrics]);

  // Obtiene el nombre visible del modelo con mejor MAE y RMSE.
  const bestMaeName =
    (bestMaeMetric && modelById.get(bestMaeMetric.id)?.name) ?? bestMaeMetric?.id ?? "-";
  const bestRmseName =
    (bestRmseMetric && modelById.get(bestRmseMetric.id)?.name) ?? bestRmseMetric?.id ?? "-";

  const rangeStart = range ? formatDate(range.start) : "-";
  const rangeEnd = range ? formatDate(range.end) : "-";

  return (
    <div className="page">
      {/* Cabecera principal del dashboard con descripción, rango y accesos principales. */}
      <section className="hero">
        <div>
          <p className="hero__eyebrow">TFG Energia</p>
          <h1 className="hero__title">Centro de control del mercado electrico</h1>
          <p className="hero__lead">
            Panel central para el analisis historico, comparativa de modelos y
            seguimiento de predicciones del mercado electrico espanol.
          </p>
          <div className="hero__meta">
            <span className="pill">{"\u26A1"} Zona horaria: {APP_TIMEZONE}</span>
            {range ? (
              <span className="pill pill--warm">
                Rango: {rangeStart} - {rangeEnd}
              </span>
            ) : (
              <span className="pill pill--muted">Rango: cargando...</span>
            )}
          </div>
          <div className="hero__actions">
            <Link className="btn btn--primary" to="/historico">
              Explorar historico
            </Link>
            <Link className="btn" to="/forecast">
              Comparar predicciones
            </Link>
          </div>
        </div>

        {/* Panel lateral con resumen rápido del estado de la aplicación. */}
        <div className="hero-panel">
          <div className="hero-panel__card">
            <p className="hero-panel__label">Modelos disponibles</p>
            <p className="hero-panel__value">{loading ? "-" : models.length}</p>
          </div>
          <div className="hero-panel__card">
            <p className="hero-panel__label">Metricas cargadas</p>
            <p className="hero-panel__value hero-panel__value--warm">
              {loading ? "-" : metrics.length}
            </p>
          </div>
          <div className="hero-panel__card">
            <p className="hero-panel__label">Ultima fecha disponible</p>
            <p className="hero-panel__value">{loading ? "-" : rangeEnd}</p>
          </div>
        </div>
      </section>

      {/* Mensaje de error si alguna petición inicial al backend falla. */}
      {error && <p className="status status--error">Error al cargar resumen: {error}</p>}

      {/* KPIs principales del sistema: mejores métricas y rango histórico. */}
      <section className="section">
        <div className="section__header">
          <h2>Indicadores clave</h2>
          <p>Resumen rapido de modelos y rendimiento global en EUR/MWh.</p>
        </div>

        <div className="kpi-grid">
          <div className="kpi-card">
            <p className="kpi-card__label">Mejor MAE (EUR/MWh)</p>
            <p className="kpi-card__value">
              {bestMaeMetric ? formatNumber(bestMaeMetric.mae, 4) : "-"}
            </p>
            <p className="kpi-card__meta">Modelo: {bestMaeName}</p>
          </div>

          <div className="kpi-card">
            <p className="kpi-card__label">Mejor RMSE (EUR/MWh)</p>
            <p className="kpi-card__value kpi-card__value--warm">
              {bestRmseMetric ? formatNumber(bestRmseMetric.rmse, 4) : "-"}
            </p>
            <p className="kpi-card__meta">Modelo: {bestRmseName}</p>
          </div>

          <div className="kpi-card">
            <p className="kpi-card__label">Rango historico</p>
            <p className="kpi-card__value">{loading ? "-" : `${rangeStart}`}</p>
            <p className="kpi-card__meta">{loading ? "-" : `Hasta ${rangeEnd}`}</p>
          </div>
        </div>
      </section>

      {/* Tarjetas para entrar rápidamente a los módulos principales. */}
      <section className="section">
        <div className="section__header">
          <h2>Accesos rapidos</h2>
          <p>Explora los modulos principales de la plataforma.</p>
        </div>

        <div className="quick-grid">
          <Link className="quick-card" to="/historico">
            <h3 className="quick-card__title">Historico</h3>
            <p className="quick-card__meta">
              Filtra rangos y consulta la tabla y la grafica de precios en EUR/MWh.
            </p>
          </Link>
          <Link className="quick-card" to="/forecast">
            <h3 className="quick-card__title">Prediccion</h3>
            <p className="quick-card__meta">
              Compara modelos y revisa las predicciones generadas en EUR/MWh.
            </p>
          </Link>
          <Link className="quick-card" to="/models">
            <h3 className="quick-card__title">Modelos</h3>
            <p className="quick-card__meta">
              Analiza el rendimiento global y los horizontes disponibles.
            </p>
          </Link>
        </div>
      </section>
    </div>
  );
}

export default DashboardPage;

import { useEffect, useMemo, useState } from "react";
import { fetchForecast, type ForecastPoint } from "../api/forecastApi";
import {
  fetchHistoricalData,
  fetchHistoricalRange,
  type HistoricalDataPoint,
} from "../api/historicalApi";
import { fetchMetrics, type ModelMetric } from "../api/metricsApi";
import { fetchModels, type ModelInfo } from "../api/modelsApi";
import PriceChart from "../components/PriceChart";
import { APP_TIMEZONE, toDateInputValue } from "../utils/date";
import { formatNumber } from "../utils/number";

function ForecastPage() {
  const MAX_FORECAST_CALLS_PER_MODEL = 3;
  const MAX_FORECAST_RANGE_DAYS = 3;

  const [range, setRange] = useState<{ start: string; end: string } | null>(null);
  const [forecastA, setForecastA] = useState<ForecastPoint[]>([]);
  const [forecastB, setForecastB] = useState<ForecastPoint[]>([]);
  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);

  const [forecastError, setForecastError] = useState<string | null>(null);
  const [historicalError, setHistoricalError] = useState<string | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [loadingModels, setLoadingModels] = useState(true);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [historicalLoading, setHistoricalLoading] = useState(false);

  const [selectedModelA, setSelectedModelA] = useState("seasonal_naive");
  const [selectedModelB, setSelectedModelB] = useState("seasonal_naive");
  const [historicalStart, setHistoricalStart] = useState("2022-01-01");
  const [historicalEnd, setHistoricalEnd] = useState("2022-01-04");
  const [forecastStart, setForecastStart] = useState("2022-01-03");
  const [forecastEnd, setForecastEnd] = useState("2022-01-03");

  const [forecastLoaded, setForecastLoaded] = useState(false);
  const [lastForecastParams, setLastForecastParams] = useState<{
    modelA: string;
    modelB: string;
    forecastStart: string;
    forecastEnd: string;
    historicalStart: string;
    historicalEnd: string;
  } | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoadingModels(true);
        setMetricsError(null);

        const [rangeResponse, modelsResponse, metricsResponse] = await Promise.all([
          fetchHistoricalRange(),
          fetchModels(),
          fetchMetrics(),
        ]);

        setRange(rangeResponse);
        setModels(modelsResponse.models);
        setMetrics(metricsResponse.metrics);
      } catch (err) {
        setMetricsError(err instanceof Error ? err.message : "Error desconocido");
      } finally {
        setLoadingModels(false);
      }
    }

    loadData();
  }, []);

  function parseDate(value: string): Date | null {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return null;
    }
    return parsed;
  }

  function toIsoHour(date: Date): string {
    return new Date(
      Date.UTC(
        date.getFullYear(),
        date.getMonth(),
        date.getDate(),
        date.getHours(),
        date.getMinutes(),
        0,
        0
      )
    ).toISOString();
  }

  function addHours(date: Date, hours: number): Date {
    const copy = new Date(date.getTime());
    copy.setHours(copy.getHours() + hours);
    return copy;
  }

  function buildForecastBaseDates(startDate: Date, endDate: Date, horizonHours: number) {
    const dates: string[] = [];
    let cursor = new Date(startDate.getTime());
    while (cursor <= endDate) {
      dates.push(toIsoHour(cursor));
      cursor = addHours(cursor, horizonHours);
    }
    return dates;
  }

  function diffDays(startDate: Date, endDate: Date): number {
    const msPerDay = 24 * 60 * 60 * 1000;
    return Math.ceil((endDate.getTime() - startDate.getTime()) / msPerDay);
  }

  async function loadForecastComparison() {
    try {
      setForecastLoading(true);
      setForecastError(null);
      setValidationError(null);
      setHistoricalError(null);

      if (historicalStart > historicalEnd) {
        setValidationError("El rango del historico es invalido (inicio > fin).");
        setForecastLoaded(false);
        return;
      }

      if (forecastStart > forecastEnd) {
        setValidationError("El rango de prediccion es invalido (inicio > fin).");
        setForecastLoaded(false);
        return;
      }

      if (range) {
        const minDate = toDateInputValue(range.start);
        const maxDate = toDateInputValue(range.end);
        if (
          (minDate && historicalStart < minDate) ||
          (maxDate && historicalStart > maxDate) ||
          (minDate && historicalEnd < minDate) ||
          (maxDate && historicalEnd > maxDate)
        ) {
          setValidationError(
            `El historico debe estar dentro del rango disponible (${minDate} a ${maxDate}).`
          );
          setForecastLoaded(false);
          return;
        }
      }

      const forecastStartDate = parseDate(`${forecastStart}T00:00:00`);
      const forecastEndDate = parseDate(`${forecastEnd}T23:00:00`);
      if (!forecastStartDate || !forecastEndDate) {
        setValidationError("Las fechas de prediccion no son validas.");
        setForecastLoaded(false);
        return;
      }

      if (diffDays(forecastStartDate, forecastEndDate) > MAX_FORECAST_RANGE_DAYS) {
        setValidationError(
          `El rango de prediccion es demasiado grande. Maximo ${MAX_FORECAST_RANGE_DAYS} dias.`
        );
        setForecastLoaded(false);
        return;
      }

      const modelA = models.find((model) => model.id === selectedModelA);
      const modelB = models.find((model) => model.id === selectedModelB);
      const horizonA = modelA?.horizon_hours ?? 24;
      const horizonB = modelB?.horizon_hours ?? 24;

      const baseDatesA = buildForecastBaseDates(
        forecastStartDate,
        forecastEndDate,
        horizonA
      );
      const baseDatesB = buildForecastBaseDates(
        forecastStartDate,
        forecastEndDate,
        horizonB
      );

      if (
        baseDatesA.length > MAX_FORECAST_CALLS_PER_MODEL ||
        baseDatesB.length > MAX_FORECAST_CALLS_PER_MODEL
      ) {
        setValidationError(
          `El rango de prediccion es demasiado grande. Ajusta a un rango menor (maximo ${MAX_FORECAST_CALLS_PER_MODEL} llamadas por modelo).`
        );
        setForecastLoaded(false);
        return;
      }

      const combinedForecastA: ForecastPoint[] = [];
      const combinedForecastB: ForecastPoint[] = [];

      for (const baseDate of baseDatesA) {
        const response = await fetchForecast(baseDate, selectedModelA);
        combinedForecastA.push(...response.forecast);
      }

      for (const baseDate of baseDatesB) {
        const response = await fetchForecast(baseDate, selectedModelB);
        combinedForecastB.push(...response.forecast);
      }

      setForecastA(combinedForecastA);
      setForecastB(combinedForecastB);
      setForecastLoaded(true);
      setLastForecastParams({
        modelA: selectedModelA,
        modelB: selectedModelB,
        forecastStart,
        forecastEnd,
        historicalStart,
        historicalEnd,
      });

      setHistoricalLoading(true);
      try {
        const historicalResponse = await fetchHistoricalData(
          historicalStart,
          historicalEnd,
          300
        );
        setHistoricalData(historicalResponse);
      } catch (err) {
        setHistoricalError(err instanceof Error ? err.message : "Error desconocido");
        setHistoricalData([]);
      } finally {
        setHistoricalLoading(false);
      }
    } catch (err) {
      setForecastError(err instanceof Error ? err.message : "Error desconocido");
      setForecastLoaded(false);
    } finally {
      setForecastLoading(false);
    }
  }

  const metricsById = useMemo(() => {
    return new Map(metrics.map((metric) => [metric.id, metric]));
  }, [metrics]);

  const modelById = useMemo(() => {
    return new Map(models.map((model) => [model.id, model]));
  }, [models]);

  const selectedMetricA = metricsById.get(selectedModelA);
  const selectedMetricB = metricsById.get(selectedModelB);
  const selectedModelInfoA = modelById.get(selectedModelA);
  const selectedModelInfoB = modelById.get(selectedModelB);

  const modelTypeLabels: Record<string, string> = {
    baseline: "modelo base",
    machine_learning: "aprendizaje automatico",
    foundation_model: "modelo fundacional",
  };

  const chartData = useMemo(() => {
    const byTimestamp = new Map<
      string,
      { timestamp: string; price?: number; forecastA?: number; forecastB?: number }
    >();

    historicalData.forEach((row) => {
      byTimestamp.set(row.timestamp, {
        timestamp: row.timestamp,
        price: row.price,
      });
    });

    forecastA.forEach((row) => {
      const existing = byTimestamp.get(row.timestamp);

      if (existing) {
        existing.forecastA = row.value;
      } else {
        byTimestamp.set(row.timestamp, {
          timestamp: row.timestamp,
          forecastA: row.value,
        });
      }
    });

    forecastB.forEach((row) => {
      const existing = byTimestamp.get(row.timestamp);

      if (existing) {
        existing.forecastB = row.value;
      } else {
        byTimestamp.set(row.timestamp, {
          timestamp: row.timestamp,
          forecastB: row.value,
        });
      }
    });

    return Array.from(byTimestamp.values()).sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }, [forecastA, forecastB, historicalData]);

  const forecastIsStale =
    !!lastForecastParams &&
    (lastForecastParams.modelA !== selectedModelA ||
      lastForecastParams.modelB !== selectedModelB ||
      lastForecastParams.forecastStart !== forecastStart ||
      lastForecastParams.forecastEnd !== forecastEnd ||
      lastForecastParams.historicalStart !== historicalStart ||
      lastForecastParams.historicalEnd !== historicalEnd);

  return (
    <div className="page">
      <header className="page__header">
        <div>
          <h1 className="page__title">Prediccion y comparativa</h1>
          <p className="page__subtitle">
            Generacion de predicciones horarias y comparacion de modelos. Zona
            horaria: {` ${APP_TIMEZONE}`}.
          </p>
        </div>
      </header>

      <section className="card">
        <div className="card__header">
          <h2>Configuracion de prediccion</h2>
          <p>Selecciona modelos y rango temporal para comparar historico y prediccion.</p>
        </div>

        <div className="form-grid">
          <label className="field">
            <span>Modelo A</span>
            <select
              value={selectedModelA}
              onChange={(e) => {
                setSelectedModelA(e.target.value);
                setForecastLoaded(false);
              }}
              disabled={loadingModels}
            >
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Modelo B</span>
            <select
              value={selectedModelB}
              onChange={(e) => {
                setSelectedModelB(e.target.value);
                setForecastLoaded(false);
              }}
              disabled={loadingModels}
            >
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Inicio de prediccion</span>
            <input
              type="date"
              value={forecastStart}
              onChange={(e) => {
                setForecastStart(e.target.value);
                setForecastLoaded(false);
              }}
            />
          </label>

          <label className="field">
            <span>Fin de prediccion</span>
            <input
              type="date"
              value={forecastEnd}
              onChange={(e) => {
                setForecastEnd(e.target.value);
                setForecastLoaded(false);
              }}
            />
          </label>

          <label className="field">
            <span>Historico inicio</span>
            <input
              type="date"
              value={historicalStart}
              min={range ? toDateInputValue(range.start) : undefined}
              max={range ? toDateInputValue(range.end) : undefined}
              onChange={(e) => {
                setHistoricalStart(e.target.value);
                setForecastLoaded(false);
              }}
            />
          </label>

          <label className="field">
            <span>Historico fin</span>
            <input
              type="date"
              value={historicalEnd}
              min={range ? toDateInputValue(range.start) : undefined}
              max={range ? toDateInputValue(range.end) : undefined}
              onChange={(e) => {
                setHistoricalEnd(e.target.value);
                setForecastLoaded(false);
              }}
            />
          </label>

          <div className="field field--actions">
            <span className="field__hint">
              Genera historico y prediccion para el rango seleccionado.
            </span>
            <button
              className="btn btn--primary"
              onClick={loadForecastComparison}
              disabled={
                !selectedModelA ||
                !selectedModelB ||
                !forecastStart ||
                !forecastEnd ||
                !historicalStart ||
                !historicalEnd ||
                forecastLoading
              }
            >
              {forecastLoading ? "Cargando..." : "Cargar comparacion"}
            </button>
          </div>
        </div>

        {validationError && <p className="status status--error">{validationError}</p>}

        {!forecastLoaded && !forecastError && forecastIsStale && (
          <p className="status status--notice">
            La seleccion ha cambiado. Pulsa "Cargar comparacion" para actualizar la
            prediccion.
          </p>
        )}

        {!forecastLoaded && !forecastError && !forecastIsStale && (
          <p className="status">Selecciona los parametros y carga la comparacion.</p>
        )}

        {forecastError && (
          <p className="status status--error">
            Error al cargar la prediccion: {forecastError}
          </p>
        )}
      </section>

      <section className="card">
        <div className="card__header">
          <h2>Metricas por modelo</h2>
          <p>Comparativa cuantitativa para los modelos seleccionados.</p>
        </div>

        {loadingModels && (
          <div className="loading-stack" aria-live="polite">
            <div className="loading-bar" />
            <p className="status">Cargando metricas...</p>
          </div>
        )}
        {!loadingModels && metricsError && (
          <p className="status status--error">Error al cargar metricas: {metricsError}</p>
        )}

        {!loadingModels && !metricsError && (
          <div className="metrics-grid">
            <div className="metrics-card metrics-card--active">
              <div className="metrics-card__tag">Modelo A</div>
              <h3 className="metrics-card__title">
                {selectedModelInfoA?.name ?? selectedModelA}
              </h3>
              <p className="metrics-card__meta">
                Tipo: {modelTypeLabels[selectedModelInfoA?.type ?? ""] ?? selectedModelInfoA?.type ?? "-"}
              </p>
              <div className="metrics-list">
                <div className="metrics-list__item">
                  <span className="metrics-label">MAE</span>
                  <strong className="metrics-value">
                    {formatNumber(selectedMetricA?.mae ?? null, 4)}
                  </strong>
                </div>
                <div className="metrics-list__item">
                  <span className="metrics-label">RMSE</span>
                  <strong className="metrics-value metrics-value--warm">
                    {formatNumber(selectedMetricA?.rmse ?? null, 4)}
                  </strong>
                </div>
              </div>
            </div>

            <div className="metrics-card metrics-card--active">
              <div className="metrics-card__tag">Modelo B</div>
              <h3 className="metrics-card__title">
                {selectedModelInfoB?.name ?? selectedModelB}
              </h3>
              <p className="metrics-card__meta">
                Tipo: {modelTypeLabels[selectedModelInfoB?.type ?? ""] ?? selectedModelInfoB?.type ?? "-"}
              </p>
              <div className="metrics-list">
                <div className="metrics-list__item">
                  <span className="metrics-label">MAE</span>
                  <strong className="metrics-value">
                    {formatNumber(selectedMetricB?.mae ?? null, 4)}
                  </strong>
                </div>
                <div className="metrics-list__item">
                  <span className="metrics-label">RMSE</span>
                  <strong className="metrics-value metrics-value--warm">
                    {formatNumber(selectedMetricB?.rmse ?? null, 4)}
                  </strong>
                </div>
              </div>
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <div className="card__header">
          <h2>Grafica de prediccion</h2>
          <p>Comparativa visual de las predicciones generadas.</p>
        </div>

        <div className="chart-wrap">
          {forecastLoading && (
            <div className="loading-stack" aria-live="polite">
              <div className="loading-bar" />
              <p className="status">Cargando prediccion...</p>
            </div>
          )}
          {!forecastLoading && historicalLoading && (
            <div className="loading-stack" aria-live="polite">
              <div className="loading-bar" />
              <p className="status">Cargando historico...</p>
            </div>
          )}
          {!forecastLoading && !forecastError && chartData.length === 0 && (
            <p className="status">No hay datos de prediccion disponibles para la seleccion.</p>
          )}
          {!forecastLoading && !historicalLoading && historicalError && (
            <p className="status status--error">
              Error al cargar historico: {historicalError}
            </p>
          )}
          {!forecastLoading && !forecastError && chartData.length > 0 && (
            <PriceChart data={chartData} />
          )}
        </div>
      </section>
    </div>
  );
}

export default ForecastPage;

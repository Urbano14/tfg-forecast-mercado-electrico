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
import { APP_TIMEZONE, formatDate, toDateInputValue } from "../utils/date";
import { formatNumber } from "../utils/number";

function ForecastPage() {
  const [range, setRange] = useState<{ start: string; end: string } | null>(null);
  const [forecastA, setForecastA] = useState<ForecastPoint[]>([]);
  const [forecastB, setForecastB] = useState<ForecastPoint[]>([]);
  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);

  const [forecastError, setForecastError] = useState<string | null>(null);
  const [historicalError, setHistoricalError] = useState<string | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [loadingModels, setLoadingModels] = useState(true);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [historicalLoading, setHistoricalLoading] = useState(false);

  const [selectedModelA, setSelectedModelA] = useState("seasonal_naive");
  const [selectedModelB, setSelectedModelB] = useState("seasonal_naive");
  const [historicalStart, setHistoricalStart] = useState("2022-01-01");
  const [historicalEnd, setHistoricalEnd] = useState("2022-01-04");
  const [forecastBaseDate, setForecastBaseDate] = useState("2022-01-03");

  const [forecastLoaded, setForecastLoaded] = useState(false);
  const [lastForecastParams, setLastForecastParams] = useState<{
    modelA: string;
    modelB: string;
    forecastBaseDate: string;
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

  function parseDateOnly(dateStr: string): Date {
    const [year, month, day] = dateStr.split("-").map(Number);
    return new Date(year, month - 1, day);
  }

  function addDays(dateStr: string, days: number): string {
    const date = parseDateOnly(dateStr);
    date.setDate(date.getDate() + days);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  function diffDays(startDateStr: string, endDateStr: string): number {
    const start = parseDateOnly(startDateStr);
    const end = parseDateOnly(endDateStr);
    return Math.round((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
  }

  function getErrorMessage(err: unknown): string {
    if (typeof err === "object" && err !== null) {
      const detail = "detail" in err ? err.detail : null;
      if (typeof detail === "string" && detail) {
        return detail;
      }

      const message = "message" in err ? err.message : null;
      if (typeof message === "string" && message) {
        return message;
      }
    }

    if (err instanceof Error && err.message) {
      return err.message;
    }

    return "Error desconocido";
  }

  const availableMinDate = range ? toDateInputValue(range.start) : null;
  const availableMaxDate = range ? toDateInputValue(range.end) : null;
  const availableMinDateLabel = availableMinDate ? formatDate(availableMinDate) : null;
  const availableMaxDateLabel = availableMaxDate ? formatDate(availableMaxDate) : null;
  const maxHistoricalEndDate =
    availableMaxDate && availableMaxDate < addDays(historicalStart, 5)
      ? availableMaxDate
      : addDays(historicalStart, 5);

  function validateForecastDates(): string | null {
    if (availableMinDate && historicalStart < availableMinDate) {
      return "El inicio del histórico no puede ser anterior al primer dato disponible.";
    }

    if (availableMaxDate && historicalEnd > availableMaxDate) {
      return "El fin del histórico no puede ser posterior al último dato disponible.";
    }

    if (historicalStart > historicalEnd) {
      return "La fecha de inicio del histórico no puede ser posterior a la fecha de fin.";
    }

    if (diffDays(historicalStart, historicalEnd) > 5) {
      return "El rango histórico no puede superar los 5 días.";
    }

    if (forecastBaseDate < historicalStart) {
      return "La fecha base de predicción no puede ser anterior al inicio del histórico.";
    }

    if (forecastBaseDate > historicalEnd) {
      return "La fecha a predecir no puede ser posterior al fin del histórico.";
    }

    if (availableMaxDate && forecastBaseDate > availableMaxDate) {
      return "La fecha base de predicción no puede ser posterior al último dato disponible.";
    }

    return null;
  }

  async function loadForecastComparison() {
    const validationMessage = validateForecastDates();
    if (validationMessage) {
      setForecastError(validationMessage);
      return;
    }

    try {
      setForecastError(null);
      setHistoricalError(null);
      setForecastLoading(true);
      const baseDateIso = `${forecastBaseDate}T00:00:00`;
      const responseA = await fetchForecast(baseDateIso, selectedModelA);
      const forecastForA = responseA.forecast;

      let forecastForB: ForecastPoint[] = forecastForA;
      if (selectedModelB !== selectedModelA) {
        const responseB = await fetchForecast(baseDateIso, selectedModelB);
        forecastForB = responseB.forecast;
      }

      setForecastA(forecastForA);
      setForecastB(forecastForB);
      setForecastLoaded(true);
      setLastForecastParams({
        modelA: selectedModelA,
        modelB: selectedModelB,
        forecastBaseDate,
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
        setHistoricalError(getErrorMessage(err));
        setHistoricalData([]);
      } finally {
        setHistoricalLoading(false);
      }
    } catch (err) {
      setForecastError(getErrorMessage(err));
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
      lastForecastParams.forecastBaseDate !== forecastBaseDate ||
      lastForecastParams.historicalStart !== historicalStart ||
      lastForecastParams.historicalEnd !== historicalEnd);

  return (
    <div className="page">
      <header className="page__header">
        <div>
          <h1 className="page__title">Prediccion y comparativa</h1>
          <p className="page__subtitle">
            Selecciona una fecha base para generar la prediccion de las 24 horas
            posteriores y comparar modelos. Zona horaria: {` ${APP_TIMEZONE}`}.
          </p>
        </div>
      </header>

      <section className="card">
        <div className="card__header">
          <h2>Configuracion de prediccion</h2>
          <p>
            Selecciona una fecha base dentro del rango historico disponible para
            comparar modelos en las 24 horas posteriores.
          </p>
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
            <span>Fecha base de predicción</span>
            <input
              type="date"
              value={forecastBaseDate}
              min={historicalStart}
              max={historicalEnd}
              onChange={(e) => {
                setForecastBaseDate(e.target.value);
                setForecastLoaded(false);
              }}
            />
            <span className="field__hint">Seleccionada: {formatDate(forecastBaseDate)}</span>
          </label>

          <label className="field">
            <span>Historico inicio</span>
            <input
              type="date"
              value={historicalStart}
              min={availableMinDate ?? undefined}
              max={historicalEnd}
              onChange={(e) => {
                setHistoricalStart(e.target.value);
                setForecastLoaded(false);
              }}
            />
            <span className="field__hint">Seleccionada: {formatDate(historicalStart)}</span>
          </label>

          <label className="field">
            <span>Historico fin</span>
            <input
              type="date"
              value={historicalEnd}
              min={historicalStart}
              max={maxHistoricalEndDate}
              onChange={(e) => {
                setHistoricalEnd(e.target.value);
                setForecastLoaded(false);
              }}
            />
            <span className="field__hint">Seleccionada: {formatDate(historicalEnd)}</span>
          </label>

          <div className="field field--actions">
            <span className="field__hint">
              XGBoost puede usar una variante minima si no hay exogenas futuras;
              Chronos requiere covariables futuras disponibles.
            </span>
            {availableMinDateLabel && availableMaxDateLabel && (
              <span className="field__hint">
                Rango disponible: {availableMinDateLabel} - {availableMaxDateLabel}
              </span>
            )}
            <button
              className="btn btn--primary"
              onClick={loadForecastComparison}
              disabled={
                !selectedModelA ||
                !selectedModelB ||
                !forecastBaseDate ||
                forecastLoading
              }
            >
              {forecastLoading ? "Cargando..." : "Cargar comparacion"}
            </button>
          </div>
        </div>

        {!forecastLoaded && !forecastError && forecastIsStale && (
          <p className="status status--notice">
            La seleccion ha cambiado. Pulsa "Cargar comparacion" para actualizar la
            prediccion de las proximas 24 horas.
          </p>
        )}

        {!forecastLoaded && !forecastError && !forecastIsStale && (
          <p className="status">
            Selecciona una fecha base y carga la prediccion de las proximas 24 horas.
          </p>
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
          <p>
            Comparativa visual de la prediccion de las proximas 24 horas generada
            con los datos disponibles.
          </p>
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
            <p className="status">
              No hay datos de prediccion de las proximas 24 horas para la seleccion.
            </p>
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

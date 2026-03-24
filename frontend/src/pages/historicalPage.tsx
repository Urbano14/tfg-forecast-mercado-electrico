import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchHistoricalData,
  fetchHistoricalRange,
  type HistoricalDataPoint,
} from "../api/historicalApi";
import { fetchForecast, type ForecastPoint } from "../api/forecastApi";
import { fetchModels, type ModelInfo } from "../api/modelsApi.ts";
import PriceChart from "../components/PriceChart";

function HistoricalPage() {
  const initialStart = "2022-01-01";
  const initialEnd = "2022-01-04";

  const [range, setRange] = useState<{ start: string; end: string } | null>(null);
  const [data, setData] = useState<HistoricalDataPoint[]>([]);
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);

  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [start, setStart] = useState(initialStart);
  const [end, setEnd] = useState(initialEnd);

  const [selectedModel, setSelectedModel] = useState("seasonal_naive");
  const [forecastDate, setForecastDate] = useState("2022-01-03T00:00");

  const loadHistoricalData = useCallback(async (currentStart: string, currentEnd: string) => {
    try {
      setLoading(true);
      setError(null);

      const historicalData = await fetchHistoricalData(currentStart, currentEnd, 200);
      setData(historicalData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  async function loadForecast() {
    try {
      setError(null);

      const normalizedForecastDate = `${forecastDate}:00`;
      const forecastResponse = await fetchForecast(
        normalizedForecastDate,
        selectedModel
      );

      setForecast(forecastResponse.forecast);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  }

  useEffect(() => {
    async function initializePage() {
      try {
        const rangeData = await fetchHistoricalRange();
        setRange(rangeData);

        const modelsResponse = await fetchModels();
        setModels(modelsResponse.models);

        await loadHistoricalData(initialStart, initialEnd);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
        setLoading(false);
      }
    }

    initializePage();
  }, [initialStart, initialEnd, loadHistoricalData]);

  function handleLoadClick() {
    loadHistoricalData(start, end);
  }

  const chartData = useMemo(() => {
    const byTimestamp = new Map<string, { timestamp: string; price?: number; forecast?: number }>();
    data.forEach((row) => {
      byTimestamp.set(row.timestamp, { timestamp: row.timestamp, price: row.price });
    });
    forecast.forEach((row) => {
      const existing = byTimestamp.get(row.timestamp);
      if (existing) {
        existing.forecast = row.value;
      } else {
        byTimestamp.set(row.timestamp, { timestamp: row.timestamp, forecast: row.value });
      }
    });
    return Array.from(byTimestamp.values());
  }, [data, forecast]);

  return (
    <div className="page">
      <header className="page__header">
        <div>
          <h1 className="page__title">Analisis Historico y Prediccion</h1>
          <p className="page__subtitle">
            Serie temporal del mercado electrico espanol. Visualizacion del historico y forecast de precios.
          </p>
        </div>
        {range && (
          <div className="range-card">
            <div className="range-card__label">Rango disponible</div>
            <div className="range-card__row">
              <span>Inicio:</span>
              <strong>{range.start}</strong>
            </div>
            <div className="range-card__row">
              <span>Fin:</span>
              <strong>{range.end}</strong>
            </div>
          </div>
        )}
      </header>

      <section className="card">
        <div className="card__header">
          <h2>Filtros de historico</h2>
          <p>Selecciona un rango de fechas para cargar los datos.</p>
        </div>
        <div className="form-grid">
          <label className="field">
            <span>Fecha de inicio</span>
            <input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
            />
          </label>

          <label className="field">
            <span>Fecha de fin</span>
            <input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
            />
          </label>

          <div className="field field--actions">
            <span className="field__hint">Carga el historico para el rango indicado.</span>
            <button className="btn btn--primary" onClick={handleLoadClick}>
              Cargar historico
            </button>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card__header">
          <h2>Forecast</h2>
          <p>Elige modelo y fecha base para generar la prediccion.</p>
        </div>
        <div className="form-grid">
          <label className="field">
            <span>Modelo</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Fecha de forecast</span>
            <input
              type="datetime-local"
              value={forecastDate}
              onChange={(e) => setForecastDate(e.target.value)}
            />
          </label>

          <div className="field field--actions">
            <span className="field__hint">Genera la serie prevista para ese punto.</span>
            <button className="btn" onClick={loadForecast}>
              Cargar forecast
            </button>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card__header">
          <h2>Grafica de precios</h2>
          <p>Historico y prediccion en una sola vista.</p>
        </div>
        <div className="chart-wrap">
          <PriceChart data={chartData} />
        </div>
      </section>

      <section className="card">
        <div className="card__header">
          <h2>Registros historicos</h2>
          <p>Detalle por timestamp con variables auxiliares.</p>
        </div>

        {loading && <p className="status">Cargando datos historicos...</p>}
        {error && <p className="status status--error">Error: {error}</p>}

        {!loading && !error && (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Price</th>
                  <th>Demand</th>
                  <th>Wind</th>
                  <th>Solar</th>
                  <th>Hydro</th>
                </tr>
              </thead>
              <tbody>
                {data.map((row) => (
                  <tr key={row.timestamp}>
                    <td>{row.timestamp}</td>
                    <td>{row.price}</td>
                    <td>{row.demand_forecast ?? "-"}</td>
                    <td>{row.wind_forecast ?? "-"}</td>
                    <td>{row.solar_forecast ?? "-"}</td>
                    <td>{row.hydro_programmed ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default HistoricalPage;


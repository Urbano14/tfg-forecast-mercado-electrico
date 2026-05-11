import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchHistoricalData,
  fetchHistoricalRange,
  type HistoricalDataPoint,
} from "../api/historicalApi";
import PriceChart from "../components/PriceChart";
import { APP_TIMEZONE, formatTimestamp, toDateInputValue } from "../utils/date";
import { formatNumber } from "../utils/number";

function HistoricalPage() {
  const initialStart = "2022-01-01";
  const initialEnd = "2022-01-04";
  const TABLE_LIMIT = 50;

  const [range, setRange] = useState<{ start: string; end: string } | null>(null);
  const [data, setData] = useState<HistoricalDataPoint[]>([]);

  const [historicalError, setHistoricalError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tableQuery, setTableQuery] = useState("");

  const [start, setStart] = useState(initialStart);
  const [end, setEnd] = useState(initialEnd);

  const loadHistoricalData = useCallback(async (currentStart: string, currentEnd: string) => {
    try {
      setLoading(true);
      setHistoricalError(null);

      const historicalData = await fetchHistoricalData(currentStart, currentEnd, 200);
      setData(historicalData);
    } catch (err) {
      setHistoricalError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    async function initializePage() {
      try {
        const rangeData = await fetchHistoricalRange();
        setRange(rangeData);

        await loadHistoricalData(initialStart, initialEnd);
      } catch (err) {
        setHistoricalError(err instanceof Error ? err.message : "Error desconocido");
        setLoading(false);
      }
    }

    initializePage();
  }, [loadHistoricalData]);

  function handleLoadClick() {
    setValidationError(null);

    if (start > end) {
      setValidationError("La fecha de inicio debe ser anterior o igual a la fecha de fin.");
      return;
    }

    if (range) {
      const minDate = toDateInputValue(range.start);
      const maxDate = toDateInputValue(range.end);

      if (
        (minDate && start < minDate) ||
        (maxDate && start > maxDate) ||
        (minDate && end < minDate) ||
        (maxDate && end > maxDate)
      ) {
        setValidationError(
          `Las fechas seleccionadas deben estar dentro del rango disponible (${minDate} a ${maxDate}).`
        );
        return;
      }
    }

    loadHistoricalData(start, end);
  }

  const chartData = useMemo(() => {
    const byTimestamp = new Map<string, { timestamp: string; price?: number }>();

    data.forEach((row) => {
      byTimestamp.set(row.timestamp, {
        timestamp: row.timestamp,
        price: row.price,
      });
    });

    return Array.from(byTimestamp.values()).sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }, [data]);

  const filteredRows = useMemo(() => {
    const query = tableQuery.trim().toLowerCase();
    if (!query) {
      return data;
    }

    return data.filter((row) => {
      const rawTimestamp = row.timestamp.toLowerCase();
      const formattedTimestamp = formatTimestamp(row.timestamp).toLowerCase();
      return rawTimestamp.includes(query) || formattedTimestamp.includes(query);
    });
  }, [data, tableQuery]);

  const visibleRows = useMemo(() => {
    return filteredRows.slice(0, TABLE_LIMIT);
  }, [filteredRows]);

  return (
    <div className="page">
      <header className="page__header">
        <div>
          <h1 className="page__title">Analisis historico</h1>
          <p className="page__subtitle">
            Serie temporal del mercado electrico espanol. Visualizacion del historico de
            precios en EUR/MWh.
          </p>
          <p className="page__subtitle">Zona horaria: {APP_TIMEZONE}</p>
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
              min={range ? toDateInputValue(range.start) : undefined}
              max={range ? toDateInputValue(range.end) : undefined}
              onChange={(e) => {
                setStart(e.target.value);
                setValidationError(null);
              }}
            />
          </label>

          <label className="field">
            <span>Fecha de fin</span>
            <input
              type="date"
              value={end}
              min={range ? toDateInputValue(range.start) : undefined}
              max={range ? toDateInputValue(range.end) : undefined}
              onChange={(e) => {
                setEnd(e.target.value);
                setValidationError(null);
              }}
            />
          </label>

          <div className="field field--actions">
            <span className="field__hint">Carga el historico para el rango indicado.</span>
            <button className="btn btn--primary" onClick={handleLoadClick}>
              Cargar historico
            </button>
          </div>
        </div>

        {validationError && <p className="status status--error">{validationError}</p>}
      </section>

      <section className="card">
        <div className="card__header">
          <h2>Grafica de precios</h2>
          <p>Serie historica con precios horarios del mercado en EUR/MWh.</p>
        </div>

        <div className="chart-wrap">
          {loading && (
            <div className="loading-stack" aria-live="polite">
              <div className="loading-bar" />
              <p className="status">Cargando datos historicos...</p>
            </div>
          )}
          {!loading && historicalError && (
            <p className="status status--error">
              Error al cargar historicos: {historicalError}
            </p>
          )}
          {!loading && !historicalError && data.length === 0 && (
            <p className="status">No hay datos historicos disponibles para el rango seleccionado.</p>
          )}
          {!loading && !historicalError && data.length > 0 && (
            <PriceChart data={chartData} showForecastA={false} showForecastB={false} />
          )}
        </div>
      </section>

      <section className="card">
        <div className="card__header">
          <h2>Registros historicos</h2>
          <p>Detalle por timestamp con variables auxiliares.</p>
        </div>

        {loading && (
          <div className="loading-stack" aria-live="polite">
            <div className="loading-bar" />
            <p className="status">Cargando datos historicos...</p>
          </div>
        )}
        {!loading && historicalError && (
          <p className="status status--error">
            Error al cargar historicos: {historicalError}
          </p>
        )}
        {!loading && !historicalError && data.length === 0 && (
          <p className="status">No hay datos historicos disponibles para el rango seleccionado.</p>
        )}

        {!loading && !historicalError && data.length > 0 && (
          <>
            <div className="table-controls">
              <label className="table-controls__search">
                <span>Buscar por timestamp</span>
                <input
                  type="text"
                  placeholder="Ej: 2022-01-03 o 03/01/2022"
                  value={tableQuery}
                  onChange={(e) => setTableQuery(e.target.value)}
                />
              </label>
              <p className="table-controls__meta">
                Mostrando {visibleRows.length} de {filteredRows.length} filas filtradas
                (total {data.length}).
              </p>
            </div>

            {filteredRows.length === 0 ? (
              <p className="status">No hay resultados para esa busqueda.</p>
            ) : (
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Precio (EUR/MWh)</th>
                      <th>Prevision de demanda</th>
                      <th>Prevision de viento</th>
                      <th>Prevision solar</th>
                      <th>Hidraulica programada</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleRows.map((row) => (
                      <tr key={row.timestamp}>
                        <td>{formatTimestamp(row.timestamp)}</td>
                        <td>{formatNumber(row.price, 2)}</td>
                        <td>{formatNumber(row.demand_forecast, 1)}</td>
                        <td>{formatNumber(row.wind_forecast, 1)}</td>
                        <td>{formatNumber(row.solar_forecast, 1)}</td>
                        <td>{formatNumber(row.hydro_programmed, 1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}

export default HistoricalPage;

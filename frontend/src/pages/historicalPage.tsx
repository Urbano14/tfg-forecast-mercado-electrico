// Esta página muestra el análisis histórico de precios del mercado eléctrico español.
// Permite seleccionar un rango de fechas, visualizar la serie temporal de precios y consultar los datos en una tabla.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchHistoricalData,
  fetchHistoricalRange,
  type HistoricalDataPoint,
} from "../api/historicalApi";
import PriceChart from "../components/PriceChart";
import {
  APP_TIMEZONE,
  formatDate,
  formatTimestamp,
  toDateInputValue,
} from "../utils/date";
import { formatNumber } from "../utils/number";

function HistoricalPage() {
  // Numero maximo de filas visibles en la tabla para no saturar la interfaz.
  const TABLE_LIMIT = 50;
  const MAX_HISTORICAL_DAYS = 5;

  // Rango total disponible en backend/base de datos.
  const [range, setRange] = useState<{ start: string; end: string } | null>(null);

  // Datos historicos cargados para el rango seleccionado.
  const [data, setData] = useState<HistoricalDataPoint[]>([]);

  // historicalError: errores al pedir datos al backend.
  // validationError: errores detectados en frontend antes de llamar al backend.
  const [historicalError, setHistoricalError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Estado de carga para mostrar mensajes/spinners mientras se consultan datos.
  const [loading, setLoading] = useState(true);

  // Texto usado para filtrar la tabla por timestamp.
  const [tableQuery, setTableQuery] = useState("");

  // Fechas seleccionadas en los inputs de fecha.
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");

  // Convierte una fecha YYYY-MM-DD en Date sin depender del parseo automatico del navegador.
  function parseDateOnly(dateStr: string): Date {
    const [year, month, day] = dateStr.split("-").map(Number);
    return new Date(year, month - 1, day);
  }

  // Desplaza una fecha YYYY-MM-DD un numero concreto de dias y devuelve el mismo formato.
  function shiftDate(dateStr: string, days: number): string {
    const date = parseDateOnly(dateStr);
    date.setDate(date.getDate() + days);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  // Fuerza una fecha a permanecer dentro de un rango YYYY-MM-DD.
  function clampDate(dateStr: string, minDate: string, maxDate: string): string {
    if (dateStr < minDate) {
      return minDate;
    }

    if (dateStr > maxDate) {
      return maxDate;
    }

    return dateStr;
  }

  // Calcula la diferencia en dias para limitar el rango historico.
  function diffDays(startDateStr: string, endDateStr: string): number {
    const startDate = parseDateOnly(startDateStr);
    const endDate = parseDateOnly(endDateStr);
    return Math.round((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  }

  // Funcion reutilizable para cargar datos historicos desde el backend.
  // Recibe fecha de inicio y fin, llama a /historical y guarda la respuesta en el estado data.
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

  const availableMinDate = range ? toDateInputValue(range.start) : null;
  const availableMaxDate = range ? toDateInputValue(range.end) : null;
  const maxEndDate =
    availableMinDate && availableMaxDate && start
      ? clampDate(shiftDate(start, MAX_HISTORICAL_DAYS), availableMinDate, availableMaxDate)
      : availableMaxDate;

  // Al montar la pagina, primero se carga el rango disponible y despues un historico inicial.
  useEffect(() => {
    async function initializePage() {
      try {
        const rangeData = await fetchHistoricalRange();
        setRange(rangeData);
        const minDate = toDateInputValue(rangeData.start);
        const maxDate = toDateInputValue(rangeData.end);
        const defaultEnd = maxDate;
        const defaultStart = clampDate(
          shiftDate(defaultEnd, -MAX_HISTORICAL_DAYS),
          minDate,
          defaultEnd
        );

        setStart(defaultStart);
        setEnd(defaultEnd);

        await loadHistoricalData(defaultStart, defaultEnd);
      } catch (err) {
        setHistoricalError(err instanceof Error ? err.message : "Error desconocido");
        setLoading(false);
      }
    }

    initializePage();
  }, [loadHistoricalData]);

  // Se ejecuta al pulsar el boton de cargar historico.
  // Valida fechas en frontend antes de llamar al backend.
  function handleLoadClick() {
    setValidationError(null);

    if (!start || !end) {
      setValidationError("Selecciona un rango historico valido dentro del rango disponible.");
      return;
    }

    // En formato YYYY-MM-DD se pueden comparar strings para comprobar orden temporal.
    if (start > end) {
      setValidationError("La fecha de inicio debe ser anterior o igual a la fecha de fin.");
      return;
    }

    // Si conocemos el rango real disponible, bloqueamos peticiones fuera de ese rango.
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
          `Las fechas seleccionadas deben estar dentro del rango disponible (${formatDate(minDate)} a ${formatDate(maxDate)}).`
        );
        return;
      }
    }

    if (diffDays(start, end) > MAX_HISTORICAL_DAYS) {
      setValidationError(
        `El rango historico esta limitado a ${MAX_HISTORICAL_DAYS} dias para evitar cargas demasiado grandes.`
      );
      return;
    }

    loadHistoricalData(start, end);
  }

  function handleStartChange(nextStart: string) {
    setValidationError(null);
    setStart(nextStart);

    if (!range) {
      return;
    }

    const minDate = toDateInputValue(range.start);
    const maxDate = toDateInputValue(range.end);
    const allowedEnd = clampDate(
      shiftDate(nextStart, MAX_HISTORICAL_DAYS),
      minDate,
      maxDate
    );

    if (!end || end < nextStart || diffDays(nextStart, end) > MAX_HISTORICAL_DAYS) {
      setEnd(allowedEnd);
    }
  }

  function handleEndChange(nextEnd: string) {
    setValidationError(null);
    setEnd(nextEnd);

    if (!range) {
      return;
    }

    const minDate = toDateInputValue(range.start);
    const adjustedStart = clampDate(
      shiftDate(nextEnd, -MAX_HISTORICAL_DAYS),
      minDate,
      nextEnd
    );

    if (!start || start > nextEnd || diffDays(start, nextEnd) > MAX_HISTORICAL_DAYS) {
      setStart(adjustedStart);
    }
  }

  // Adapta los datos historicos al formato que espera PriceChart.
  const chartData = useMemo(() => {
    const byTimestamp = new Map<string, { timestamp: string; price?: number }>();

    data.forEach((row) => {
      byTimestamp.set(row.timestamp, {
        timestamp: row.timestamp,
        price: row.price,
      });
    });

    // La grafica recibe los puntos ordenados cronologicamente.
    return Array.from(byTimestamp.values()).sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }, [data]);

  // Filtra las filas de la tabla usando el texto introducido por el usuario.
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

  // Limita el numero de filas renderizadas en la tabla.
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

        {/* Tarjeta con el rango temporal disponible en la base de datos. */}
        {range && (
          <div className="range-card">
            <div className="range-card__label">Rango disponible</div>
            <div className="range-card__row">
              <span>Inicio:</span>
              <strong>{formatDate(range.start)}</strong>
            </div>
            <div className="range-card__row">
              <span>Fin:</span>
              <strong>{formatDate(range.end)}</strong>
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
              // min y max evitan seleccionar fechas fuera del rango disponible.
              min={availableMinDate ?? undefined}
              max={end || availableMaxDate || undefined}
              onChange={(e) => handleStartChange(e.target.value)}
            />
            <span className="field__hint">Seleccionada: {formatDate(start)}</span>
          </label>

          <label className="field">
            <span>Fecha de fin</span>
            <input
              type="date"
              value={end}
              min={start || availableMinDate || undefined}
              max={maxEndDate ?? undefined}
              onChange={(e) => handleEndChange(e.target.value)}
            />
            <span className="field__hint">Seleccionada: {formatDate(end)}</span>
          </label>

          <div className="field field--actions">
            <span className="field__hint">
              El rango historico esta limitado a 5 dias para evitar cargas demasiado grandes.
            </span>
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
            // En esta pagina solo se pinta historico; las lineas de forecast se desactivan.
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

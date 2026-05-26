//Componente  que pinta la gráfica de precios en el frontend

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatTimestamp } from "../utils/date";

// Punto que recibe la gráfica.
// timestamp es obligatorio porque se usa como eje X.
// price representa el histórico real y forecastA/forecastB las dos posibles predicciones comparadas.
interface ChartPoint {
  timestamp: string;
  price?: number;
  forecastA?: number;
  forecastB?: number;
}

// data contiene todos los puntos a pintar y los flags permiten mostrar/ocultar cada línea.
interface Props {
  data: ChartPoint[];
  showHistorical?: boolean;
  showForecastA?: boolean;
  showForecastB?: boolean;
}

// Tipo simplificado de las props que Recharts pasa al tooltip personalizado.
interface CustomTooltipProps {
  active?: boolean;
  label?: string | number;
  payload?: Array<{
    color?: string;
    dataKey?: string | number;
    name?: unknown;
    value?: unknown;
  }>;
}

function PriceChart({
  data,
  showHistorical = true,
  showForecastA = true,
  showForecastB = true,
}: Props) {
  // Convierte los nombres internos de las series en etiquetas legibles para el usuario.
  const formatSeriesLabel = (value: string) => {
    if (value === "price") {
      return "Precio historico (EUR/MWh)";
    }

    if (value === "forecastA") {
      return "Prediccion A (EUR/MWh)";
    }

    if (value === "forecastB") {
      return "Prediccion B (EUR/MWh)";
    }

    return value;
  };

  // Formatea los valores numéricos del tooltip con dos decimales y unidad.
  const formatTooltipValue = (value: unknown) => {
    if (typeof value === "number" && Number.isFinite(value)) {
      return `${value.toFixed(2)} EUR/MWh`;
    }

    return "-";
  };

  // Muestra la fecha formateada y los valores de las series visibles en ese timestamp.
  const CustomTooltip = ({ active, label, payload }: CustomTooltipProps) => {
    if (!active || !payload?.length) {
      return null;
    }

    return (
      <div className="chart-tooltip">
        <p className="chart-tooltip__label">{formatTimestamp(String(label))}</p>
        <ul className="chart-tooltip__list">
          {payload.map((entry) => (
            <li
              className="chart-tooltip__item"
              key={String(entry.dataKey ?? entry.name ?? "series")}
            >
              <span className="chart-tooltip__name">
                <span
                  className="chart-tooltip__swatch"
                  style={{ backgroundColor: entry.color, color: entry.color }}
                />
                {formatSeriesLabel(String(entry.name))}
              </span>
              <strong>{formatTooltipValue(entry.value)}</strong>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    // Contenedor de la gráfica. Ocupa todo el ancho disponible y mantiene una altura fija.
    <div style={{ width: "100%", height: 400 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />

          {/* Eje X: usa timestamp y lo formatea para mostrar fechas legibles. */}
          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => formatTimestamp(value)}
            minTickGap={28}
            interval="preserveStartEnd"
            tick={{ fontSize: 12, fill: "#94a3b8" }}
            tickLine={{ stroke: "rgba(255,255,255,0.08)" }}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
          />

          {/* Eje Y: representa el precio del mercado eléctrico en EUR/MWh. */}
          <YAxis
            tick={{ fontSize: 12, fill: "#94a3b8" }}
            tickLine={{ stroke: "rgba(255,255,255,0.08)" }}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
            label={{
              value: "Precio (EUR/MWh)",
              angle: -90,
              position: "insideLeft",
              fill: "#94a3b8",
              fontSize: 12,
            }}
          />

          <Tooltip
            content={<CustomTooltip />}
            cursor={{ stroke: "rgba(56, 189, 248, 0.22)", strokeWidth: 1 }}
          />

          {/* Leyenda de las líneas visibles. */}
          <Legend wrapperStyle={{ color: "#e2e8f0", paddingTop: 12 }} />

          {/* Línea del histórico real. Se pinta si showHistorical está activo. */}
          {showHistorical && (
            <Line
              type="monotone"
              dataKey="price"
              stroke="#38bdf8"
              dot={false}
              name="Precio historico (EUR/MWh)"
              strokeWidth={2.5}
              style={{ filter: "drop-shadow(0 0 4px #38bdf8)" }}
            />
          )}

          {/* Línea de la primera predicción. Se pinta discontinua para distinguirla del histórico. */}
          {showForecastA && (
            <Line
              type="monotone"
              dataKey="forecastA"
              stroke="#f59e0b"
              dot={false}
              name="Prediccion A (EUR/MWh)"
              strokeDasharray="6 4"
              strokeWidth={2.5}
              style={{ filter: "drop-shadow(0 0 4px #f59e0b)" }}
            />
          )}

          {/* Línea de la segunda predicción, usada para comparar dos modelos. */}
          {showForecastB && (
            <Line
              type="monotone"
              dataKey="forecastB"
              stroke="#2dd4bf"
              dot={false}
              name="Prediccion B (EUR/MWh)"
              strokeDasharray="3 3"
              strokeWidth={2.5}
              style={{ filter: "drop-shadow(0 0 4px #2dd4bf)" }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PriceChart;

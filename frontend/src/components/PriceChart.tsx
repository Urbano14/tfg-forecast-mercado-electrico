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

interface ChartPoint {
  timestamp: string;
  price?: number;
  forecastA?: number;
  forecastB?: number;
}

interface Props {
  data: ChartPoint[];
  showHistorical?: boolean;
  showForecastA?: boolean;
  showForecastB?: boolean;
}

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
  const formatSeriesLabel = (value: string) => {
    if (value === "price") {
      return "Precio historico (€/MWh)";
    }

    if (value === "forecastA") {
      return "Forecast A (€/MWh)";
    }

    if (value === "forecastB") {
      return "Forecast B (€/MWh)";
    }

    return value;
  };

  const formatTooltipValue = (value: unknown) => {
    if (typeof value === "number" && Number.isFinite(value)) {
      return `${value.toFixed(2)} €/MWh`;
    }

    return "-";
  };

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
    <div style={{ width: "100%", height: 400 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => formatTimestamp(value)}
            minTickGap={28}
            interval="preserveStartEnd"
            tick={{ fontSize: 12, fill: "#94a3b8" }}
            tickLine={{ stroke: "rgba(255,255,255,0.08)" }}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
          />
          <YAxis
            tick={{ fontSize: 12, fill: "#94a3b8" }}
            tickLine={{ stroke: "rgba(255,255,255,0.08)" }}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
            label={{
              value: "Precio (€/MWh)",
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
          <Legend wrapperStyle={{ color: "#e2e8f0", paddingTop: 12 }} />
          {showHistorical && (
            <Line
              type="monotone"
              dataKey="price"
              stroke="#38bdf8"
              dot={false}
              name="Precio historico (€/MWh)"
              strokeWidth={2.5}
              style={{ filter: "drop-shadow(0 0 4px #38bdf8)" }}
            />
          )}
          {showForecastA && (
            <Line
              type="monotone"
              dataKey="forecastA"
              stroke="#f59e0b"
              dot={false}
              name="Forecast A (€/MWh)"
              strokeDasharray="6 4"
              strokeWidth={2.5}
              style={{ filter: "drop-shadow(0 0 4px #f59e0b)" }}
            />
          )}
          {showForecastB && (
            <Line
              type="monotone"
              dataKey="forecastB"
              stroke="#2dd4bf"
              dot={false}
              name="Forecast B (€/MWh)"
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

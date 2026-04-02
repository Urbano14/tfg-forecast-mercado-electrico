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

function PriceChart({
  data,
  showHistorical = true,
  showForecastA = true,
  showForecastB = true,
}: Props) {
  const formatSeriesLabel = (value: string) => {
    if (value === "price") {
      return "Historical price";
    }

    if (value === "forecastA") {
      return "Forecast A";
    }

    if (value === "forecastB") {
      return "Forecast B";
    }

    return value;
  };

  const formatTooltipValue = (value: unknown) => {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value.toFixed(2);
    }

    return "-";
  };

  return (
    <div style={{ width: "100%", height: 400 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid stroke="#ccc" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => formatTimestamp(value)}
            minTickGap={28}
            interval="preserveStartEnd"
            tick={{ fontSize: 12 }}
          />
          <YAxis />
          <Tooltip
            labelFormatter={(value) => formatTimestamp(value)}
            formatter={(value, name) => [
              formatTooltipValue(value),
              formatSeriesLabel(String(name)),
            ]}
          />
          <Legend />
          {showHistorical && (
            <Line
              type="monotone"
              dataKey="price"
              stroke="#2f5b7c"
              dot={false}
              name="Historical price"
              strokeWidth={2.5}
            />
          )}
          {showForecastA && (
            <Line
              type="monotone"
              dataKey="forecastA"
              stroke="#3aa272"
              dot={false}
              name="Forecast A"
              strokeDasharray="6 4"
              strokeWidth={2.5}
            />
          )}
          {showForecastB && (
            <Line
              type="monotone"
              dataKey="forecastB"
              stroke="#d4832b"
              dot={false}
              name="Forecast B"
              strokeDasharray="3 3"
              strokeWidth={2.5}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PriceChart;

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

interface ChartPoint {
  timestamp: string;
  price?: number;
  forecast?: number;
}

interface Props {
  data: ChartPoint[];
}

function PriceChart({ data }: Props) {
  return (
    <div style={{ width: "100%", height: 400 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid stroke="#ccc" />
          <XAxis dataKey="timestamp" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="price" stroke="#8884d8" dot={false} name="Historical price" />
          <Line type="monotone" dataKey="forecast" stroke="#82ca9d" dot={false} name="Forecast" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PriceChart;
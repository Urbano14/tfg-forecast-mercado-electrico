import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/spot_es_processed.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["month"] = df["timestamp"].dt.month

monthly = df.groupby("month")["price"].agg(["mean", "std"]).reset_index()

print("Estadísticos por mes:")
print(monthly)

plt.figure()
plt.plot(monthly["month"], monthly["mean"])
plt.xticks(range(1, 13))
plt.title("Precio medio por mes (2020-2024)")
plt.xlabel("Mes")
plt.ylabel("Precio medio (€)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/spot_es_processed.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year

years = sorted(df["year"].unique())

plt.figure()

for y in years:
    yearly = df[df["year"] == y].set_index("timestamp")["price"]
    yearly.plot()

plt.legend(years)
plt.title("Comparativa anual del precio SPOT")
plt.xlabel("Fecha")
plt.ylabel("Precio (€)")
plt.show()


stats = df.groupby("year")["price"].agg(["mean", "std", "min", "max"])

print(stats)
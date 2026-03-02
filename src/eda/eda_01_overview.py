import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/spot_es_processed.parquet")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

plt.figure()
df["price"].plot()
plt.title("Precio mercado SPOT España (2020-2024)")
plt.xlabel("Fecha")
plt.ylabel("Precio (€)")
plt.tight_layout()
plt.show()
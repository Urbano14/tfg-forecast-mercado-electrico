import pandas as pd
import matplotlib.pyplot as plt

# Cargar dataset procesado
df = pd.read_parquet("data/processed/spot_es_processed.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extraer hora del día
df["hour"] = df["timestamp"].dt.hour

# Media y dispersión por hora
hourly = df.groupby("hour")["price"].agg(["mean", "std", "min", "max"]).reset_index()

print("Estadísticos por hora:")
print(hourly)

# Gráfica: media por hora
plt.figure()
plt.plot(hourly["hour"], hourly["mean"])
plt.title("Precio medio por hora del día (2020-2024)")
plt.xlabel("Hora del día")
plt.ylabel("Precio medio (€)")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
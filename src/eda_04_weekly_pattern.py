import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/spot_es_processed.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Día de la semana: 0=Lunes ... 6=Domingo
df["dow"] = df["timestamp"].dt.dayofweek

weekly = df.groupby("dow")["price"].agg(["mean", "std", "min", "max"]).reset_index()

# Etiquetas bonitas
dow_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
weekly["name"] = weekly["dow"].map(lambda x: dow_names[x])

print("Estadísticos por día de la semana:")
print(weekly[["name", "mean", "std", "min", "max"]])

# Gráfica: media por día de la semana
plt.figure()
plt.plot(weekly["name"], weekly["mean"])
plt.title("Precio medio por día de la semana (2020-2024)")
plt.xlabel("Día de la semana")
plt.ylabel("Precio medio (€)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
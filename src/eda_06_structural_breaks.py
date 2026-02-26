import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/spot_es_processed.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# Media móvil 30 días
rolling_mean = df["price"].rolling(window=24*30).mean()

# Volatilidad móvil 30 días
rolling_std = df["price"].rolling(window=24*30).std()

plt.figure()
rolling_mean.plot()
plt.title("Media móvil 30 días (2020-2024)")
plt.ylabel("Precio medio (€)")
plt.show()

plt.figure()
rolling_std.plot()
plt.title("Volatilidad móvil 30 días (2020-2024)")
plt.ylabel("Desviación estándar")
plt.show()
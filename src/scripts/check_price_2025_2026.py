import pandas as pd

df = pd.read_parquet("data/processed/spot_es_with_exogenous.parquet")

df["year"] = df["timestamp"].dt.year

stats = df.groupby("year")["price"].agg(
    ["count", "mean", "std", "min", "max"]
)

print(stats)

recent = df[df["timestamp"].dt.year >= 2025]

print("\nDescribe 2025-2026:")
print(recent["price"].describe())

print("\nTop 20 precios más altos:")
print(
    recent[["timestamp", "price"]]
    .sort_values("price", ascending=False)
    .head(20)
)

print("\nTop 20 precios más bajos:")
print(
    recent[["timestamp", "price"]]
    .sort_values("price", ascending=True)
    .head(20)
)

print("\nNegativos por año:")
print(df.groupby("year")["price"].apply(lambda s: (s < 0).sum()))

print("\nCeros por año:")
print(df.groupby("year")["price"].apply(lambda s: (s == 0).sum()))

print("\nNulos por año:")
print(df.groupby("year").apply(lambda g: g.isna().sum()))

test = df[df["timestamp"].dt.year >= 2025]

print("\nPrimeras filas test:")
print(test.head(10))

print("\nÚltimas filas test:")
print(test.tail(10))

import matplotlib.pyplot as plt

df_plot = df.set_index("timestamp")

plt.figure(figsize=(16, 5))
df_plot["price"].plot()
plt.title("Precio horario 2020-2026")
plt.tight_layout()
plt.savefig("results/price_2020_2026.png")
plt.close()

recent_plot = recent.set_index("timestamp")

plt.figure(figsize=(16, 5))
recent_plot["price"].plot()
plt.title("Precio horario 2025-2026")
plt.tight_layout()
plt.savefig("results/price_2025_2026.png")
plt.close()

print("\nGráficas guardadas en:")
print("results/price_2020_2026.png")
print("results/price_2025_2026.png")
import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = "data/processed/spot_es_with_exogenous.parquet"
COLS = [
    "price",
    "demand_forecast",
    "wind_forecast",
    "solar_forecast",
    "hydro_programmed",
]


df = pd.read_parquet(DATA_PATH)

missing_cols = [col for col in COLS if col not in df.columns]
if missing_cols:
    raise ValueError(f"Faltan columnas en el parquet: {missing_cols}")

corr = df[COLS].corr()

print("Matriz de correlacion:")
print(corr)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(COLS)))
ax.set_yticks(range(len(COLS)))
ax.set_xticklabels(COLS, rotation=45, ha="right")
ax.set_yticklabels(COLS)
ax.set_title("Matriz de correlacion entre variables")

for i in range(len(COLS)):
    for j in range(len(COLS)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlacion")
plt.tight_layout()
plt.show()

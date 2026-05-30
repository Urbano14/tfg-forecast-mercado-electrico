from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/spot_es_with_exogenous.parquet")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"] = df["timestamp"].dt.year

    print("Estadísticas por año de price:")
    print(df.groupby("year")["price"].agg(["count", "mean", "std", "min", "max"]))

    print("\nTop 20 precios más altos:")
    print(df[["timestamp", "price"]].sort_values("price", ascending=False).head(20))

    print("\nTop 20 precios más bajos:")
    print(df[["timestamp", "price"]].sort_values("price", ascending=True).head(20))

    print("\nNegativos por año:")
    print(df.groupby("year")["price"].apply(lambda s: int((s < 0).sum())))

    print("\nCeros por año:")
    print(df.groupby("year")["price"].apply(lambda s: int((s == 0).sum())))

    print("\nNulos por año:")
    print(df.groupby("year")["price"].apply(lambda s: int(s.isna().sum())))


if __name__ == "__main__":
    main()

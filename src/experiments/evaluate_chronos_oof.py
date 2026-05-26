from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Sirve para evaluar las predicciones OOF que AutoGluon/Chronos dejó guardadas dentro de la carpeta del modelo.
#OOF significa out-of-fold predictions. Son predicciones internas generadas por AutoGluon durante su proceso de validación/evaluación

DATA_PATH = Path("data/processed/spot_es_with_exogenous.parquet")

RESULTS_DIR = Path("results/chronos")

# Rutas a las predicciones OOF generadas internamente por AutoGluon para cada variante de Chronos-2.
FINETUNED_OOF_PATH = Path("models/chronos2_with_covariates/models/Chronos2FineTuned/utils/oof.pkl")
ZEROSHOT_OOF_PATH = Path("models/chronos2_with_covariates/models/Chronos2ZeroShot/utils/oof.pkl")


# Normaliza una columna temporal para que las predicciones y los valores reales puedan unirse correctamente por timestamp.
def normalize_timestamp_column(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column])

    try:
        df[column] = df[column].dt.tz_localize(None)
    except TypeError:
        pass

    return df


# Carga las predicciones OOF de Chronos desde el fichero oof.pkl de AutoGluon.
def load_oof_predictions(path: Path) -> pd.DataFrame:
    obj = pd.read_pickle(path)

    # AutoGluon guarda las predicciones OOF como una lista. Se valida para detectar cambios de formato o ficheros incorrectos.
    if not isinstance(obj, list):
        raise TypeError(f"{path} does not contain a list")
    if len(obj) == 0:
        raise ValueError(f"{path} contains an empty list")

    # Se toma el primer elemento de la lista, que contiene las predicciones OOF.
    first = obj[0]
    if isinstance(first, pd.DataFrame):
        df = first.copy()
    else:
        df = pd.DataFrame(first)

    df = df.reset_index()

    # Las predicciones deben tener timestamp y una columna mean, que es la predicción media de Chronos.
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} is missing 'timestamp' column after reset_index")
    if "mean" not in df.columns:
        raise KeyError(f"{path} is missing 'mean' column")

    # Se deja solo timestamp y predicción, y se renombra mean a y_pred para evaluar después.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[["timestamp", "mean"]].rename(columns={"mean": "y_pred"})
    return normalize_timestamp_column(df)


# Carga los precios reales del dataset final para compararlos con las predicciones OOF.
def load_true_values() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # price es la variable real. Se renombra a y_true para que quede claro que es el valor observado.
    df = df[["timestamp", "price"]].rename(columns={"price": "y_true"})
    return normalize_timestamp_column(df)


# Alinea predicciones y valores reales por timestamp y calcula MAE/RMSE.
def evaluate_predictions(name: str, pred_df: pd.DataFrame, true_df: pd.DataFrame) -> dict:
    pred_df = normalize_timestamp_column(pred_df)
    true_df = normalize_timestamp_column(true_df)

    # Une cada predicción con su precio real usando el timestamp como clave.
    merged = pred_df.merge(true_df, on="timestamp", how="left")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Comprueba que el merge no haya duplicado o perdido predicciones.
    if len(merged) != len(pred_df):
        raise ValueError(f"{name}: merged rows {len(merged)} != prediction rows {len(pred_df)}")

    # Si hay y_true nulos, significa que alguna predicción OOF no se ha podido alinear con un precio real del dataset.
    if merged["y_true"].isna().any():
        missing = int(merged["y_true"].isna().sum())
        raise ValueError(f"{name}: {missing} predictions could not be aligned with true values")

    # Calcula MAE y RMSE principales sobre las predicciones OOF alineadas.
    mae = float(mean_absolute_error(merged["y_true"], merged["y_pred"]))
    rmse = float(np.sqrt(mean_squared_error(merged["y_true"], merged["y_pred"])))

    # Guarda un CSV con timestamp, predicción y valor real para poder inspeccionar los errores manualmente.
    output_path = RESULTS_DIR / f"{name}_oof_predictions_aligned.csv"
    merged.to_csv(output_path, index=False)

    # Devuelve un resumen de métricas y rango temporal evaluado.
    return {
        "model": name,
        "n_points": int(len(merged)),
        "start_timestamp": merged["timestamp"].min().isoformat(),
        "end_timestamp": merged["timestamp"].max().isoformat(),
        "mae": mae,
        "rmse": rmse,
    }


# Evalúa las predicciones
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Carga los valores reales y las predicciones OOF de las dos variantes de Chronos.
    true_df = load_true_values()
    finetuned_pred_df = load_oof_predictions(FINETUNED_OOF_PATH)
    zeroshot_pred_df = load_oof_predictions(ZEROSHOT_OOF_PATH)

    # Alinea cada variante con los precios reales y calcula sus métricas.
    finetuned_result = evaluate_predictions("Chronos2FineTuned", finetuned_pred_df, true_df)
    zeroshot_result = evaluate_predictions("Chronos2ZeroShot", zeroshot_pred_df, true_df)

    results = [finetuned_result, zeroshot_result]

    # Guarda el resumen de métricas OOF en JSON.
    output_json = RESULTS_DIR / "chronos_oof_metrics.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Chronos2FineTuned: MAE={finetuned_result['mae']}, RMSE={finetuned_result['rmse']}")
    print(f"Chronos2ZeroShot: MAE={zeroshot_result['mae']}, RMSE={zeroshot_result['rmse']}")


if __name__ == "__main__":
    main()

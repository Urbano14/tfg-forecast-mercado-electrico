#script auxiliar de inspección
#Como AutoGluon guarda muchas cosas internas en .pkl, este script ayuda a abrir esos ficheros 
# y ver qué contienen: tipo de objeto, columnas, shape, índice, primeras filas, etc.

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]

# Lista de artefactos internos que AutoGluon ha generado al entrenar/evaluar Chronos.
# Este script no crea estos ficheros; solo los abre para inspeccionar su estructura.
ARTIFACTS = [
    ROOT / "models/chronos2_with_covariates/models/cached_predictions.pkl",
    ROOT / "models/chronos2_with_covariates/models/Chronos2FineTuned/utils/oof.pkl",
    ROOT / "models/chronos2_with_covariates/models/Chronos2ZeroShot/utils/oof.pkl",
    ROOT / "models/chronos2_with_covariates/utils/data/train.pkl",
]


# Sirve para entender qué tipo de estructura guarda AutoGluon en cada .pkl.
def inspect_object(name, obj) -> None:
    print(f"=== {name} ===")
    print(f"type: {type(obj)}")

    # Si el objeto tiene shape, se imprime. Esto es útil para DataFrames, arrays, etc.
    if hasattr(obj, "shape"):
        try:
            print(f"shape: {obj.shape}")
        except Exception as exc:
            print(f"shape: <error: {exc}>")

    # Si el objeto tiene índice, se imprime el tipo de índice y sus nombres.
    if hasattr(obj, "index"):
        try:
            index = obj.index
            index_names = getattr(index, "names", None)
            print(f"index type: {type(index)}")
            print(f"index names: {index_names}")
        except Exception as exc:
            print(f"index: <error: {exc}>")

    # Si el objeto tiene columnas, se imprimen para saber qué campos contiene.
    if hasattr(obj, "columns"):
        try:
            print(f"columns: {obj.columns}")
        except Exception as exc:
            print(f"columns: <error: {exc}>")

    # Si el objeto es un diccionario, se imprimen sus claves.
    if isinstance(obj, dict):
        try:
            print(f"keys: {list(obj.keys())}")
        except Exception as exc:
            print(f"keys: <error: {exc}>")

    # Si parece un DataFrame o tiene método head(), se imprimen las primeras filas.
    if isinstance(obj, pd.DataFrame) or hasattr(obj, "head"):
        try:
            print("head:")
            print(obj.head())
        except Exception as exc:
            print(f"head: <error: {exc}>")

    print()


# Carga un artefacto .pkl.
# Primero intenta con pandas, y si falla usa pickle directamente.
# Esto se hace porque AutoGluon puede guardar objetos con distintas estructuras internas.
def load_artifact(path: Path):
    try:
        return pd.read_pickle(path), "pd.read_pickle"
    except Exception as read_exc:
        print(f"[load error] {path} with pd.read_pickle: {read_exc}")

    try:
        with path.open("rb") as f:
            return pickle.load(f), "pickle.load"
    except Exception as pickle_exc:
        print(f"[load error] {path} with pickle.load: {pickle_exc}")

    return None, None


# Inspecciona cached_predictions.pkl.
# Este fichero puede contener predicciones cacheadas por AutoGluon para distintos modelos o configuraciones.
def inspect_cached_predictions(obj) -> None:
    if not isinstance(obj, dict):
        print("cached_predictions: object is not a dict")
        print()
        return

    keys = list(obj.keys())
    print(f"cached_predictions keys count: {len(keys)}")

    # Solo se inspeccionan las primeras claves para no imprimir demasiado.
    for idx, key in enumerate(keys[:3], start=1):
        value = obj[key]
        print(f"cached_predictions key {idx}: {key}")
        print(f"value type: {type(value)}")

        if hasattr(value, "shape"):
            try:
                print(f"value shape: {value.shape}")
            except Exception as exc:
                print(f"value shape: <error: {exc}>")

        if hasattr(value, "index"):
            try:
                print(f"value index type: {type(value.index)}")
                print(f"value index names: {getattr(value.index, 'names', None)}")
            except Exception as exc:
                print(f"value index: <error: {exc}>")

        if hasattr(value, "columns"):
            try:
                print(f"value columns: {value.columns}")
            except Exception as exc:
                print(f"value columns: <error: {exc}>")

        if isinstance(value, dict):
            try:
                print(f"value dict keys: {list(value.keys())}")
            except Exception as exc:
                print(f"value dict keys: <error: {exc}>")

        if isinstance(value, list):
            print(f"value list length: {len(value)}")
            first_type = type(value[0]) if value else None
            print(f"value first element type: {first_type}")

        if isinstance(value, pd.DataFrame) or hasattr(value, "head"):
            try:
                print("value head:")
                print(value.head())
            except Exception as exc:
                print(f"value head: <error: {exc}>")

        print()


# Inspecciona los oof.pkl de Chronos.
# Esta función permite ver si vienen como lista, DataFrame, qué columnas tienen, etc.
def inspect_oof(name, obj) -> None:
    if not isinstance(obj, list):
        print(f"{name}: object is not a list")
        print()
        return

    print(f"{name} length: {len(obj)}")

    # Se inspeccionan solo los primeros elementos para entender la estructura sin saturar la consola.
    for idx, element in enumerate(obj[:3], start=1):
        print(f"{name} element {idx} type: {type(element)}")

        if hasattr(element, "shape"):
            try:
                print(f"{name} element {idx} shape: {element.shape}")
            except Exception as exc:
                print(f"{name} element {idx} shape: <error: {exc}>")

        if hasattr(element, "index"):
            try:
                print(f"{name} element {idx} index type: {type(element.index)}")
                print(f"{name} element {idx} index names: {getattr(element.index, 'names', None)}")
            except Exception as exc:
                print(f"{name} element {idx} index: <error: {exc}>")

        if hasattr(element, "columns"):
            try:
                print(f"{name} element {idx} columns: {element.columns}")
            except Exception as exc:
                print(f"{name} element {idx} columns: <error: {exc}>")

        if isinstance(element, dict):
            try:
                print(f"{name} element {idx} dict keys: {list(element.keys())}")
            except Exception as exc:
                print(f"{name} element {idx} dict keys: <error: {exc}>")

        if isinstance(element, tuple):
            print(f"{name} element {idx} tuple length: {len(element)}")
            print(f"{name} element {idx} tuple element types: {[type(item) for item in element]}")

        if isinstance(element, pd.DataFrame) or hasattr(element, "head"):
            try:
                print(f"{name} element {idx} head:")
                print(element.head())
            except Exception as exc:
                print(f"{name} element {idx} head: <error: {exc}>")

        print()


# Es un script de diagnóstico: no entrena modelos y no calcula métricas finales.
def main() -> None:
    print(f"repo root: {ROOT}")
    print()

    # Recorre cada artefacto de Chronos/AutoGluon que queremos inspeccionar.
    for path in ARTIFACTS:
        print(f"file: {path}")

        # Si el fichero no existe, se informa y se pasa al siguiente.
        if not path.exists():
            print("status: does not exist")
            print()
            continue

        # Carga el artefacto con pandas o pickle.
        obj, loader = load_artifact(path)
        if loader is None:
            print("status: could not load")
            print()
            continue

        print(f"loaded with: {loader}")

        # Imprime información general del objeto cargado.
        inspect_object(path.name, obj)

        # Si es cached_predictions, se hace una inspección específica de predicciones cacheadas.
        if path.name == "cached_predictions.pkl":
            inspect_cached_predictions(obj)

        # Si es un oof.pkl, se inspecciona como predicción out-of-fold.
        elif path.name == "oof.pkl":
            inspect_oof(str(path), obj)


if __name__ == "__main__":
    main()

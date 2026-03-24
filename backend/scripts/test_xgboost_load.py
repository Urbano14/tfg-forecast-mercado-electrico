from app.infrastructure.ml.xgboost_loader import load_xgboost_model


def main():
    model = load_xgboost_model()
    print(type(model))
    print("XGBoost model loaded successfully")


if __name__ == "__main__":
    main()
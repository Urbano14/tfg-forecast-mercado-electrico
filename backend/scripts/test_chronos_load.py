from app.infrastructure.ml.chronos_loader import load_chronos_predictor


def main():
    predictor = load_chronos_predictor()
    print(type(predictor))
    print("Chronos predictor loaded successfully")


if __name__ == "__main__":
    main()

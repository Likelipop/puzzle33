from src.data_pipeline import build_local_training_data


if __name__ == "__main__":
    path = build_local_training_data()
    print(f"Generated training data at {path}")

from pathlib import Path
import numpy as np
import pandas as pd


def generate_classification_data(n_samples=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    X1 = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
    X2 = rng.normal(loc=3.0, scale=1.5, size=(n_samples, 2))
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])

    df = pd.DataFrame(X, columns=["feature_a", "feature_b"])
    df["target"] = y.astype(int)
    return df


def save_data(df: pd.DataFrame, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def load_data(data_path: str):
    return pd.read_csv(data_path)


def build_local_training_data(output_path: str = "data/processed/training_data.csv", n_samples: int = 1000):
    df = generate_classification_data(n_samples=n_samples)
    return save_data(df, output_path)

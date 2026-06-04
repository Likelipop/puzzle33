import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        st.error("No trained model found. Please train a model in Colab and save it to models/model.joblib.")
        return None
    return joblib.load(MODEL_PATH)


def main():
    st.title("Puzzle33 Model Deployment")
    st.write("This app loads a trained model and makes predictions from user inputs.")

    model = load_model()
    if model is None:
        return

    feature_a = st.slider("Feature A", -5.0, 8.0, 0.0)
    feature_b = st.slider("Feature B", -5.0, 8.0, 0.0)

    input_df = pd.DataFrame({"feature_a": [feature_a], "feature_b": [feature_b]})
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0].max()
        st.success(f"Predicted class: {int(prediction)}")
        st.write(f"Confidence: {proba:.2f}")

    st.write("---")
    st.write("### Example input")
    st.write(input_df)


if __name__ == "__main__":
    main()

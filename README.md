# Puzzle33 Training Workspace

This workspace is structured for local dataset preparation, Colab training, and Streamlit deployment.

## Structure

- `notebooks/` — Jupyter/Colab notebook templates and examples.
- `data/raw/` — raw input data.
- `data/processed/` — processed training data ready for training.
- `models/` — saved model artifacts.
- `src/` — reusable Python modules for data generation and model training.
- `scripts/` — local pipeline scripts.
- `streamlit_app/` — Streamlit app for deployment.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Generate training data locally:
   ```bash
   python scripts/prepare_data.py
   ```

3. Transfer data to Colab:
   - Copy `data/processed/training_data.csv` to Google Drive, or
   - Use `scripts/push_to_drive.sh` with a mounted drive path:
     ```bash
     DRIVE_PATH=/path/to/GoogleDrive ./scripts/push_to_drive.sh
     ```

4. Open `notebooks/training_colab.ipynb` in Colab and train the model there.

5. Save the trained model to `models/model.joblib` and optionally download it.

6. Run Streamlit for deployment:
   ```bash
   streamlit run streamlit_app/app.py
   ```

## Workflow

- Local: generate data and prepare it for training.
- Colab: import the dataset, train a model, and persist it.
- Streamlit: load the trained model and serve predictions.

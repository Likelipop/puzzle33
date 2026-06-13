# Puzzle33 Training Workspace

This repository is organized to support the complete workflow of **dataset preparation**, **model training**, and **deployment**.

For the best experience when exploring the project, use the notebooks provided in the `notebooks/` directory. Due to the limited computational resources of my local machine, all model training was performed using **Google Colab**, while data preparation and deployment were handled locally.

You can either:

- Run the notebooks to reproduce the data preparation, training, and inference process.
- Clone the repository and launch the Streamlit application locally for a more interactive experience.

---

## Project Structure

```text
.
├── notebooks/        # Jupyter/Colab notebooks for training and experimentation
├── model/
│   ├── ViT/          # trained model (onnx)
├── src/              # Reusable Python modules
├── scripts/          # Data processing and training scripts
└── streamlit_app/    # Streamlit deployment application
````

---

## Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model

Open `notebooks/training_colab.ipynb` in Google Colab and execute the notebook to train the model.

### 3. Launch the Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

The application will load the trained model and provide an interface for making predictions.

---

## Workflow

### Data Preparation (Local)

* Generate or collect raw data.
* Preprocess and transform the data.
* Save the processed dataset to `data/processed/`.

### Model Training (Google Colab)

* Upload or access the processed dataset.
* Train the model using Colab's compute resources.
* Save trained model artifacts to the `models/` directory.

### Deployment (Local)

* Load the trained model.
* Launch the Streamlit application.
* Perform inference through the web interface.

---

## Notes

* Training was conducted on Google Colab due to hardware limitations on the local machine.
* The notebooks contain the complete workflow for data preparation, training, and inference.
* The Streamlit application provides a user-friendly interface for interacting with the trained model.


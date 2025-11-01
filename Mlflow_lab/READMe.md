# 🎮 Video Game Blockbuster Prediction - MLflow Lab

A complete end-to-end MLOps pipeline using MLflow to predict whether a video game will be a blockbuster (>1M sales) based on platform, year, genre, and publisher.

## 🎯 Project Overview

This project demonstrates a production-ready MLOps workflow using **MLflow** for experiment tracking, model management, and deployment. We built a machine learning pipeline to predict video game success (blockbuster vs. regular) using historical sales data.

**Problem Statement:** Predict whether a video game will become a blockbuster (≥1M global sales) based on its features.

**Target Variable:** Binary classification
- `1` = Blockbuster (≥1M sales)
- `0` = Regular game (<1M sales)

---

## ✨ Features

- **Complete MLflow Pipeline**: Experiment tracking, model registry, and serving
- **Multiple Model Comparison**: Logistic Regression, Random Forest, XGBoost
- **Hyperparameter Tuning**: Automated optimization using Hyperopt with 30+ trials
- **Model Versioning**: Full lifecycle management (Staging → Production)
- **REST API Deployment**: Serve models as HTTP endpoints
- **Comprehensive Visualizations**: Confusion matrices, feature importance, model comparisons
- **Reproducible Experiments**: All runs tracked with parameters, metrics, and artifacts

---

## 🚀 Installation

### Prerequisites
- Python 3.12
- Conda or virtualenv
- Homebrew (for Mac users)

### Setup Instructions

1. **Clone or download the project:**
```bash
cd Mlflow_lab
```

2. **Create and activate virtual environment:**
```bash
# Using Conda
conda create -n MLOPs_env python=3.12 -y
conda activate MLOPs_env

# Or using venv
python -m venv MLOPs_env
source MLOPs_env/bin/activate  # On Mac/Linux
# MLOPs_env\Scripts\activate   # On Windows
```
3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **For Mac users (XGBoost requirement):**
```bash
brew install libomp
```

5. **Download dataset:**
- Download `data.csv` from [Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales)
- Place it in the project root directory

---

## 💻 Usage

### 1. Run the Jupyter Notebook
```bash
jupyter notebook video_game_mlflow.ipynb
```

Run all cells sequentially to:
- Load and preprocess data
- Train multiple models
- Perform hyperparameter tuning
- Register and deploy the best model

### 2. View MLflow UI

**In a separate terminal:**
```bash
conda activate MLOPs_env
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Open in browser:** http://localhost:5000

### 3. Serve the Model (REST API)

**In another terminal:**
```bash
conda activate MLOPs_env
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow models serve -m 'models:/video_game_blockbuster_predictor/Production' -p 5001 --env-manager local
```


### 4. Make Predictions via API
```python
import requests
import json

url = 'http://localhost:5001/invocations'
headers = {'Content-Type': 'application/json'}

data = {
    "dataframe_split": {
        "columns": ["Platform", "Year", "Genre", "Publisher"],
        "data": [[20, 2015.0, 0, 100]]  # Encoded values
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))
prediction = response.json()
print(f"Prediction: {'Blockbuster' if prediction['predictions'][0] == 1 else 'Regular'}")
```

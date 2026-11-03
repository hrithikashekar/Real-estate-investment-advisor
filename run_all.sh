#!/usr/bin/env bash
# Full pipeline: clean -> EDA -> train (MLflow) -> launch app
set -e
python src/preprocess.py
python src/eda.py
python src/train.py
echo "MLflow UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db"
streamlit run app/app.py

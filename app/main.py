from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os
import mlflow
import time

app = FastAPI(title="API Previsão Preço Ações Banco do Brasi")

MODEL_PATH = "model/prophet_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

class PredictionRequest(BaseModel):
    future_dates: List[str]

@app.get("/", summary="Página inicial")
def root():
    return {"mensagem": "API em funcionamento"}

@app.post("/predict", summary="Retorna previsão valores de fechamento futuros")
def predict(request: PredictionRequest):
    try:
        start_time = time.time()

        future_df = pd.DataFrame({'ds': pd.to_datetime(request.future_dates)})

        forecast = model.predict(future_df)

        result_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Price'})
        
        result = result_df.to_dict(orient="records")

        duration = time.time() - start_time

        mlflow.set_experiment("previsao-acoes-bbas3")

        with mlflow.start_run(nested=True):

            mlflow.log_param("input_dates", request.future_dates)
            mlflow.log_param("model_type", "Prophet")
            mlflow.log_metric("latency", duration)
            mlflow.log_metric("prediction_count", len(result))
            mlflow.log_dict(result, "forecast.json")

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
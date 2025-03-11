from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from predict import PipelinePredictor

# Инициализируем предиктор
predictor = PipelinePredictor()

app = FastAPI()

# Определяем формат данных для запроса
class CarFeatures(BaseModel):
    Doors: int
    Year: int
    Owner_Count: int
    Brand: str
    Model: str
    Fuel_Type: str
    Transmission: str
    Engine_Size: float
    Mileage: float
    

@app.post("/predict")
def predict(features: CarFeatures):
    # Принимаем данные и преобразуем в DataFrame
    input_data = pd.DataFrame([features.model_dump()])
    # Предсказываем цену с предиктора
    prediction = predictor.predict(input_data)
    
    return {"prediction": prediction[0]}

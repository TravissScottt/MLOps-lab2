from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from .predict import PipelinePredictor

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

@app.get("/")
def home():
    return {"message": "API is working!"}

@app.post("/predict")
def predict(features: CarFeatures):
    # Принимаем данные и преобразуем в DataFrame
    input_data = pd.DataFrame([features.model_dump()])
    # Предсказываем цену с предиктора
    prediction = predictor.predict(input_data)
    
    return {"prediction": prediction[0]}

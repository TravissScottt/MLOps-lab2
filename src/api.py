import uvicorn
import pickle
from joblib import load
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import configparser
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from logger import Logger

# Создаем API
app = FastAPI(title="Car Price Prediction")

# Загружаем конфигурацию
config = configparser.ConfigParser()
config.read("config.ini")

# Загружаем модель
try:
    model_path = config["RAND_FOREST"]["path"]
    # model = pickle.load(open(model_path, "rb"))
    model = load(model_path)
except FileNotFoundError:
    raise Exception("Ошибка: модель не найдена!")

# Настраиваем логгер
logger = Logger(show=True)
log = logger.get_logger(__name__)

# Определяем числовые, порядковые и категориальные признаки
ordinal_columns = ["Doors", "Year", "Owner_Count"]
categorical_columns = ["Brand", "Model", "Fuel_Type", "Transmission"]
num_columns = ["Engine_Size", "Mileage"]

# Создаем энкодеры и скейлер
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = MinMaxScaler()

# Загружаем тренировочные данные (чтобы энкодеры и скейлер обучились)
try:
    X_train = pd.read_csv(config["SPLIT_DATA"]["X_train"], index_col=0)
    ordinal_encoder.fit(X_train[ordinal_columns])
    scaler.fit(X_train[num_columns])
except FileNotFoundError:
    raise Exception("Ошибка: тренировочные данные не найдены!")

# Определяем формат входных данных
class InputData(BaseModel):
    Engine_Size: float
    Mileage: int
    Year: int
    Doors: int
    Owner_Count: int
    Brand: str
    Model: str
    Fuel_Type: str
    Transmission: str

@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Преобразуем входные данные в DataFrame
        df = pd.DataFrame([data.model_dump()])

        log.info(f"Фичи в `df` перед предсказанием: {list(df.columns)}")
        # Кодируем порядковые признаки
        df[ordinal_columns] = ordinal_encoder.transform(df[ordinal_columns])
        log.info(f"Фичи в `df` перед предсказанием: {list(df.columns)}")
        # Кодируем категориальные признаки
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype="int")
        df, _ = df.align(X_train, join="left", axis=1, fill_value=0)
        log.info(f"Фичи в `df` перед предсказанием: {list(df.columns)}")
        # Масштабируем числовые признаки
        df[num_columns] = scaler.transform(df[num_columns])
        log.info(f"Фичи в `df` перед предсказанием: {list(df.columns)}")
        log.info(f"Фичи в `X_train`: {list(X_train.columns)}")
        log.info(f"Фичи в `df` перед предсказанием: {list(df.columns)}")

        # Делаем предсказание
        price_pred = model.predict(df)[0]

        log.info(f"Предсказанная цена: {price_pred:.2f}")

        return {"predicted_price": round(price_pred, 2)}

    except Exception as e:
        log.error(f"Ошибка при предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка сервера")


# Запуск сервера (локально)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

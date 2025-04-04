import pytest
import json
import requests
import os
import sys
from sklearn.metrics import r2_score

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from logger import Logger

SHOW_LOG = True
logger = Logger(SHOW_LOG).get_logger(__name__)

# Получаем пути до тестовых JSON-файлов
BASE_DIR = os.path.dirname(__file__)
TESTS_DIR = os.path.join(BASE_DIR, "test_data")

# URL API, запущенного в контейнере
SERVER_URL = "http://localhost:8000/predict"

def test_functional_predict_live():
    """Функциональный тест API"""
    test_files = [os.path.join(TESTS_DIR, f) for f in os.listdir(TESTS_DIR) if f.endswith('.json')]
    
    all_inputs = []
    all_expected = []
    
    # Собираем входные данные и ожидаемые результаты
    for test_file in test_files:
        with open(test_file, "r") as f:
            test_data = json.load(f)
        all_inputs.append(test_data["X"])
        all_expected.append(test_data["y"]["prediction"])
    
    # Отправляем запросы и собираем предсказания
    predictions = []
    for input_data in all_inputs:
        response = requests.post(SERVER_URL, json=input_data)
        assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
        result = response.json()
        assert "prediction" in result, "No prediction in response"
        predictions.append(result["prediction"])
    
    # Вычисляем R²-метрику по всем примерам
    r2 = r2_score(all_expected, predictions)
    logger.info(f"Functional test R2 Score: {r2:.4f}")
    
    # Проверяем, что R² выше порога
    assert r2 >= 0.8, f"R2 Score {r2:.4f} is below acceptable threshold"

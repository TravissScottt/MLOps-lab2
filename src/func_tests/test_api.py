import pytest
import json
import math
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from api import app  # Импортируем API

client = TestClient(app)  # Создаём тестового клиента

TESTS_DIR = os.path.join(os.getcwd(), "tests")

@pytest.mark.parametrize("test_file", [
    os.path.join(TESTS_DIR, "test_0.json"),
    os.path.join(TESTS_DIR, "test_1.json")
])
def test_functional_predict(test_file):
    """
        Функциональный тест API с JSON-файлами
    """
    # Открываем тест
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    # Отправляем POST-запрос в API
    response = client.post("/predict", json=test_data["X"])
    
    # Проверяем, что API вернул 200 OK
    assert response.status_code == 200

    # Проверяем, что в ответе есть предсказание
    assert "prediction" in response.json()

    # Ожидаемое и полученное предсказание
    y_true = test_data["y"]["prediction"]
    y_pred = response.json()["prediction"]

    # Проверяем, что предсказание близко к ожидаемому (допуск ±1%)
    assert math.isclose(y_pred, y_true, rel_tol=0.01)

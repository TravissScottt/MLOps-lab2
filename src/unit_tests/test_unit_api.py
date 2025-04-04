import os
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

# Сохраняем оригинальные модули
original_predict = sys.modules.get('predict')
original_database = sys.modules.get('database')
original_logger = sys.modules.get('logger')

# Мокаем модули
import unittest.mock
sys.modules['predict'] = unittest.mock.Mock()
sys.modules['database'] = unittest.mock.Mock()
sys.modules['logger'] = unittest.mock.Mock()

from api import app

# Фикстура для создания тестового клиента
@pytest.fixture
def client(mocker):
    # Мокаем зависимости
    mocker.patch('src.api.PipelinePredictor')
    mocker.patch('src.api.MongoDBConnector')
    mocker.patch('src.api.Logger')

    api = app.state.api
    api.predictor.predict.return_value = [42.0]
    api.db.predictions.insert_one.return_value.inserted_id = "mocked_id"
    api.logger.info = mocker.Mock()
    api.logger.error = mocker.Mock()
    
    return TestClient(api.get_app())

def test_health_check(client):
    '''Проверяем health_check'''
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"health_check": "OK"}

def test_predict(client):
    '''Проверяем predict-ручку'''
    car_data = {
        "Doors": 4,
        "Year": 2020,
        "Owner_Count": 1,
        "Brand": "Toyota",
        "Model": "Camry",
        "Fuel_Type": "Petrol",
        "Transmission": "Automatic",
        "Engine_Size": 2.5,
        "Mileage": 30000.0
    }

    response = client.post("/predict", json=car_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": 42.0}
    
    # Проверяем, что запись в базу произошла
    client.app.state.api.logger.info.assert_called_once_with("Prediction saved with id: mocked_id")
    
# Восстанавливаем модули после тестов
def pytest_sessionfinish():
    if original_predict is not None:
        sys.modules['predict'] = original_predict
    else:
        sys.modules.pop('predict', None)
    if original_database is not None:
        sys.modules['database'] = original_database
    else:
        sys.modules.pop('database', None)
    if original_logger is not None:
        sys.modules['logger'] = original_logger
    else:
        sys.modules.pop('logger', None)
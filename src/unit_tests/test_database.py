import os
import sys
import pytest

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from database import MongoDBConnector

# Фикстура для создания тестового коннектора
@pytest.fixture
def mongo_connector(mocker):
    mocker.patch('pymongo.MongoClient')  # Мокаем MongoClient
    connector = MongoDBConnector()
    connector.db_config = {
        "host": "localhost",
        "port": 27017,
        "user": "testuser",
        "password": "testpass",
        "name": "testdb"
    }
    return connector

def test_get_database_success(mongo_connector, mocker):
    # Настраиваем мок для успешного подключения
    mock_client = mongo_connector.client
    mock_client.admin.command.return_value = {"ok": 1}  # Мок пинга
    
    db = mongo_connector.get_database()

    # Проверяем, что возвращается база
    assert db == mock_client["testdb"]
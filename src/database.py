import configparser
import sys
import os
from pymongo import MongoClient
from logger import Logger

SHOW_LOG = True
logger = Logger(SHOW_LOG).get_logger(__name__)

def get_database():
    """
    Читает параметры подключения к MongoDB из config_secret.ini, устанавливает соединение
    и возвращает объект базы данных.
    """
    # Определяем путь к конфигу
    base_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(base_dir, '..', 'config_secret.ini')
    
    config = configparser.ConfigParser()
    config.read(config_file_path)
    
    if 'DATABASE' not in config:
        logger.error(f"В файле {config_file_path} не найден раздел [DATABASE].")
        sys.exit(1)
    
    db_config = config['DATABASE']
    host = db_config.get('host')
    port = db_config.getint('port')
    user = db_config.get('user')
    password = db_config.get('password')
    dbname = db_config.get('name')
    
    # Формируем URI подключения
    uri = f"mongodb://{user}:{password}@{host}:{port}/{dbname}?authSource=admin"
    
    try:
        client = MongoClient(uri)
        # Проверка подключения
        client.admin.command('ping')
        logger.info(f"Успешное подключение к базе данных '{dbname}' на {host}:{port}")
    except Exception as e:
        logger.error("Ошибка подключения к MongoDB", exc_info=True)
        sys.exit(1)
    
    return client[dbname]

if __name__ == "__main__":
    db = get_database()
    logger.info(f"База данных подключена: {db.name}")
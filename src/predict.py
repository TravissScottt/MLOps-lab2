import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
from sklearn.metrics import r2_score
from pickle import load
import shutil
import sys
import traceback
import yaml

from .logger import Logger

SHOW_LOG = True


class PipelinePredictor():
    def __init__(self) -> None:
        # Создаем объекты логера и конфигуратора
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        
        # Путь к пайплайну
        self.pipeline_path = self.config["RAND_FOREST"]["path"]
        
        # Загружаем пайплайн
        try:
            with open(self.pipeline_path, "rb") as f:
                self.pipeline = load(f)
            self.log.info("Пайплайн успешно загружен")
        except FileNotFoundError:
            self.log.error("Файл с пайплайном не найден")
            sys.exit(1)

    def predict(self, X_input: pd.DataFrame) -> float:
        """
            Предсказание через API
        """
        return self.pipeline.predict(X_input)
    
    def test(self) -> bool:
        """
            Тестирование модели
        """
        # Создаем парсер для аргументов
        parser = argparse.ArgumentParser(description="Predictor")
        # Выбор типа теста:
        # smoke -> быстрая проверка работы модели (Smoke Test)
        # func -> функциональные тесты на готовых JSON-файлах
        parser.add_argument("--test", "-t",
                                 type=str,
                                 help="Тип тестирования (smoke или func)",
                                 default="smoke",
                                 choices=["smoke", "func"])
        
        # Передаем аргументы
        args = parser.parse_args()
            
        # Smoke test    
        if args.test == "smoke":
            try:
                X = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
                y = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0).values.ravel()
                y_pred = self.pipeline.predict(X)
                r2 = r2_score(y, y_pred)
                self.log.info(f"Smoke test пройден. R2: {r2:.4f}")
                              
            except Exception:
                self.log.error("Ошибка во время smoke теста")
                self.log.error(traceback.format_exc())
                sys.exit(1)
        
        # Functional test                      
        elif args.test == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            
            # Проходимся по всем тестам
            for test in os.listdir(tests_path):
                test_file = os.path.join(tests_path, test)
                
                # Загружаем данные
                with open(test_file) as f:
                    test_data = json.load(f)

                X = pd.json_normalize(test_data, record_path=['X'])
                y = pd.json_normalize(test_data, record_path=['y']).values.ravel()

                try:
                    # Тестируем модель
                    y_pred = self.pipeline.predict(X)
                    r2 = r2_score(y, y_pred)
                    self.log.info(f"Func test {test} пройден. R2: {r2:.4f}")

                    # Параметры эксперимента
                    exp_dir = os.path.join(os.getcwd(), "experiments", f"experiment_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
                    os.makedirs(exp_dir, exist_ok=True)
                    
                    config_data = {
                        "model_params": dict(self.config.items("RAND_FOREST")),
                        "test_data_path": test_file,
                        "model_path": self.pipeline_path
                    }

                    metrics_data = {
                        "R2_score": r2
                    }
                    
                    # Сохраняем данные эксперимента
                    with open(os.path.join(exp_dir, "config.yaml"), 'w') as cfg_f:
                        yaml.safe_dump(config_data, cfg_f, sort_keys=False)

                    with open(os.path.join(exp_dir, "metrics.yaml"), 'w') as metrics_f:
                        yaml.safe_dump(metrics_data, metrics_f, sort_keys=False)

                    shutil.copy("logfile.log", os.path.join(exp_dir, 'logs.txt'))

                    self.log.info(f"Функциональный тест {test} завершен успешно")

                except Exception:
                    self.log.error(f"Ошибка во время функционального теста: {test}")
                    self.log.error(traceback.format_exc())
                    sys.exit(1)


if __name__ == "__main__":
    predictor = PipelinePredictor()
    predictor.test()

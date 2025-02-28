import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import shutil
import sys
import time
import traceback
import yaml

from logger import Logger

SHOW_LOG = True


class Predictor():
    def __init__(self) -> None:
        # Создаем объекты логера и конфигуратора
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        
        # Создаем парсер для передачи аргументов через командную строку
        self.parser = argparse.ArgumentParser(description="Predictor")
        # Выбор типа теста:
        # smoke -> быстрая проверка работы модели (Smoke Test)
        # func -> функциональные тесты на готовых JSON-файлах
        self.parser.add_argument("-t", "--tests",
                                 type=str,
                                 help="Select tests",
                                 required=True,
                                 default="smoke",
                                 const="smoke",
                                 nargs="?",
                                 choices=["smoke", "func"])
        
        # Загружаем данные
        try:
            self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
            self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
            self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
            self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)
            self.log.info("Данные успешно загружены")
        except FileNotFoundError:
            self.log.error("Данные не найдены!")
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        try:    
            # Кодируем порядковые признаки
            self.ordinal_encoder = OrdinalEncoder()
            self.ordinal_columns = ["Doors", "Year", "Owner_Count"]

            self.X_train[self.ordinal_columns] = self.ordinal_encoder.fit_transform(self.X_train[self.ordinal_columns])
            self.X_test[self.ordinal_columns] = self.ordinal_encoder.transform(self.X_test[self.ordinal_columns])
                
            # Кодирование категориальных признаков
            self.categorical_columns = ["Brand", "Model", "Fuel_Type", "Transmission"]
            self.X_train = pd.get_dummies(self.X_train, columns=self.categorical_columns, drop_first=True, dtype="int")
            self.X_test = pd.get_dummies(self.X_test, columns=self.categorical_columns, drop_first=True, dtype="int")
                
            # Нормализация числовых признаков
            self.num_columns = ["Engine_Size", "Mileage"]
            self.scaler = MinMaxScaler()
            self.X_train[self.num_columns] = self.scaler.fit_transform(self.X_train[self.num_columns])
            self.X_test[self.num_columns] = self.scaler.transform(self.X_test[self.num_columns])
            
            self.log.info("Данные готовы для подачи в модель")
        except Exception:
            self.log.error('Ошибка подготовки данных для подачи в модель')
            self.log.error(traceback.format_exc())
            sys.exit(1)

        self.log.info("Предиктор готов!")

    def predict(self) -> bool:
        """
            Загружает модель и делает предсказание
        """
        # Передаем аргументы
        args = self.parser.parse_args()
        
        # Создаем нашу обученную модель
        model_path = self.config["RAND_FOREST"]["path"]
        try:
            model = pickle.load(open(model_path, "rb"))
        except FileNotFoundError:
            self.log.error("Ошибка: модель не найдена!")
            self.log.error(traceback.format_exc())
            sys.exit(1)
            
        # Smoke test    
        if args.tests == "smoke":
            try:
                y_pred = model.predict(self.X_test)
                r2 = r2_score(self.y_test, y_pred)
                self.log.info(f"Smoke test пройден. R²: {r2:.4f}")
            except Exception:
                self.log.error("Ошибка во время smoke теста!")
                self.log.error(traceback.format_exc())
                sys.exit(1)
        
        # Functional test                      
        elif args.tests == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            
            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:
                    try:
                        # Подгружаем данные
                        data = json.load(f)
                        X = pd.json_normalize(data, record_path=['X'])
                        y = pd.json_normalize(data, record_path=['y'])
                        
                        # Препроцессинг
                        X[self.ordinal_columns] = self.ordinal_encoder.transform(X[self.ordinal_columns])
                        X = pd.get_dummies(X, columns=self.categorical_columns, drop_first=True, dtype="int")
                        X[self.num_columns] = self.scaler.transform(X[self.num_columns])
                        
                        # Предикт и результаты
                        y_pred = model.predict(X)
                        r2 = r2_score(y, y_pred)
                        self.log.info(f"Func test {test} пройден. R²: {r2:.4f}")
                        
                        # Cохраняем эксперементальные данные
                        exp_data = {
                            "model": 'RAND_FOREST',
                            "model params": dict(self.config.items("RAND_FOREST")),
                            "tests": args.tests,
                            "R2": f"{r2:.4f}",
                            "X_test path": self.config["SPLIT_DATA"]["X_test"],
                            "y_test path": self.config["SPLIT_DATA"]["y_test"],
                        }
                        
                        # Создаем папку эксперимента
                        date_time = datetime.fromtimestamp(time.time())
                        str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                        exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                        os.makedirs(exp_dir, exist_ok=True)
                        
                        # Сохраняем YAML с результатами тестов
                        with open(os.path.join(exp_dir,"exp_config.yaml"), 'w') as exp_f:
                            yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                            
                        # Копируем лог-файл и модель
                        shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir,"exp_logfile.log"))
                        shutil.copy(model_path, os.path.join(exp_dir, 'exp_RAND_FOREST.sav'))
                    
                    except Exception:
                        self.log.error(f"Ошибка во время func теста: {test}!")
                        self.log.error(traceback.format_exc())
                        sys.exit(1)
                        
        return True


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()

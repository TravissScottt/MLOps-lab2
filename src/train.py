import configparser
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import r2_score
from joblib import dump
import sys
import traceback

from logger import Logger

SHOW_LOG = True


class ForestModel():
    def __init__(self) -> None:
        # Создаем объекты логера и конфигуратора,
        # и считываем конфигурацию
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        
        # Загружаем данные
        try:
            self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
            self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
            self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
            self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)
            self.log.info("Данные загружены успешно")
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        try:    
            # Кодируем порядковые признаки
            ordinal_encoder = OrdinalEncoder()
            ordinal_columns = ["Doors", "Year", "Owner_Count"]
            self.X_train[ordinal_columns] = ordinal_encoder.fit_transform(self.X_train[ordinal_columns])
            self.X_test[ordinal_columns] = ordinal_encoder.transform(self.X_test[ordinal_columns])
                
            # Кодирование категориальных признаков
            categorical_columns = ["Brand", "Model", "Fuel_Type", "Transmission"]
            self.X_train = pd.get_dummies(self.X_train, columns=categorical_columns, drop_first=True, dtype="int")
            self.X_test = pd.get_dummies(self.X_test, columns=categorical_columns, drop_first=True, dtype="int")
                
            # Нормализация числовых признаков
            num_columns = ["Engine_Size", "Mileage"]
            self.scaler = MinMaxScaler()
            self.X_train[num_columns] = self.scaler.fit_transform(self.X_train[num_columns])
            self.X_test[num_columns] = self.scaler.transform(self.X_test[num_columns])
            
            self.log.info("Данные готовы для подачи в модель")                    
        except Exception:
            self.log.error('Ошибка подготовки данных для подачи в модель')
            self.log.error(traceback.format_exc())
            sys.exit(1)

        # Создаем пути
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.rand_forest_path = os.path.join(self.project_path, "rand_forest.sav")

        self.log.info("Модель готова!")

    def train_forest(self, use_config: bool, n_trees=500, criterion="poisson", max_depth=20, min_samples_leaf=2, predict=False) -> bool:
        """
            Обучение модели RandomForestRagressor
        """
        # Получаем параметры
        if use_config:
            try:
                n_trees = self.config.getint("RAND_FOREST", "n_estimators")
                criterion = self.config["RAND_FOREST"]["criterion"]
                max_depth = self.config.getint("RAND_FOREST", "max_depth")
                min_samples_leaf = self.config.getint("RAND_FOREST", "min_samples_leaf")
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning('Ошибка при загрузке параметров из config.ini')
                sys.exit(1)
         
        # Создаем модель        
        model = RandomForestRegressor(
            n_estimators=n_trees,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # Обучение    
        try:
            model.fit(self.X_train, self.y_train)
            self.log.info("Модель Random Forest обучена успешно")
        except Exception:
            self.log.error("Ошибка при обучении модели")
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        # Рассчитываем метрику, если нужно
        if predict:
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            self.log.info(f" R2 Score: {r2:.4f}")
        
        # Сохраняем модель и параметры    
        params = {
            "n_estimators": n_trees,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "path": self.rand_forest_path
        }
        
        return self.save_model(model, self.rand_forest_path, "RAND_FOREST", params)

    def save_model(self, model, path: str, name: str, params: dict) -> bool:
        """
            Сохранение модели и параметров в config.ini
        """
        self.config[name] = params
        # os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
            
        # with open(path, 'wb') as f:
        #     pickle.dump(model, f)
        
        # Сохраняем модель с компрессией
        dump(model, path, compress=3)

        self.log.info(f'Модель сохранена в {path}')
        
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = ForestModel()
    multi_model.train_forest(use_config=False, predict=True)

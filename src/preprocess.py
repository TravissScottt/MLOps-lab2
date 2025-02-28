import configparser
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import traceback

from logger import Logger

TEST_SIZE = 0.2 # Доля тестовой выборки
SHOW_LOG = True # Отображать ли логи в консоли


class DataMaker():

    def __init__(self) -> None:
        # Создаем объекты логера и конфигуратора
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        
        # Пути к файлам
        self.project_path = os.path.join(os.getcwd(), "data")
        self.data_path = os.path.join(self.project_path, "car_price_dataset.csv")
        self.X_path = os.path.join(self.project_path, "Car_X.csv")
        self.y_path = os.path.join(self.project_path, "Car_y.csv")
        self.train_path = [
            os.path.join(self.project_path, "Train_Car_X.csv"),
            os.path.join(self.project_path, "Train_Car_y.csv")
        ]
        self.test_path = [
            os.path.join(self.project_path, "Test_Car_X.csv"),
            os.path.join(self.project_path, "Test_Car_y.csv")
        ]
        
        # Препроцессоры
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        
        self.log.info("DataMaker is ready")

    def get_data(self) -> bool:
        """
            Выполняет предобработку данных и сохраняет (X,y) в файлы
        """
        try:
            dataset = pd.read_csv(self.data_path)
            self.log.info(f"Dataset loaded from {self.data_path}")
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        # Кодируем порядковые признаки
        dataset["Doors"] = self.label_encoder.fit_transform(dataset["Doors"])
        dataset["Owner_Count"] = self.label_encoder.fit_transform(dataset["Owner_Count"])

        # Кодируем номинальные признаки (One-Hot Encoding)
        columns_encode = ["Brand", "Model", "Year", "Fuel_Type", "Transmission"]
        dataset = pd.get_dummies(dataset, columns=columns_encode, drop_first=True, dtype="int")
        
        # Нормализуем числовые признаки
        num_columns = ["Engine_Size", "Mileage"]
        dataset[num_columns] = self.scaler.fit_transform(dataset[num_columns])
        
        # Разделяем X и y и сохраняем по файлам
        X = dataset.drop(["Price"], axis=1)
        y = dataset["Price"]
        
        X.to_csv(self.X_path, index=True)
        y.to_csv(self.y_path, index=True)
        
        if os.path.isfile(self.X_path) and os.path.isfile(self.y_path):
            self.log.info("X and y data is ready")
            self.config["DATA"] = {'X_data': self.X_path, 'y_data': self.y_path}
            return True
        else:
            self.log.error("X and y data is not ready")
            return False

    def split_data(self, test_size=TEST_SIZE) -> bool:
        """
            Разделяет данные на train/test и сохраняет в файлы
        """
        
        self.get_data()
        try:
            X = pd.read_csv(self.X_path, index_col=0)
            y = pd.read_csv(self.y_path, index_col=0)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        # Разделяем данные на train/test    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        
        # Сохраняем train/test
        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])
        
        self.config["SPLIT_DATA"] = {
            'X_train': self.train_path[0],
            'y_train': self.train_path[1],
            'X_test': self.test_path[0],
            'y_test': self.test_path[1]
        }
        
        self.log.info("Train and test data is ready")
        
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
            
        return os.path.isfile(self.train_path[0]) and \
            os.path.isfile(self.train_path[1]) and \
            os.path.isfile(self.test_path[0]) and \
            os.path.isfile(self.test_path[1])

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        """
            Сохраняет данные в CSV
        """
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()

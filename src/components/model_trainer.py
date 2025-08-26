import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_models

@dataclass
class modelTrainerConfig:
    training_model_file_Path = os.path.join("artifacts", "model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            train_X, train_y, test_X, test_y = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "KNeighbors Regressor" : KNeighborsRegressor(),
                "DecisionTree Regressor" : DecisionTreeRegressor(),
                "RandomForest Regressor" : RandomForestRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressot" : AdaBoostRegressor(),
                "GradientBoosting Regressor" : GradientBoostingRegressor()
            }

            model_Report:dict = evaluate_models(X_train = train_X, y_train = train_y, X_test = test_X, y_test = test_y, models = models)
            
            
            best_model_score = max(sorted(model_Report.values()))

            best_model_name = list(model_Report.keys())[
                list(model_Report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6 : 
                raise CustomException("No best model found")
                
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.training_model_file_Path,
                obj = best_model
            )

            logging.info("Model is saved to file")

            return best_model_score


        except Exception as e:
            raise CustomException(e, sys)

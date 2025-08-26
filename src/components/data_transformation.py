import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.dataTransformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try: 
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            numerical_columns = ['writing_score', 'reading_score']

            input_features_train_df = train_df.drop(target_column_name, axis=1)

            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(target_column_name, axis=1)

            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_features_train_array = preprocessor_obj.fit_transform(input_features_train_df)

            input_features_test_array = preprocessor_obj.transform(input_features_test_df)

            train_array = np.c_[input_features_train_array
                                , np.array(target_feature_train_df)]
            
            test_array = np.c_[input_features_test_array
                               , np.array(target_feature_test_df)]
            

            save_object(
                obj = preprocessor_obj,
                file_path = self.dataTransformation_config.preprocesser_obj_file_path
            )
            

            return(
                train_array, 
                test_array,
                self.dataTransformation_config.preprocesser_obj_file_path
            )


        except Exception as ex:
            raise CustomException(ex, sys)
            
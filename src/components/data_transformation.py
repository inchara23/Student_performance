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

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=['writing score','reading score']
            categorical_columns=[
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info("Numerical columns encoding completed")
            logging.info("categorical columns encoding completed")

            proprocessor=ColumnTransformer(
                transformers=[
                    ("num",num_pipeline,numerical_columns),
                    ("cat",cat_pipeline,categorical_columns),
                ] 
            )
            return preprocessor
        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")
            raise CustomException(f"Error in get_data_transformer_object: {str(e)}")
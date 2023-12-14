import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        region: str,
        education_level,
        income: str,
        early_health_issue: str,
        person: int,
        amount: int):

        self.gender = gender

        self.region = region

        self.education_level = education_level

        self.income = income

        self.early_health_issue = early_health_issue

        self.person = person

        self.amount = amount

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "region": [self.region],
                "education_level": [self.education_level],
                "income": [self.income],
                "early_health_issue": [self.early_health_issue],
                "person": [self.person],
                "amount": [self.amount],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlProject.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    

    def save_results(self):
        test_data = joblib.load(self.config.test_data_path)
        model = joblib.load(self.config.model_path)



        X_test, y_test =(test_data[:,:-1],
                test_data[:,-1])
        

        predicted_qualities = model.predict(X_test)


        (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)


        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=scores)

        


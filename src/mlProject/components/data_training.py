import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
import numpy as np
from mlProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = joblib.load(self.config.train_data_path)
        test_data = joblib.load(self.config.test_data_path)


        X_train, y_train, X_test, y_test = (
        train_data[:,:-1],
        train_data[:,-1],
        test_data[:,:-1],
        test_data[:,-1]
            )


        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(X_train, y_train)


        

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))



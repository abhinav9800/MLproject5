import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import joblib
from sklearn.model_selection import train_test_split
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig
import os


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

    def get_data_transformation(self):
        
        try:
            logger.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['type']
            numerical_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']



 ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            cat_pipeline =  Pipeline(
    steps=[
          ('imputer', SimpleImputer(strategy='most_frequent')),
          ('onehotencoder', OneHotEncoder(categories='auto', drop='first')),
          ('scaler', StandardScaler(with_mean=False))
     ]
 )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline, categorical_cols)
            ])
            


            return preprocessor

        
            

            
            
        
        except Exception as e:
            logger.info("Exception occured in the initiate_datatransformation")

            pass
        


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)


        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        preprocessing_obj = self.get_data_transformation()

        
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        train_x_arr = preprocessing_obj.fit_transform(train_x)

        test_x_arr = preprocessing_obj.transform(test_x)


       

        joblib.dump(train_x_arr , os.path.join(self.config.root_dir, "train_arr.joblib"))
        joblib.dump(test_x_arr, os.path.join(self.config.root_dir, "test_arr.joblib"))


        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)


        




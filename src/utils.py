import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score


## Define a function to save the transformations and model in pickle format.
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for key, classifier in models.items():

            ## Train model
            classifier.fit(X_train, y_train)

            ## Predict Testing Data
            y_test_pred = classifier.predict(X_test)
            # y_train_pred = classifier.predict(X_train)

            ## Get Performance Scores for train and test data:
            # train_model_accuracy_score = accuracy_score(y_train, y_train_pred)
            test_model_accuracy_score = f1_score(y_test, y_test_pred)

            report[key] = test_model_accuracy_score

            return report

    except Exception as e:
        logging.info("Model Evaluation Error Occured!")
        raise CustomException(e, sys)
    
def print_classification_report(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    return classification_report(y_test, y_pred)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file=file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object functionalities!')
        raise CustomException(e, sys)
    
##############################################################################################
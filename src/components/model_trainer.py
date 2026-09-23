import os
import sys
from dataclasses import dataclass

import shap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class EnsemblePredictWrapper:
    """
    A wrapper class for the ensemble's predict_proba to make the 
    prediction function pickleable for the SHAP explainer.
    """
    def __init__(self, model):
        self.model = model
        
    def __call__(self, x):
        return self.model.predict_proba(x)[:, 1]

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    shap_explainer_file_path: str = os.path.join("artifacts", "shap_explainer.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Initializing XGBoost and Random Forest base models")
            xgb_clf = XGBClassifier(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.1, 
                random_state=42, 
                eval_metric='logloss'
            )
            rf_clf = RandomForestClassifier(
                n_estimators=100, 
                max_depth=4, 
                random_state=42
            )

            logging.info("Creating and training the ensemble VotingClassifier")
            ensemble = VotingClassifier(
                estimators=[('xgb', xgb_clf), ('rf', rf_clf)],
                voting='soft'
            )
            
            ensemble.fit(X_train, y_train)
            
            logging.info("Predicting on test set and calculating accuracy")
            predicted = ensemble.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            logging.info(f"Model Accuracy on Test Set: {accuracy:.4f}")

            logging.info("Initializing SHAP KernelExplainer")
            # Using shap.kmeans for background data to accelerate SHAP computation
            background_summary = shap.kmeans(X_train, 50)
            predict_wrapper = EnsemblePredictWrapper(ensemble)
            explainer = shap.KernelExplainer(predict_wrapper, background_summary)

            logging.info("Saving the model and explainer")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=ensemble
            )
            
            save_object(
                file_path=self.model_trainer_config.shap_explainer_file_path,
                obj=explainer
            )

            logging.info("Model training component completed successfully")
            
            return accuracy
        
        except Exception as e:
            raise CustomException(e, sys)

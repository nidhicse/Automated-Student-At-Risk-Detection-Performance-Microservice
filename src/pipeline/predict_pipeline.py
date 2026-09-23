import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.utils import load_object

# We must import the wrapper class so that joblib can find it when unpickling the shap explainer
from src.components.model_trainer import EnsemblePredictWrapper

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            explainer_path = os.path.join("artifacts", "shap_explainer.pkl")
            
            # Load artifacts
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            explainer = load_object(file_path=explainer_path)
            
            # Apply transformation
            data_scaled = preprocessor.transform(features)
            
            # Predict the probability using the ensemble model
            probability = float(model.predict_proba(data_scaled)[0, 1])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(data_scaled)
            feature_names = preprocessor.get_feature_names_out()
            
            if isinstance(shap_values, list):
                instance_shap = shap_values[1][0]
            else:
                instance_shap = shap_values[0]
                
            # Create a horizontal bar chart of feature importances
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort features by absolute value for a clean waterfall-like look
            idx = np.argsort(np.abs(instance_shap))
            
            # Set colors: Red for pushing risk higher, Blue for pushing risk lower
            colors = np.where(instance_shap[idx] > 0, 'crimson', 'dodgerblue')
            
            ax.barh(feature_names[idx], instance_shap[idx], color=colors)
            ax.set_title("SHAP Feature Contributions to At-Risk Prediction")
            ax.set_xlabel("Impact on Model Output")
            plt.tight_layout()
            
            result = {
                "at_risk_probability": round(probability, 4),
                "shap_plot": fig
            }
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        math_score: int,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "math_score": [self.math_score],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

import sys
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def train(self):
        try:
            print(">>> Starting Data Ingestion")
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()

            print(">>> Starting Data Transformation")
            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            print(">>> Starting Model Training")
            trainer = ModelTrainer()
            accuracy = trainer.initiate_model_trainer(train_arr, test_arr)
            
            print(f">>> Training Complete! Model Accuracy: {accuracy}")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.train()

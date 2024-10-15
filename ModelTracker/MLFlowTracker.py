from Dataset.dataset import Dataset
from mlflow.models import infer_signature
from Utils.utility import inferModel
import mlflow, mlflow.experiments

def trainAndLog(dataset : Dataset, trainer, experimentName, datasetName, modelName, tags : dict = None):
    """
    Manages the training of the model within an MLFlow run by 
    logging information, training parameters, and evaluation metrics.
    """

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    
    if not mlflow.get_experiment_by_name(experimentName):
        mlflow.create_experiment(experimentName)

    mlflow.set_experiment(experimentName)
    
    with mlflow.start_run():
        # tags log
        if tags is not None:
            for title, tag in tags.items():
                mlflow.set_tag(title, tag)

        # dataset logs
        rawdata = mlflow.data.from_pandas(dataset.getDataset(), name = datasetName)
        mlflow.log_input(rawdata, context="training")
        
        # Search for and log hyperparameters
        trainer.findBestParams()
        mlflow.log_params(trainer.getParams())

        # model training
        trainer.run()

        # metrics log
        mlflow.log_metrics(trainer.getMetrics())

        # Register the trained model and its information
        X_test = trainer.getX()
        model = trainer.getModel()
        modelInfo = mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = "Model_Info",
            signature = infer_signature(X_test, model.predict(X_test)),
            input_example = X_test,
            registered_model_name = modelName,
        )

        # Create and log a predictions file as an artifact
        y_test = trainer.getY()
        inferModel(dataset, modelInfo, X_test, y_test)
        mlflow.log_artifact('ModelTracker/Utils/predictions.csv', "Predictions_Test")
    mlflow.end_run()
    return None
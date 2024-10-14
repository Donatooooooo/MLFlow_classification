from Dataset.dataset import Dataset
from mlflow.models import infer_signature
from Utils.utility import inferModel
import mlflow, mlflow.experiments

def trainAndLog(dataset : Dataset, trainer, experimentName, datasetName, modelName, tags : dict = None):
    """
    Gestisce l'addestramento del modello all'interno di un run di MLFlow registrando informazioni,
    parametri di addestramento e metriche di valutazione.
    """
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    
    if not mlflow.get_experiment_by_name(experimentName):
        mlflow.create_experiment(experimentName)

    mlflow.set_experiment(experimentName)
    
    with mlflow.start_run():
        # log dei tag
        if tags is not None:
            for title, tag in tags.items():
                mlflow.set_tag(title, tag)

        # log del dataset usato per l'addestramento
        rawdata = mlflow.data.from_pandas(dataset.getDataset(), name = datasetName)
        mlflow.log_input(rawdata, context="training")
        
        # ricerca e log degli iperparametri
        trainer.findBestParams()
        mlflow.log_params(trainer.getParams())

        # fase di addestramento
        trainer.run()

        # log delle metriche
        mlflow.log_metrics(trainer.getMetrics())

        # registra il modello addestrato e le informazioni
        X_test = trainer.getX()
        model = trainer.getModel()
        modelInfo = mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = "Model_Info",
            signature = infer_signature(X_test, model.predict(X_test)),
            input_example = X_test,
            registered_model_name = modelName,
        )

        # crea e registra un file di previsioni come artifact
        y_test = trainer.getY()
        inferModel(dataset, modelInfo, X_test, y_test)
        mlflow.log_artifact('ModelTracker/Utils/predictions.csv', "Predictions_Test")
    mlflow.end_run()
    return None
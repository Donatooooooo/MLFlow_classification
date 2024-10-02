import mlflow.experiments
from Dataset.dataset import Dataset
from classifier import ModelTrainerClass
from mlflow.models import infer_signature
from kmeans import kMeans
import mlflow, pandas as pd

def preprocessing(dataset : Dataset):
    dataset.dropDatasetColumns(["id"])
    dataset.replaceBoolean("M", "B")
    for column in dataset.getDataset().columns:
        dataset.normalizeColumn(column)
    
    features = dataset.getDataFrame(['radius_mean', 'texture_mean', 'perimeter_mean'])
    kmeans = kMeans().clustering(features)
    dataset.addDatasetColumn('Appearance Cluster', kmeans.fit_predict(features))
    dataset.dropDatasetColumns(columnsToRemove=['radius_mean', 'texture_mean', 'perimeter_mean'])
    dataset.normalizeColumn('Appearance Cluster')
    return dataset

def makePredictionsArtifact(dataset : Dataset, modelInfo, X_test, y_test):
    loadedModel = mlflow.pyfunc.load_model(modelInfo.model_uri)
    predictions = loadedModel.predict(X_test)
    featureNames = dataset.getDataset().columns.tolist()
    result = pd.DataFrame(X_test, columns = featureNames)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions
    result[:10]
    result.to_csv('Evaluation/predictions.csv', index=False)

def main():
    dataset = Dataset("Dataset/brest_cancer.csv")
    dataset = preprocessing(dataset)

    trainer = ModelTrainerClass('diagnosis', ['diagnosis'], dataset)

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    # mlflow.create_experiment("RFClassifier v2")
    mlflow.set_experiment("RFclassifier v2")
    
    with mlflow.start_run():
        # tag che identifica il tipo di esperimento
        tag = "Random forest with kMeans"
        mlflow.set_tag("Training Info", tag)
        
        # registra il dataset usato per l'addestramento
        rawdata = mlflow.data.from_numpy(dataset.getDataset().to_numpy(), source="brest_cancer.csv")
        mlflow.log_input(rawdata, context="training")
        
        # fase di addestramento
        trainer.run()
        model = trainer.getModel()
        X_test = trainer.getX()
        y_test = trainer.getY()
        
        # log delle metriche e degli iperparametri
        mlflow.log_metrics(trainer.getMetrics())
        mlflow.log_params(trainer.getParams())
        
        # resgistra il modello addestrato e le informazioni
        modelInfo = mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = "Random_Forest_Model",
            signature = infer_signature(X_test, model.predict(X_test)),
            input_example = X_test,
            registered_model_name = tag,
        )

        # crea e registra un file di previsioni come artifact
        makePredictionsArtifact(dataset, modelInfo, X_test, y_test)
        mlflow.log_artifact('Evaluation/predictions.csv', "predictions.csv")
    mlflow.end_run()

main()
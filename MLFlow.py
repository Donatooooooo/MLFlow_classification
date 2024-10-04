import mlflow.experiments
from Dataset.dataset import Dataset
from classifier import ModelTrainerClass
from mlflow.models import infer_signature
import mlflow, pandas as pd

def makePredictionsArtifact(dataset : Dataset, modelInfo, X_test, y_test):
    loadedModel = mlflow.pyfunc.load_model(modelInfo.model_uri)
    predictions = loadedModel.predict(X_test)
    featureNames = dataset.getDataset().columns.tolist()
    result = pd.DataFrame(X_test, columns = featureNames)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions
    result.sample(100).to_csv('Evaluation/predictions.csv', index=False)

def trainAndLog(dataset : Dataset, trainer : ModelTrainerClass, experimentName, tag):
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    
    if not mlflow.get_experiment_by_name(experimentName):
        mlflow.create_experiment(experimentName)
    
    mlflow.set_experiment(experimentName)
    
    with mlflow.start_run():   
        # tag che identifica il tipo di esperimento
        mlflow.set_tag("Training Info", tag)

        # registra il dataset usato per l'addestramento
        rawdata = mlflow.data.from_numpy(dataset.getDataset().to_numpy())
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
            registered_model_name = tag,
        )

        # crea e registra un file di previsioni come artifact
        y_test = trainer.getY()
        makePredictionsArtifact(dataset, modelInfo, X_test, y_test)
        mlflow.log_artifact('Evaluation/predictions.csv', "Predictions_Test")
    mlflow.end_run()

#------------------------------------------------------------------------------------

from mlflow.tracking import MlflowClient
from datetime import datetime
import json

def convertTime(unixTime):
    return datetime.fromtimestamp(unixTime / 1000.0)

def extractInfo(tags):
    data_tags = json.loads(tags.get('mlflow.log-model.history', ''))
    flavors = data_tags[0]['flavors']
    py_version = flavors['python_function']['python_version']
    lib = str([key for key in flavors.keys() if key != 'python_function'][0])
    lib_version = flavors[lib].get(f'{lib}_version')
    return py_version, lib, lib_version

def fetchAndCreateMD(modelName, version):
    #ricerca il modello in base al nome
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{modelName}'")

    #estrapola il runID per ricercare tra le run degli esperimenti
    runID = None
    mlmodel = None
    for item in model_versions:
        if item.version == version:
            runID = item.run_id
            mlmodel = item.name
            break
    
    if runID is None or mlmodel is None:
        print("No model in Model Registry")
        return

    run = client.get_run(runID)
    
    params = run.data.params
    metrics = run.data.metrics
    
    py, lib, libv = extractInfo(run.data.tags)
    startTime = convertTime(run.info.start_time)
    endTime = convertTime(run.info.end_time)
    
    print(mlmodel, params, metrics, startTime, endTime, lib, libv, py, sep="\n")

fetchAndCreateMD("Random forest with kMeans", 21)
# from Dataset.dataset import Dataset
# from mlflow.models import infer_signature
# from Utils.utility import inferModel
# import mlflow, mlflow.experiments

# def trainAndLog(dataset : Dataset, trainer, experimentName, datasetName, modelName, tags : dict = None):
#     """
#     Gestisce l'addestramento del modello all'interno di un run di MLFlow registrando informazioni,
#     parametri di addestramento e metriche di valutazione.
#     """

#     mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    
#     if not mlflow.get_experiment_by_name(experimentName):
#         mlflow.create_experiment(experimentName)

#     mlflow.set_experiment(experimentName)
    
#     with mlflow.start_run():
#         # log dei tag
#         if tags is not None:
#             for title, tag in tags.items():
#                 mlflow.set_tag(title, tag)

#         # log del dataset usato per l'addestramento
#         rawdata = mlflow.data.from_pandas(dataset.getDataset(), name = datasetName)
#         mlflow.log_input(rawdata, context="training")
        
#         # ricerca e log degli iperparametri
#         trainer.findBestParams()
#         mlflow.log_params(trainer.getParams())

#         # fase di addestramento
#         trainer.run()

#         # log delle metriche
#         mlflow.log_metrics(trainer.getMetrics())

#         # registra il modello addestrato e le informazioni
#         X_test = trainer.getX()
#         model = trainer.getModel()
#         modelInfo = mlflow.sklearn.log_model(
#             sk_model = model,
#             artifact_path = "Model_Info",
#             signature = infer_signature(X_test, model.predict(X_test)),
#             input_example = X_test,
#             registered_model_name = modelName,
#         )

#         # crea e registra un file di previsioni come artifact
#         y_test = trainer.getY()
#         inferModel(dataset, modelInfo, X_test, y_test)
#         mlflow.log_artifact('src/Utils/predictions.csv', "Predictions_Test")
#     mlflow.end_run()
#     return None

#--------------------------------------------------------------------------------------------------------
from mlflow.tracking import MlflowClient
from jinja2 import Environment, FileSystemLoader
from Utils.utility import convertTime, extractInfoTags, extratDatasetName, getPath

def fetchData(modelName, version):
    """
    Rintraccia le informazioni riguardante un modello attraverso il suo nome e la specifica versione
    memorizzata in MLflow Model Registry. Ottenuta la run corrispondente, rintraccia le informazioni. 
    """

    # ricerca il modello in base al nome
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{modelName}'")

    # estrapola runID e nome in base alla versione
    runID = None
    mlmodel = None
    for item in model_versions:
        if int(item.version) == version:
            runID = item.run_id
            mlmodel = item.name
            break

    if runID is None or mlmodel is None:
        raise ValueError("No model in Model Registry")

    # attraverso la run, estrapola le informazioni
    run = client.get_run(runID)

    py, lib, libv = extractInfoTags(run.data.tags)
    datasetName = extratDatasetName(run.inputs.dataset_inputs)
    startTime = convertTime(run.info.start_time)
    endTime = convertTime(run.info.end_time)

    data = {
        "modelName": mlmodel,
        "version": version,
        "author": run.info.user_id,
        "modelType": mlmodel,
        "library": lib,
        "libraryVersion": libv,
        "pythonVersion": py,
        "datasetName": datasetName,
        "parameters": run.data.params,
        "startTime": startTime,
        "endTime": endTime,
        "evaluations": run.data.metrics 
    }

    return data

def ModelCard(modelName, version):
    """
    Crea una Model Card del modello instanziando un template 
    predefinito attraverso le informazioni rintracciate.   
    """

    try:
        data = fetchData(modelName, version)
    except Exception as e:
        print(e)
        return

    environment = Environment(loader = FileSystemLoader("src/Templates"))
    modelcard_template = environment.get_template("modelCard_template.md")
    instance = modelcard_template.render(data)

    with open(getPath(data), 'w') as file:
        file.write(instance)

    return None


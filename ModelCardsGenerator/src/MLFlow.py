from mlflow.tracking import MlflowClient
from jinja2 import Environment, FileSystemLoader
from Utils.utility import convertTime, extractInfoTags
from Utils.utility import extratDatasetName, getPath
import os

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

    environment = Environment(loader = FileSystemLoader("ModelCardsGenerator/src/Utils/Templates"))
    modelcard_template = environment.get_template("modelCard_template.md")
    instance = modelcard_template.render(data)

    path = getPath(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as file:
        file.write(instance)

    return None
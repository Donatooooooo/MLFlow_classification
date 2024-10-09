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
        mlflow.log_artifact('src/Utils/predictions.csv', "Predictions_Test")
    mlflow.end_run()
    return None

#--------------------------------------------------------------------------------------------------------
from mlflow.tracking import MlflowClient
from Utils.utility import convertTime, extractInfoTags, extratDatasetName, organize

def fetchInfo(modelName, version):
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
    
    params = run.data.params
    metrics = run.data.metrics
    userID = run.info.user_id
    py, lib, libv = extractInfoTags(run.data.tags)
    datasetName = extratDatasetName(run.inputs.dataset_inputs)
    startTime = convertTime(run.info.start_time)
    endTime = convertTime(run.info.end_time)
    
    return [mlmodel, version, userID, lib, libv, py, 
                datasetName, params, startTime, endTime, metrics]


def createMD(modelName, version):
    """
    Crea un file markdown con le informazioni rintracciate.   
    """
    
    try:
        info = fetchInfo(modelName, version)
    except Exception as e:
        print(e)
        return

    title = f"# {info[0]} - v{info[1]}\n"

    general_info = (
        f"## General Information \n"
        f"- Developed by: {info[2]}\n"
        f"- Model Type: {info[0]}\n"
        f"- {info[3]}: {info[4]}\n"
        f"- Python Version: {info[5]}\n"
    )

    params = organize("Parameters:", info[7])

    training_info = (
        f"## Training Details\n"
        f"- Dataset: {info[6]}\n"
        f"- {params}\n"
        f"- Training started at: {info[8]}\n"
        f"- Training ended at: {info[9]}\n"
    )
    
    eval_info = organize("## Evaluation", info[10])
    
    part = info[0].replace(" ", "")
    fname = f"{part}_v{info[1]}.md"
    path = f"ModelCards/{fname}"
    with open(path, 'w') as file:
        file.write(f"{title}{general_info}{training_info}{eval_info}")

    print(f"{fname} saved")
    return None
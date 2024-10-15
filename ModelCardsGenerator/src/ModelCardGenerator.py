from mlflow.tracking import MlflowClient
from Utils.exceptions import NoModelException
from Utils.utility import convertTime, extractInfoTags
from Utils.utility import extratDatasetName, getPath, templateRender
from Utils.logger import Logger
import os, sys

def fetchData(modelName, version):
    """
    Trace information about a model using its name and the specific version stored in the 
    MLflow Model Registry. Once the corresponding run is obtained, retrieve the relevant information.
    """

    # Search for the model by name
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{modelName}'")

    # Extract runID and name based on the version
    runID = None
    mlmodel = None
    for item in model_versions:
        if int(item.version) == version:
            runID = item.run_id
            mlmodel = item.name
            break

    output = Logger()
    if runID is None or mlmodel is None:
        raise NoModelException()
    
    # Extract information through the run
    run = client.get_run(runID)
    params = run.data.params
    author = run.info.user_id
    metrics = run.data.metrics
    py, lib, libv = extractInfoTags(run.data.tags)
    startTime = convertTime(run.info.start_time)
    endTime = convertTime(run.info.end_time)
    datasetName = extratDatasetName(run.inputs.dataset_inputs)
    
    if not params:
        output.log("Missing parameters in Model Card generation")
    if not metrics:
        output.log("Missing metrics in Model Card generation")
    if not datasetName:
        output.log("Missing dataset information in Model Card generation")

    data = {
        "modelName": mlmodel,
        "version": version,
        "author": author,
        "modelType": mlmodel,
        "library": lib,
        "libraryVersion": libv,
        "pythonVersion": py,
        "datasetName": datasetName,
        "parameters": params,
        "startTime": startTime,
        "endTime": endTime,
        "evaluations": metrics,
    }

    return data, output


def ModelCard(modelName, version):
    """
    Create a model card by instantiating a predefined 
    template using the retrieved information.
    """
    try:
        data, output = fetchData(modelName, version)
    except NoModelException as e:
        output = Logger()
        output.log(f"Check the model name or version: {str(e)}")
        return output

    instance = templateRender("modelCard_template.md", data)

    path = getPath(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as modelCard:
        modelCard.write(instance)
    
    return output

if __name__ == "__main__":
    output = Logger()
    try:
        input = sys.argv[1]
        parts = input.rsplit(' ', 1)
        output = ModelCard(parts[0], int(parts[1]))
    except IndexError as e:
        output.log("Check the model name or version: invalid arguments")
    except ValueError as e:
        output.log("Check version format: incorrect model version")
    except Exception as e:
        output.log(f"Exception caused by: {str(e)}")
    finally:
        output.display()
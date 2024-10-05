from datetime import datetime
from Dataset.dataset import Dataset
import mlflow, json, pandas as pd
from util.kmeans import kMeans

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

def convertTime(unixTime):
    return datetime.fromtimestamp(unixTime/1000.0).strftime('%H:%M:%S %Y-%m-%d')

def extractInfoTags(tags):   
    data_tags = json.loads(tags.get('mlflow.log-model.history', ''))
    flavors = data_tags[0]['flavors']
    py_version = flavors['python_function']['python_version']
    lib = str([key for key in flavors.keys() if key != 'python_function'][0])
    lib_version = flavors[lib].get(f'{lib}_version')
    return py_version, lib, lib_version

def extratDatasetName(data):
    dataString = str(data)
    start = dataString.find("name='") + len("name='")
    end = dataString.find("'", start)
    return dataString[start:end]

def inferModel(dataset : Dataset, modelInfo, X_test, y_test):
    loadedModel = mlflow.pyfunc.load_model(modelInfo.model_uri)
    predictions = loadedModel.predict(X_test)
    featureNames = dataset.getDataset().columns.tolist()
    result = pd.DataFrame(X_test, columns = featureNames)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions
    result.sample(100).to_csv('Evaluation/predictions.csv', index=False)
    
def organize(text, info):
    string = f"{text}\n\n"
    for key, value in info.items():
        string += (f"   - `{key}` {value}\n\n")
    return string
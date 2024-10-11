from datetime import datetime
import json

# def preprocessing(dataset : Dataset, cluster = False):
#     dataset.dropDatasetColumns(["id"])
#     dataset.replaceBoolean("M", "B")
#     for column in dataset.getDataset().columns:
#         dataset.normalizeColumn(column)
    
#     if cluster:
#         features = dataset.getDataFrame(['radius_mean', 'texture_mean', 'perimeter_mean'])
#         kmeans = kMeans().clustering(features)
#         dataset.addDatasetColumn('Appearance Cluster', kmeans.fit_predict(features))
#         dataset.dropDatasetColumns(columnsToRemove=['radius_mean', 'texture_mean', 'perimeter_mean'])
#         dataset.normalizeColumn('Appearance Cluster')
        
#     dataset.dropDatasetColumns(['Unnamed: 32'])
#     return dataset

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

# def inferModel(dataset : Dataset, modelInfo, X_test, y_test):
#     loadedModel = mlflow.pyfunc.load_model(modelInfo.model_uri)
#     predictions = loadedModel.predict(X_test)
#     featureNames = dataset.getDataset().columns.tolist()
#     result = pd.DataFrame(X_test, columns = featureNames)
    
#     if y_test.ndim > 1 and y_test.shape[1] > 1:
#         y_test = np.argmax(y_test, axis=1)
        
#     if predictions.ndim > 1 and predictions.shape[1] > 1:
#         predictions = np.argmax(predictions, axis=1)
    
#     result["actual_class"] = y_test
#     result["predicted_class"] = predictions
#     result.sample(100).to_csv('src/Utils/predictions.csv', index=False)


import os

import os

def getPath(data):
    # Ottieni il nome del modello e la versione
    part = data.get("modelName").replace(" ", "")
    fname = f"{part}_v{data.get('version')}.md"
    
    # Ottieni il percorso assoluto della root del progetto
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Un livello sopra rispetto a src/
    
    # Combina il percorso della root con la cartella 'ModelCards'
    model_cards_dir = os.path.join(root_dir, 'ModelCards')  # Cartella ModelCards nella root del progetto
    
    # Crea il percorso completo del file
    path = os.path.join(model_cards_dir, fname)
    
    return path


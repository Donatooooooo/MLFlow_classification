from Utils.kmeans import kMeans
from Dataset.dataset import Dataset
import mlflow, numpy as np, pandas as pd

def preprocessing(dataset : Dataset, cluster = False):
    dataset.dropDatasetColumns(["id"])
    dataset.replaceBoolean("M", "B")
    for column in dataset.getDataset().columns:
        dataset.normalizeColumn(column)
    
    if cluster:
        features = dataset.getDataFrame(['radius_mean', 'texture_mean', 'perimeter_mean'])
        kmeans = kMeans().clustering(features)
        dataset.addDatasetColumn('Appearance Cluster', kmeans.fit_predict(features))
        dataset.dropDatasetColumns(columnsToRemove=['radius_mean', 'texture_mean', 'perimeter_mean'])
        dataset.normalizeColumn('Appearance Cluster')
        
    dataset.dropDatasetColumns(['Unnamed: 32'])
    return dataset

def inferModel(dataset : Dataset, modelInfo, X_test, y_test):
    loadedModel = mlflow.pyfunc.load_model(modelInfo.model_uri)
    predictions = loadedModel.predict(X_test)
    featureNames = dataset.getDataset().columns.tolist()
    result = pd.DataFrame(X_test, columns = featureNames)
    
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
        
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    result["actual_class"] = y_test
    result["predicted_class"] = predictions
    result.sample(100).to_csv('ModelTracker/Utils/predictions.csv', index=False)
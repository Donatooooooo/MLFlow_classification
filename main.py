from Dataset.dataset import Dataset
from classifier import *
from kmeans import kMeans

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

dataset = Dataset("Dataset/brest_cancer.csv")
dataset = preprocessing(dataset)

trainer = ModelTrainerClass('diagnosis', ['diagnosis'], dataset)
trainer.run()
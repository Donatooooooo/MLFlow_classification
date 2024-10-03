from Dataset.dataset import Dataset
from classifier import ModelTrainerClass
from kmeans import kMeans
from MLFlow import trainAndLog

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
trainAndLog(dataset, trainer, "RFclassifier v2", "Random forest with kMeans")
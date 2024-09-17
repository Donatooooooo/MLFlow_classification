from Dataset.dataset import Dataset
from classifier import *

def preprocessing(dataset : Dataset):
    dataset.dropDatasetColumns(["id"])
    dataset.replaceBoolean("M", "B")
    for column in dataset.getDataset().columns:
        dataset.normalizeColumn(column)
    return dataset

dataset = Dataset("Dataset/brest_cancer.csv")
dataset = preprocessing(dataset)

trainer = ModelTrainerClass('diagnosis', ['diagnosis'], dataset)
trainer.run()